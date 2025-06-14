"""
Unified Code Context MCP Server
A Model Context Protocol server that provides tools to:
1. List imports from code files
2. Add code to Neo4j graph with dependency tracking (with background processing)
3. Find relevant code snippets based on queries
4. Track job progress for long-running operations
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import ast
import importlib
import uuid
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.request import urlopen
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import requests

from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j connection parameters from environment variables
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

def debug_log(message):
    """Write debug message to a file"""
    debug_file = os.path.expanduser("~/mcp_debug.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(debug_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
        f.flush()

class JobStatus(Enum):
    """Job status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class JobInfo:
    """Data class for job information"""
    job_id: str
    status: JobStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    total_files: int = 0
    processed_files: int = 0
    current_file: Optional[str] = None
    estimated_duration: Optional[float] = None
    actual_duration: Optional[float] = None
    errors: List[str] = None
    result: Optional[Dict[str, Any]] = None
    path: Optional[str] = None
    is_dependency: bool = False

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage"""
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100

    @property
    def estimated_time_remaining(self) -> Optional[float]:
        """Calculate estimated time remaining"""
        if self.status != JobStatus.RUNNING or self.processed_files == 0:
            return None
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        avg_time_per_file = elapsed / self.processed_files
        remaining_files = self.total_files - self.processed_files
        return remaining_files * avg_time_per_file

class JobManager:
    """Manager for background jobs"""
    
    def __init__(self):
        self.jobs: Dict[str, JobInfo] = {}
        self.lock = threading.Lock()
    
    def create_job(self, path: str, is_dependency: bool = False) -> str:
        """Create a new job and return job ID"""
        job_id = str(uuid.uuid4())
        
        with self.lock:
            self.jobs[job_id] = JobInfo(
                job_id=job_id,
                status=JobStatus.PENDING,
                start_time=datetime.now(),
                path=path,
                is_dependency=is_dependency
            )
        
        return job_id
    
    def update_job(self, job_id: str, **kwargs):
        """Update job information"""
        with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                for key, value in kwargs.items():
                    if hasattr(job, key):
                        setattr(job, key, value)
    
    def get_job(self, job_id: str) -> Optional[JobInfo]:
        """Get job information"""
        with self.lock:
            return self.jobs.get(job_id)
    
    def list_jobs(self) -> List[JobInfo]:
        """List all jobs"""
        with self.lock:
            return list(self.jobs.values())
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up jobs older than specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with self.lock:
            jobs_to_remove = [
                job_id for job_id, job in self.jobs.items()
                if job.end_time and job.end_time < cutoff_time
            ]
            
            for job_id in jobs_to_remove:
                del self.jobs[job_id]

class ImportExtractor:
    """Module for extracting imports from different programming languages"""
    
    @staticmethod
    def extract_python_imports(file_path: str) -> Set[str]:
        """Extract imports from a Python file"""
        imports = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            patterns = [
                r'^\s*import\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)',
                r'^\s*from\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import',
            ]
            
            for line in content.split('\n'):
                for pattern in patterns:
                    match = re.match(pattern, line.strip())
                    if match:
                        import_name = match.group(1).split('.')[0]
                        imports.add(import_name)
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
        
        return imports

    @staticmethod
    def extract_javascript_imports(file_path: str) -> Set[str]:
        """Extract imports from JavaScript/TypeScript files"""
        imports = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            patterns = [
                r'import.*?from\s+[\'"]([^\'\"]+)[\'"]',
                r'require\s*\(\s*[\'"]([^\'\"]+)[\'"]\s*\)',
                r'import\s*\(\s*[\'"]([^\'\"]+)[\'"]\s*\)',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    pkg_name = match.split('/')[0] if not match.startswith('@') else '/'.join(match.split('/')[:2])
                    if not match.startswith('.'):
                        imports.add(pkg_name)
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
        
        return imports

    @staticmethod
    def extract_java_imports(file_path: str) -> Set[str]:
        """Extract imports from Java files"""
        imports = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            pattern = r'import\s+(?:static\s+)?([a-zA-Z_][a-zA-Z0-9_.]*)'
            matches = re.findall(pattern, content)
            
            for match in matches:
                pkg_parts = match.split('.')
                if len(pkg_parts) >= 2:
                    imports.add(f"{pkg_parts[0]}.{pkg_parts[1]}")
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
        
        return imports

class CodeVisitor(ast.NodeVisitor):
    """AST visitor to extract code elements with enhanced metadata"""
    
    def __init__(self, file_path: str, is_dependency: bool = False):
        self.file_path = file_path
        self.is_dependency = is_dependency
        self.functions = []
        self.classes = []
        self.variables = []
        self.imports = []
        self.function_calls = []
        self.current_context = None
        self.current_class = None
    
    def visit_FunctionDef(self, node):
        """Visit function definitions"""
        func_data = {
            'name': node.name,
            'line_number': node.lineno,
            'end_line': node.end_lineno if hasattr(node, 'end_lineno') else None,
            'args': [arg.arg for arg in node.args.args],
            'source': ast.unparse(node) if hasattr(ast, 'unparse') else '',
            'context': self.current_context,
            'class_context': self.current_class,
            'is_dependency': self.is_dependency,
            'docstring': ast.get_docstring(node),
            'decorators': [ast.unparse(dec) if hasattr(ast, 'unparse') else '' for dec in node.decorator_list]
        }
        
        prev_context = self.current_context
        self.current_context = node.name
        self.functions.append(func_data)
        
        self.generic_visit(node)
        self.current_context = prev_context
    
    def visit_ClassDef(self, node):
        """Visit class definitions"""
        class_data = {
            'name': node.name,
            'line_number': node.lineno,
            'end_line': node.end_lineno if hasattr(node, 'end_lineno') else None,
            'bases': [ast.unparse(base) if hasattr(ast, 'unparse') else '' for base in node.bases],
            'source': ast.unparse(node) if hasattr(ast, 'unparse') else '',
            'context': self.current_context,
            'is_dependency': self.is_dependency,
            'docstring': ast.get_docstring(node),
            'decorators': [ast.unparse(dec) if hasattr(ast, 'unparse') else '' for dec in node.decorator_list]
        }
        
        prev_context = self.current_context
        prev_class = self.current_class
        self.current_context = node.name
        self.current_class = node.name
        self.classes.append(class_data)
        
        self.generic_visit(node)
        
        self.current_context = prev_context
        self.current_class = prev_class
    
    def visit_Assign(self, node):
        """Visit variable assignments"""
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_data = {
                    'name': target.id,
                    'line_number': node.lineno,
                    'value': ast.unparse(node.value) if hasattr(ast, 'unparse') else '',
                    'context': self.current_context,
                    'class_context': self.current_class,
                    'is_dependency': self.is_dependency
                }
                self.variables.append(var_data)
        
        self.generic_visit(node)
    
    def visit_Import(self, node):
        """Visit import statements"""
        for name in node.names:
            import_data = {
                'name': name.name,
                'line_number': node.lineno,
                'alias': name.asname,
                'context': self.current_context,
                'is_dependency': self.is_dependency
            }
            self.imports.append(import_data)
        
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Visit from-import statements"""
        for name in node.names:
            import_data = {
                'name': f"{node.module}.{name.name}" if node.module else name.name,
                'line_number': node.lineno,
                'alias': name.asname,
                'context': self.current_context,
                'is_dependency': self.is_dependency
            }
            self.imports.append(import_data)
        
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Visit function calls"""
        if isinstance(node.func, ast.Name):
            call_data = {
                'name': node.func.id,
                'line_number': node.lineno,
                'args': [ast.unparse(arg) if hasattr(ast, 'unparse') else '' for arg in node.args],
                'context': self.current_context,
                'is_dependency': self.is_dependency
            }
            self.function_calls.append(call_data)
        elif isinstance(node.func, ast.Attribute):
            call_data = {
                'name': f"{ast.unparse(node.func.value) if hasattr(ast, 'unparse') else ''}.{node.func.attr}",
                'line_number': node.lineno,
                'args': [ast.unparse(arg) if hasattr(ast, 'unparse') else '' for arg in node.args],
                'context': self.current_context,
                'is_dependency': self.is_dependency
            }
            self.function_calls.append(call_data)
        
        self.generic_visit(node)

class GraphBuilder:
    """Module for building and managing Neo4j graphs"""
    
    def __init__(self, neo4j_uri: str = None, neo4j_user: str = None, neo4j_password: str = None):
        # Validate credentials
        self.neo4j_uri = neo4j_uri or NEO4J_URI
        self.neo4j_user = neo4j_user or NEO4J_USER
        self.neo4j_password = neo4j_password or NEO4J_PASSWORD
        
        if not all([self.neo4j_uri, self.neo4j_user, self.neo4j_password]):
            raise ValueError("Neo4j credentials must be provided via environment variables: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD")
        
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        self.create_schema()
    
    def create_schema(self):
        """Create constraints and indexes in Neo4j"""
        with self.driver.session() as session:
            try:
                # Create constraints
                session.run("CREATE CONSTRAINT repository_name IF NOT EXISTS FOR (r:Repository) REQUIRE r.name IS UNIQUE")
                session.run("CREATE CONSTRAINT file_path IF NOT EXISTS FOR (f:File) REQUIRE f.path IS UNIQUE")
                session.run("CREATE CONSTRAINT function_unique IF NOT EXISTS FOR (f:Function) REQUIRE (f.name, f.file_path, f.line_number) IS UNIQUE")
                session.run("CREATE CONSTRAINT class_unique IF NOT EXISTS FOR (c:Class) REQUIRE (c.name, c.file_path, c.line_number) IS UNIQUE")
                
                # Create indexes
                session.run("CREATE INDEX file_name IF NOT EXISTS FOR (f:File) ON (f.name)")
                session.run("CREATE INDEX function_name IF NOT EXISTS FOR (f:Function) ON (f.name)")
                session.run("CREATE INDEX class_name IF NOT EXISTS FOR (c:Class) ON (c.name)")
                session.run("CREATE INDEX variable_name IF NOT EXISTS FOR (v:Variable) ON (v.name)")
                session.run("CREATE INDEX module_name IF NOT EXISTS FOR (m:Module) ON (m.name)")
                
                # Create indexes for is_dependency on specific node types
                session.run("CREATE INDEX function_dependency IF NOT EXISTS FOR (f:Function) ON (f.is_dependency)")
                session.run("CREATE INDEX class_dependency IF NOT EXISTS FOR (c:Class) ON (c.is_dependency)")
                session.run("CREATE INDEX file_dependency IF NOT EXISTS FOR (f:File) ON (f.is_dependency)")
                session.run("CREATE INDEX variable_dependency IF NOT EXISTS FOR (v:Variable) ON (v.is_dependency)")
                session.run("CREATE INDEX repository_dependency IF NOT EXISTS FOR (r:Repository) ON (r.is_dependency)")
            except Exception as e:
                logger.warning(f"Schema creation warning: {e}")
    
    def estimate_processing_time(self, path: Path) -> Tuple[int, float]:
        """Estimate processing time and file count"""
        if path.is_file() and path.suffix == '.py':
            python_files = [path]
        else:
            python_files = list(path.glob("**/*.py"))
        
        total_files = len(python_files)
        
        # Estimate based on file size and complexity
        # Base time: 0.1 seconds per file
        # Additional time based on file size: 0.001 seconds per KB
        estimated_time = total_files * 0.1
        
        for file_path in python_files[:10]:  # Sample first 10 files for size estimation
            try:
                file_size_kb = file_path.stat().st_size / 1024
                estimated_time += file_size_kb * 0.001
            except:
                continue
        
        # Scale up based on sample
        if total_files > 10:
            estimated_time = estimated_time * (total_files / min(10, total_files))
        
        return total_files, estimated_time
    
    def parse_python_file(self, file_path: Path, is_dependency: bool = False) -> Dict:
        """Parse a Python file and extract code elements"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            visitor = CodeVisitor(str(file_path), is_dependency)
            visitor.visit(tree)
            
            return {
                'file_path': str(file_path),
                'functions': visitor.functions,
                'classes': visitor.classes,
                'variables': visitor.variables,
                'imports': visitor.imports,
                'function_calls': visitor.function_calls,
                'is_dependency': is_dependency
            }
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return {
                'file_path': str(file_path),
                'functions': [],
                'classes': [],
                'variables': [],
                'imports': [],
                'function_calls': [],
                'is_dependency': is_dependency,
                'error': str(e)
            }
    
    def add_repository_to_graph(self, repo_path: Path, is_dependency: bool = False):
        """Add repository to Neo4j graph"""
        repo_name = repo_path.name
        
        with self.driver.session() as session:
            session.run("""
                MERGE (r:Repository {name: $name})
                SET r.path = $path, r.is_dependency = $is_dependency
            """, name=repo_name, path=str(repo_path), is_dependency=is_dependency)
    
    def add_file_to_graph(self, file_data: Dict, repo_name: str):
        """Add file and its elements to Neo4j graph"""
        file_path = file_data['file_path']
        file_name = Path(file_path).name
        is_dependency = file_data.get('is_dependency', False)
        
        with self.driver.session() as session:
            # Create file node
            session.run("""
                MATCH (r:Repository {name: $repo_name})
                MERGE (f:File {path: $path})
                SET f.name = $name, f.is_dependency = $is_dependency
                MERGE (r)-[:CONTAINS]->(f)
            """, repo_name=repo_name, path=file_path, name=file_name, is_dependency=is_dependency)
            
            # Add functions
            for func in file_data['functions']:
                session.run("""
                    MATCH (f:File {path: $file_path})
                    MERGE (func:Function {name: $name, file_path: $file_path, line_number: $line_number})
                    SET func.end_line = $end_line,
                        func.args = $args,
                        func.source = $source,
                        func.context = $context,
                        func.class_context = $class_context,
                        func.is_dependency = $is_dependency,
                        func.docstring = $docstring,
                        func.decorators = $decorators
                    MERGE (f)-[:CONTAINS]->(func)
                """, file_path=file_path, **func)
            
            # Add classes
            for cls in file_data['classes']:
                session.run("""
                    MATCH (f:File {path: $file_path})
                    MERGE (c:Class {name: $name, file_path: $file_path, line_number: $line_number})
                    SET c.end_line = $end_line,
                        c.bases = $bases,
                        c.source = $source,
                        c.context = $context,
                        c.is_dependency = $is_dependency,
                        c.docstring = $docstring,
                        c.decorators = $decorators
                    MERGE (f)-[:CONTAINS]->(c)
                """, file_path=file_path, **cls)
            
            # Add variables
            for var in file_data['variables']:
                session.run("""
                    MATCH (f:File {path: $file_path})
                    MERGE (v:Variable {name: $name, file_path: $file_path, line_number: $line_number})
                    SET v.value = $value,
                        v.context = $context,
                        v.class_context = $class_context,
                        v.is_dependency = $is_dependency
                    MERGE (f)-[:CONTAINS]->(v)
                """, file_path=file_path, **var)
            
            # Add imports
            for imp in file_data['imports']:
                session.run("""
                    MATCH (f:File {path: $file_path})
                    MERGE (m:Module {name: $name})
                    SET m.alias = $alias, m.is_dependency = $is_dependency
                    MERGE (f)-[:IMPORTS]->(m)
                """, file_path=file_path, **imp)
    
    async def build_graph_from_path_async(self, path: Path, is_dependency: bool = False, job_manager: JobManager = None, job_id: str = None) -> Dict[str, Any]:
        """Build Neo4j graph from a given path asynchronously with progress tracking"""
        try:
            # Update job status to running
            if job_manager and job_id:
                job_manager.update_job(job_id, status=JobStatus.RUNNING)
            
            # Add repository
            self.add_repository_to_graph(path, is_dependency)
            
            # Find Python files
            if path.is_file() and path.suffix == '.py':
                python_files = [path]
            else:
                python_files = list(path.glob("**/*.py"))
            
            total_files = len(python_files)
            processed_files = 0
            errors = []
            
            # Update job with total files
            if job_manager and job_id:
                job_manager.update_job(job_id, total_files=total_files)
            
            for file_path in python_files:
                try:
                    # Update current file being processed
                    if job_manager and job_id:
                        job_manager.update_job(job_id, current_file=str(file_path))
                    
                    file_data = self.parse_python_file(file_path, is_dependency)
                    self.add_file_to_graph(file_data, path.name)
                    processed_files += 1
                    
                    # Update progress
                    if job_manager and job_id:
                        job_manager.update_job(job_id, processed_files=processed_files)
                    
                    # Small delay to allow other operations
                    await asyncio.sleep(0.001)
                    
                except Exception as e:
                    error_msg = f"Error processing {file_path}: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            result = {
                "success": True,
                "repository": path.name,
                "path": str(path),
                "is_dependency": is_dependency,
                "processed_files": processed_files,
                "total_files": total_files,
                "errors": errors
            }
            
            # Update job as completed
            if job_manager and job_id:
                job_manager.update_job(
                    job_id, 
                    status=JobStatus.COMPLETED, 
                    end_time=datetime.now(),
                    result=result,
                    errors=errors
                )
            
            return result
        
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "repository": path.name if path else "unknown"
            }
            
            # Update job as failed
            if job_manager and job_id:
                job_manager.update_job(
                    job_id, 
                    status=JobStatus.FAILED, 
                    end_time=datetime.now(),
                    result=error_result,
                    errors=[str(e)]
                )
            
            return error_result
    
    def build_graph_from_path(self, path: Path, is_dependency: bool = False) -> Dict[str, Any]:
        """Build Neo4j graph from a given path (synchronous version for backward compatibility)"""
        return asyncio.run(self.build_graph_from_path_async(path, is_dependency))
    
    def close(self):
        """Close Neo4j driver"""
        self.driver.close()

class CodeFinder:
    """Module for finding relevant code snippets based on queries"""
    
    def __init__(self, neo4j_uri: str = None, neo4j_user: str = None, neo4j_password: str = None):
        # Use provided credentials or fall back to environment variables
        self.neo4j_uri = neo4j_uri or NEO4J_URI
        self.neo4j_user = neo4j_user or NEO4J_USER
        self.neo4j_password = neo4j_password or NEO4J_PASSWORD
        
        if not all([self.neo4j_uri, self.neo4j_user, self.neo4j_password]):
            raise ValueError("Neo4j credentials must be provided via environment variables: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD")
        
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
    
    def find_by_function_name(self, search_term: str) -> List[Dict]:
        """Find functions by name matching"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (f:Function)
                WHERE f.name CONTAINS $search_term OR f.name =~ $regex_pattern
                RETURN f.name as name, f.file_path as file_path, f.line_number as line_number,
                       f.source as source, f.docstring as docstring, f.is_dependency as is_dependency
                ORDER BY f.is_dependency ASC, f.name
                LIMIT 20
            """, search_term=search_term, regex_pattern=f"(?i).*{re.escape(search_term)}.*")
            
            return [dict(record) for record in result]
    
    def find_by_class_name(self, search_term: str) -> List[Dict]:
        """Find classes by name matching"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Class)
                WHERE c.name CONTAINS $search_term OR c.name =~ $regex_pattern
                RETURN c.name as name, c.file_path as file_path, c.line_number as line_number,
                       c.source as source, c.docstring as docstring, c.is_dependency as is_dependency
                ORDER BY c.is_dependency ASC, c.name
                LIMIT 20
            """, search_term=search_term, regex_pattern=f"(?i).*{re.escape(search_term)}.*")
            
            return [dict(record) for record in result]
    
    def find_by_content(self, search_term: str) -> List[Dict]:
        """Find code by content matching in source or docstrings"""
        with self.driver.session() as session:
            # Search in functions
            func_result = session.run("""
                MATCH (f:Function)
                WHERE f.source CONTAINS $search_term OR f.docstring CONTAINS $search_term
                RETURN 'function' as type, f.name as name, f.file_path as file_path, 
                       f.line_number as line_number, f.source as source, 
                       f.docstring as docstring, f.is_dependency as is_dependency
                ORDER BY f.is_dependency ASC, f.name
                LIMIT 10
            """, search_term=search_term)
            
            # Search in classes
            class_result = session.run("""
                MATCH (c:Class)
                WHERE c.source CONTAINS $search_term OR c.docstring CONTAINS $search_term
                RETURN 'class' as type, c.name as name, c.file_path as file_path,
                       c.line_number as line_number, c.source as source,
                       c.docstring as docstring, c.is_dependency as is_dependency
                ORDER BY c.is_dependency ASC, c.name
                LIMIT 10
            """, search_term=search_term)
            
            results = []
            results.extend([dict(record) for record in func_result])
            results.extend([dict(record) for record in class_result])
            
            return results
    
    def find_related_code(self, user_query: str) -> Dict[str, Any]:
        """Find code related to a query using multiple search strategies"""
        results = {
            "query": user_query,
            "functions_by_name": self.find_by_function_name(user_query),
            "classes_by_name": self.find_by_class_name(user_query),
            "content_matches": self.find_by_content(user_query)
        }
        
        # Calculate relevance scores
        all_results = []
        
        for func in results["functions_by_name"]:
            func["search_type"] = "function_name"
            func["relevance_score"] = 0.9 if not func["is_dependency"] else 0.7
            all_results.append(func)
        
        for cls in results["classes_by_name"]:
            cls["search_type"] = "class_name"
            cls["relevance_score"] = 0.8 if not cls["is_dependency"] else 0.6
            all_results.append(cls)
        
        for content in results["content_matches"]:
            content["search_type"] = "content"
            content["relevance_score"] = 0.6 if not content["is_dependency"] else 0.4
            all_results.append(content)
        
        # Sort by relevance
        all_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        results["ranked_results"] = all_results[:15]
        results["total_matches"] = len(all_results)
        
        return results
    
    def close(self):
        """Close Neo4j driver"""
        self.driver.close()

class MCPServer:
    """Main MCP Server class with all tools"""
    
    def __init__(self):
        # Validate Neo4j credentials on startup
        if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
            raise ValueError(
                "Neo4j credentials must be set via environment variables:\n"
                "- NEO4J_URI\n"
                "- NEO4J_USER\n"
                "- NEO4J_PASSWORD\n"
                "Please configure these in your MCP settings."
            )
        
        self.graph_builder = GraphBuilder()
        self.code_finder = CodeFinder()
        self.import_extractor = ImportExtractor()
        self.job_manager = JobManager()
    
    def get_local_package_path(self, package_name: str) -> Optional[str]:
        """Get the local installation path of a Python package"""
        try:
            debug_log(f"Getting local path for package: {package_name}")
            
            # Try to import the package
            module = importlib.import_module(package_name)
            
            # Get the module file path
            if hasattr(module, '__file__') and module.__file__:
                module_file = module.__file__
                debug_log(f"Module file: {module_file}")
                
                # For packages, get the directory containing __init__.py
                if module_file.endswith('__init__.py'):
                    package_path = os.path.dirname(module_file)
                else:
                    # For single modules, get the directory containing the .py file
                    package_path = os.path.dirname(module_file)
                
                debug_log(f"Package path: {package_path}")
                return package_path
            
            # If no __file__ attribute, try to get path from __path__ (for namespace packages)
            elif hasattr(module, '__path__'):
                # __path__ is a list for namespace packages
                if isinstance(module.__path__, list) and module.__path__:
                    package_path = module.__path__[0]
                    debug_log(f"Package path from __path__: {package_path}")
                    return package_path
                else:
                    package_path = str(module.__path__)
                    debug_log(f"Package path from __path__ (str): {package_path}")
                    return package_path
            
            debug_log(f"Could not determine path for {package_name}")
            return None
            
        except ImportError as e:
            debug_log(f"Could not import {package_name}: {e}")
            return None
        except Exception as e:
            debug_log(f"Error getting local path for {package_name}: {e}")
            return None
        
    def __init_tools(self):
        """Initialize tool definitions"""
        self.tools = {
            "list_imports": {
                "name": "list_imports",
                "description": "Extract all package imports from code files in a directory or file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to file or directory to analyze"
                        },
                        "language": {
                            "type": "string", 
                            "description": "Programming language (python, javascript, typescript, java, etc.)",
                            "default": "python"
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "Whether to analyze subdirectories recursively",
                            "default": True
                        }
                    },
                    "required": ["path"]
                }
            },
            "add_code_to_graph": {
                "name": "add_code_to_graph",
                "description": "Add code from a local folder to Neo4j graph with dependency tracking. Returns immediately with job ID for background processing.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to directory or file to add to graph"
                        },
                        "is_dependency": {
                            "type": "boolean",
                            "description": "Whether this code is from a dependency (true) or current project (false)",
                            "default": False
                        }
                    },
                    "required": ["path"]
                }
            },
            "add_package_to_graph": {
                "name": "add_package_to_graph",
                "description": "Add a Python package (installed via pip) to Neo4j graph by automatically discovering its location. Returns immediately with job ID for background processing.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "package_name": {
                            "type": "string",
                            "description": "Name of the Python package to add (e.g., 'subprocess', 'requests', 'numpy')"
                        },
                        "is_dependency": {
                            "type": "boolean",
                            "description": "Whether this package is a dependency (true) or part of your project (false)",
                            "default": True
                        }
                    },
                    "required": ["package_name"]
                }
            },
            "check_job_status": {
                "name": "check_job_status",
                "description": "Check the status and progress of a background job",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "Job ID returned from add_code_to_graph or add_package_to_graph"
                        }
                    },
                    "required": ["job_id"]
                }
            },
            "list_jobs": {
                "name": "list_jobs",
                "description": "List all background jobs and their current status",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            },
            "find_code": {
                "name": "find_code",
                "description": "Find relevant code snippets related to a keyword (or phrase), which can be a function name, class name, or content present in the source code",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A keyword (or phrase), which can be a function name, class name, or content present in the source code"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    def list_imports_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool to list all imports from code files"""
        path = args.get('path', '.')
        language = args.get('language', 'python').lower()
        recursive = args.get('recursive', True)
        
        all_imports = set()
        file_extensions = {
            'python': ['.py'],
            'javascript': ['.js', '.jsx', '.mjs'],
            'typescript': ['.ts', '.tsx'],
            'java': ['.java'],
        }
        
        extensions = file_extensions.get(language, ['.py'])
        extract_func = {
            'python': self.import_extractor.extract_python_imports,
            'javascript': self.import_extractor.extract_javascript_imports,
            'typescript': self.import_extractor.extract_javascript_imports,
            'java': self.import_extractor.extract_java_imports,
        }.get(language, self.import_extractor.extract_python_imports)
        
        try:
            path_obj = Path(path)
            
            if path_obj.is_file():
                if any(str(path_obj).endswith(ext) for ext in extensions):
                    all_imports.update(extract_func(str(path_obj)))
            elif path_obj.is_dir():
                pattern = "**/*" if recursive else "*"
                for ext in extensions:
                    for file_path in path_obj.glob(f"{pattern}{ext}"):
                        if file_path.is_file():
                            all_imports.update(extract_func(str(file_path)))
            else:
                return {"error": f"Path {path} does not exist"}
            
            # Filter out standard library modules for Python
            if language == 'python':
                stdlib_modules = {
                    'os', 'sys', 'json', 'time', 'datetime', 'math', 'random', 're',
                    'collections', 'itertools', 'functools', 'operator', 'pathlib',
                    'urllib', 'http', 'logging', 'threading', 'multiprocessing',
                    'asyncio', 'typing', 'dataclasses', 'enum', 'abc', 'io', 'csv',
                    'sqlite3', 'pickle', 'base64', 'hashlib', 'hmac', 'secrets',
                    'unittest', 'doctest', 'pdb', 'profile', 'cProfile', 'timeit'
                }
                all_imports = all_imports - stdlib_modules
            
            return {
                "imports": sorted(list(all_imports)),
                "language": language,
                "path": path,
                "count": len(all_imports)
            }
        
        except Exception as e:
            return {"error": f"Failed to analyze imports: {str(e)}"}
    
    def add_code_to_graph_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool to add code to Neo4j graph with background processing"""
        path = args.get('path')
        is_dependency = args.get('is_dependency', False)
        
        try:
            path_obj = Path(path)
            if not path_obj.exists():
                return {"error": f"Path {path} does not exist"}
            
            # Estimate processing time and file count
            total_files, estimated_time = self.graph_builder.estimate_processing_time(path_obj)
            
            # Create job
            job_id = self.job_manager.create_job(path, is_dependency)
            
            # Update job with estimation
            self.job_manager.update_job(
                job_id, 
                total_files=total_files, 
                estimated_duration=estimated_time
            )
            
            # Start background processing
            asyncio.create_task(
                self.graph_builder.build_graph_from_path_async(
                    path_obj, is_dependency, self.job_manager, job_id
                )
            )
            
            debug_log(f"Started background job {job_id} for path: {path}, is_dependency: {is_dependency}")
            
            return {
                "success": True,
                "job_id": job_id,
                "message": f"Background processing started for {path}",
                "estimated_files": total_files,
                "estimated_duration_seconds": round(estimated_time, 2),
                "estimated_duration_human": f"{int(estimated_time // 60)}m {int(estimated_time % 60)}s" if estimated_time >= 60 else f"{int(estimated_time)}s",
                "instructions": f"Use 'check_job_status' with job_id '{job_id}' to monitor progress"
            }
        
        except Exception as e:
            debug_log(f"Error creating background job: {str(e)}")
            return {"error": f"Failed to start background processing: {str(e)}"}
    
    def add_package_to_graph_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool to add a Python package to Neo4j graph by auto-discovering its location"""
        package_name = args.get('package_name')
        is_dependency = args.get('is_dependency', True)
        
        try:
            # Get the package path
            package_path = self.get_local_package_path(package_name)
            
            if not package_path:
                return {
                    "error": f"Could not find package '{package_name}'. Make sure it's installed and importable."
                }
            
            if not os.path.exists(package_path):
                return {
                    "error": f"Package path '{package_path}' does not exist"
                }
            
            path_obj = Path(package_path)
            
            # Estimate processing time and file count
            total_files, estimated_time = self.graph_builder.estimate_processing_time(path_obj)
            
            # Create job
            job_id = self.job_manager.create_job(package_path, is_dependency)
            
            # Update job with estimation
            self.job_manager.update_job(
                job_id, 
                total_files=total_files, 
                estimated_duration=estimated_time
            )
            
            # Start background processing
            asyncio.create_task(
                self.graph_builder.build_graph_from_path_async(
                    path_obj, is_dependency, self.job_manager, job_id
                )
            )
            
            debug_log(f"Started background job {job_id} for package: {package_name} at {package_path}, is_dependency: {is_dependency}")
            
            return {
                "success": True,
                "job_id": job_id,
                "package_name": package_name,
                "discovered_path": package_path,
                "message": f"Background processing started for package '{package_name}'",
                "estimated_files": total_files,
                "estimated_duration_seconds": round(estimated_time, 2),
                "estimated_duration_human": f"{int(estimated_time // 60)}m {int(estimated_time % 60)}s" if estimated_time >= 60 else f"{int(estimated_time)}s",
                "instructions": f"Use 'check_job_status' with job_id '{job_id}' to monitor progress"
            }
        
        except Exception as e:
            debug_log(f"Error creating background job for package {package_name}: {str(e)}")
            return {"error": f"Failed to start background processing for package '{package_name}': {str(e)}"}
    
    def check_job_status_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool to check job status"""
        job_id = args.get('job_id')
        
        try:
            job = self.job_manager.get_job(job_id)
            
            if not job:
                return {"error": f"Job {job_id} not found"}
            
            # Convert job to dictionary for JSON serialization
            job_dict = asdict(job)
            
            # Add human-readable status
            if job.status == JobStatus.RUNNING:
                if job.estimated_time_remaining:
                    remaining = job.estimated_time_remaining
                    job_dict["estimated_time_remaining_human"] = (
                        f"{int(remaining // 60)}m {int(remaining % 60)}s" 
                        if remaining >= 60 else f"{int(remaining)}s"
                    )
                
                if job.start_time:
                    elapsed = (datetime.now() - job.start_time).total_seconds()
                    job_dict["elapsed_time_human"] = (
                        f"{int(elapsed // 60)}m {int(elapsed % 60)}s" 
                        if elapsed >= 60 else f"{int(elapsed)}s"
                    )
            
            elif job.status == JobStatus.COMPLETED and job.start_time and job.end_time:
                duration = (job.end_time - job.start_time).total_seconds()
                job_dict["actual_duration_human"] = (
                    f"{int(duration // 60)}m {int(duration % 60)}s" 
                    if duration >= 60 else f"{int(duration)}s"
                )
            
            # Format timestamps for readability
            job_dict["start_time"] = job.start_time.strftime("%Y-%m-%d %H:%M:%S")
            if job.end_time:
                job_dict["end_time"] = job.end_time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Convert enum to string
            job_dict["status"] = job.status.value
            
            return {
                "success": True,
                "job": job_dict
            }
        
        except Exception as e:
            debug_log(f"Error checking job status: {str(e)}")
            return {"error": f"Failed to check job status: {str(e)}"}
    
    def list_jobs_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool to list all jobs"""
        try:
            jobs = self.job_manager.list_jobs()
            
            # Convert jobs to dictionaries for JSON serialization
            jobs_data = []
            for job in jobs:
                job_dict = asdict(job)
                job_dict["status"] = job.status.value
                job_dict["start_time"] = job.start_time.strftime("%Y-%m-%d %H:%M:%S")
                if job.end_time:
                    job_dict["end_time"] = job.end_time.strftime("%Y-%m-%d %H:%M:%S")
                jobs_data.append(job_dict)
            
            # Sort by start time (newest first)
            jobs_data.sort(key=lambda x: x["start_time"], reverse=True)
            
            return {
                "success": True,
                "jobs": jobs_data,
                "total_jobs": len(jobs_data)
            }
        
        except Exception as e:
            debug_log(f"Error listing jobs: {str(e)}")
            return {"error": f"Failed to list jobs: {str(e)}"}
    
    def find_code_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool to find relevant code snippets"""
        user_query = args.get('query', '')
        
        try:
            debug_log(f"Finding code for query: {user_query}")
            results = self.code_finder.find_related_code(user_query)
            
            return {
                "success": True,
                "query": user_query,
                "results": results
            }
        
        except Exception as e:
            debug_log(f"Error finding code: {str(e)}")
            return {"error": f"Failed to find code: {str(e)}"}
    
    async def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool calls"""
        if tool_name == "list_imports":
            return self.list_imports_tool(args)
        elif tool_name == "add_code_to_graph":
            return self.add_code_to_graph_tool(args)
        elif tool_name == "add_package_to_graph":
            return self.add_package_to_graph_tool(args)
        elif tool_name == "check_job_status":
            return self.check_job_status_tool(args)
        elif tool_name == "list_jobs":
            return self.list_jobs_tool(args)
        elif tool_name == "find_code":
            return self.find_code_tool(args)
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    
    async def run_server(self):
        """Run the MCP server"""
        # Initialize tools after credentials are validated
        self.__init_tools()
        
        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                
                request = json.loads(line.strip())
                method = request.get('method')
                params = request.get('params', {})
                request_id = request.get('id')
                
                if method == 'initialize':
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {
                                "tools": {"listTools": True}
                            },
                            "serverInfo": {
                                "name": "unified-code-context-server",
                                "version": "2.0.0"
                            }
                        }
                    }
                elif method == 'tools/list':
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"tools": list(self.tools.values())}
                    }
                elif method == 'tools/call':
                    tool_name = params.get('name')
                    args = params.get('arguments', {})
                    result = await self.handle_tool_call(tool_name, args)
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
                    }
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": f"Method not found: {method}"}
                    }
                
                print(json.dumps(response))
                sys.stdout.flush()
                
            except Exception as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": request.get('id') if 'request' in locals() else None,
                    "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
                }
                print(json.dumps(error_response))
                sys.stdout.flush()
    
    def __del__(self):
        """Cleanup when server is destroyed"""
        try:
            self.graph_builder.close()
            self.code_finder.close()
        except:
            pass

if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO)
        server = MCPServer()
        asyncio.run(server.run_server())
    except ValueError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    finally:
        try:
            if 'server' in locals():
                server.graph_builder.close()
                server.code_finder.close()
        except:
            pass