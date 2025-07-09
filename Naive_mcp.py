#!/usr/bin/env python3
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
import torch
import numpy as np
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j connection parameters from environment variables
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
HF_TOKEN = os.getenv("HF_TOKEN")
JS_PARSER = str(Path(__file__).parent / "js_parser.js")

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
    """Enhanced AST visitor to extract code elements with better function call detection"""
    
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
        self.context_stack = []  # Track nested contexts
    
    def _push_context(self, name: str, node_type: str):
        """Push a new context onto the stack"""
        self.context_stack.append({
            'name': name,
            'type': node_type,
            'previous_context': self.current_context,
            'previous_class': self.current_class
        })
        self.current_context = name
        if node_type == 'class':
            self.current_class = name
    
    def _pop_context(self):
        """Pop the current context from the stack"""
        if self.context_stack:
            prev_context = self.context_stack.pop()
            self.current_context = prev_context['previous_context']
            self.current_class = prev_context['previous_class']
    
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
        
        self.functions.append(func_data)
        
        # Push function context
        self._push_context(node.name, 'function')
        
        # Visit child nodes
        self.generic_visit(node)
        
        # Pop function context
        self._pop_context()
    
    def visit_AsyncFunctionDef(self, node):
        """Visit async function definitions"""
        # Handle async functions the same way as regular functions
        self.visit_FunctionDef(node)
    
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
        
        self.classes.append(class_data)
        
        # Push class context
        self._push_context(node.name, 'class')
        
        # Visit child nodes
        self.generic_visit(node)
        
        # Pop class context
        self._pop_context()
    
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
            elif isinstance(target, ast.Attribute):
                # Handle attribute assignments like self.var = value
                var_data = {
                    'name': target.attr,
                    'line_number': node.lineno,
                    'value': ast.unparse(node.value) if hasattr(ast, 'unparse') else '',
                    'context': self.current_context,
                    'class_context': self.current_class,
                    'is_dependency': self.is_dependency
                }
                self.variables.append(var_data)
        
        self.generic_visit(node)
    
    def visit_AnnAssign(self, node):
        """Visit annotated assignments (type hints)"""
        if isinstance(node.target, ast.Name):
            var_data = {
                'name': node.target.id,
                'line_number': node.lineno,
                'value': ast.unparse(node.value) if node.value and hasattr(ast, 'unparse') else '',
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
        """Visit function calls with enhanced detection for static methods and object methods"""
        call_name = None
        call_args = []
        full_call_name = None  # Store the full call name for better matching
        
        # Extract arguments
        try:
            call_args = [ast.unparse(arg) if hasattr(ast, 'unparse') else '' for arg in node.args]
        except:
            call_args = []
        
        # Determine the function being called
        if isinstance(node.func, ast.Name):
            # Direct function call: func()
            call_name = node.func.id
            full_call_name = call_name
        elif isinstance(node.func, ast.Attribute):
            # Method call: obj.method() or module.func()
            if isinstance(node.func.value, ast.Name):
                # obj.method() or ClassName.static_method()
                object_name = node.func.value.id
                method_name = node.func.attr
                call_name = method_name
                full_call_name = f"{object_name}.{method_name}"
            elif isinstance(node.func.value, ast.Attribute):
                # nested.obj.method() like self.import_extractor.extract_python_imports()
                try:
                    base_name = ast.unparse(node.func.value) if hasattr(ast, 'unparse') else ''
                    method_name = node.func.attr
                    call_name = method_name
                    full_call_name = f"{base_name}.{method_name}"
                except:
                    call_name = node.func.attr
                    full_call_name = call_name
            else:
                call_name = node.func.attr
                full_call_name = call_name
        elif isinstance(node.func, ast.Subscript):
            # Function call with subscript: func[key]()
            try:
                call_name = ast.unparse(node.func) if hasattr(ast, 'unparse') else ''
                full_call_name = call_name
            except:
                call_name = 'subscript_call'
                full_call_name = call_name
        
        if call_name:
            # Filter out built-in functions and common patterns
            builtin_functions = {
                'print', 'len', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple',
                'range', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'sum', 'min', 'max',
                'abs', 'round', 'isinstance', 'hasattr', 'getattr', 'setattr', 'delattr',
                'type', 'super', 'property', 'staticmethod', 'classmethod', 'open', 'input'
            }
            
            # Extract just the function name for built-in check
            simple_name = call_name.split('.')[-1] if '.' in call_name else call_name
            
            if simple_name not in builtin_functions:
                call_data = {
                    'name': call_name,  # Just the method/function name
                    'full_name': full_call_name,  # Full call path like "self.import_extractor.extract_python_imports"
                    'line_number': node.lineno,
                    'args': call_args,
                    'context': self.current_context,
                    'class_context': self.current_class,
                    'is_dependency': self.is_dependency
                }
                self.function_calls.append(call_data)
        
        self.generic_visit(node)
    
    def visit_Lambda(self, node):
        """Visit lambda expressions"""
        # Lambdas can contain function calls too
        lambda_name = f"lambda_line_{node.lineno}"
        
        # Push lambda context
        self._push_context(lambda_name, 'lambda')
        
        # Visit the lambda body
        self.generic_visit(node)
        
        # Pop lambda context
        self._pop_context()
    
    def visit_ListComp(self, node):
        """Visit list comprehensions"""
        # List comprehensions can contain function calls
        comp_name = f"listcomp_line_{node.lineno}"
        
        # Push comprehension context
        self._push_context(comp_name, 'comprehension')
        
        # Visit the comprehension
        self.generic_visit(node)
        
        # Pop comprehension context
        self._pop_context()
    
    def visit_DictComp(self, node):
        """Visit dictionary comprehensions"""
        comp_name = f"dictcomp_line_{node.lineno}"
        
        self._push_context(comp_name, 'comprehension')
        self.generic_visit(node)
        self._pop_context()
    
    def visit_SetComp(self, node):
        """Visit set comprehensions"""
        comp_name = f"setcomp_line_{node.lineno}"
        
        self._push_context(comp_name, 'comprehension')
        self.generic_visit(node)
        self._pop_context()
    
    def visit_GeneratorExp(self, node):
        """Visit generator expressions"""
        gen_name = f"genexp_line_{node.lineno}"
        
        self._push_context(gen_name, 'generator')
        self.generic_visit(node)
        self._pop_context()


class GraphBuilder:
    """Module for building and managing Neo4j graphs with proper relationships"""
    
    def __init__(self, neo4j_uri: str = None, neo4j_user: str = None, neo4j_password: str = None, embedding_model = None, embedding_tokenizer = None):
        # Validate credentials
        self.neo4j_uri = neo4j_uri or NEO4J_URI
        self.neo4j_user = neo4j_user or NEO4J_USER
        self.neo4j_password = neo4j_password or NEO4J_PASSWORD
        
        if not all([self.neo4j_uri, self.neo4j_user, self.neo4j_password]):
            raise ValueError("Neo4j credentials must be provided via environment variables: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD")
        
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        self.create_schema()
        
        # For embedding
        self.embedding_tokenizer = embedding_tokenizer
        self.embedding_model = embedding_model

    def _embed_text(self, text: str):
        if self.embedding_model is None or self.embedding_tokenizer is None:
            return None
        
        inputs = self.embedding_tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            if hasattr(outputs, 'last_hidden_state'):
                emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            else:
                emb = outputs[0][:, 0, :].cpu().numpy().flatten()
        return emb.tolist()
    
    def create_schema(self):
        """Create constraints and indexes in Neo4j"""
        with self.driver.session() as session:
            try:
                # Create constraints
                session.run("CREATE CONSTRAINT repository_name IF NOT EXISTS FOR (r:Repository) REQUIRE r.name IS UNIQUE")
                session.run("CREATE CONSTRAINT file_path IF NOT EXISTS FOR (f:File) REQUIRE f.path IS UNIQUE")
                session.run("CREATE CONSTRAINT function_unique IF NOT EXISTS FOR (f:Function) REQUIRE (f.name, f.file_path, f.line_number) IS UNIQUE")
                session.run("CREATE CONSTRAINT class_unique IF NOT EXISTS FOR (c:Class) REQUIRE (c.name, c.file_path, c.line_number) IS UNIQUE")
                session.run("CREATE CONSTRAINT variable_unique IF NOT EXISTS FOR (v:Variable) REQUIRE (v.name, v.file_path, v.line_number) IS UNIQUE")
                session.run("CREATE CONSTRAINT module_name IF NOT EXISTS FOR (m:Module) REQUIRE m.name IS UNIQUE")
                
                # Create indexes for performance
                session.run("CREATE INDEX file_name IF NOT EXISTS FOR (f:File) ON (f.name)")
                session.run("CREATE INDEX function_name IF NOT EXISTS FOR (f:Function) ON (f.name)")
                session.run("CREATE INDEX class_name IF NOT EXISTS FOR (c:Class) ON (c.name)")
                session.run("CREATE INDEX variable_name IF NOT EXISTS FOR (v:Variable) ON (v.name)")
                session.run("CREATE INDEX module_name_idx IF NOT EXISTS FOR (m:Module) ON (m.name)")
                
                # Create indexes for dependency tracking
                session.run("CREATE INDEX function_dependency IF NOT EXISTS FOR (f:Function) ON (f.is_dependency)")
                session.run("CREATE INDEX class_dependency IF NOT EXISTS FOR (c:Class) ON (c.is_dependency)")
                session.run("CREATE INDEX file_dependency IF NOT EXISTS FOR (f:File) ON (f.is_dependency)")
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
        # Base time: 0.15 seconds per file (increased due to relationship creation)
        # Additional time based on file size: 0.002 seconds per KB
        estimated_time = total_files * 0.15
        
        for file_path in python_files[:10]:  # Sample first 10 files for size estimation
            try:
                file_size_kb = file_path.stat().st_size / 1024
                estimated_time += file_size_kb * 0.002
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
    
    def parse_js_file(self, file_path: Path, is_dependency: bool = False) -> Dict:
        """Parse a Python file and extract code elements"""
        try:
            result = subprocess.run(
                ['node', JS_PARSER, file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            # Parse JSON result from stdout
            analysis = json.loads(result.stdout)
            return analysis

        except subprocess.CalledProcessError as e:
            logger.error(f"Error parsing {file_path}: {e.stderr}")
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
                SET r.path = $path, 
                    r.url = $url,
                    r.is_dependency = $is_dependency
            """, name=repo_name, path=str(repo_path), url=None, is_dependency=is_dependency)
    
    def add_python_file_to_graph(self, file_data: Dict, repo_name: str):
        """Add file and its elements to Neo4j graph with proper relationships"""
        file_path = file_data['file_path']
        file_name = Path(file_path).name
        relative_path = str(Path(file_path).relative_to(Path(file_path).parent.parent))
        is_dependency = file_data.get('is_dependency', False)
        
        with self.driver.session() as session:
            # Create file node and CONTAINS relationship from Repository
            session.run("""
                MATCH (r:Repository {name: $repo_name})
                MERGE (f:File {path: $path})
                SET f.name = $name, 
                    f.relative_path = $relative_path,
                    f.is_dependency = $is_dependency
                MERGE (r)-[:CONTAINS]->(f)
            """, repo_name=repo_name, path=file_path, name=file_name, 
                relative_path=relative_path, is_dependency=is_dependency)
            
            # Add functions with CONTAINS relationships
            for func in file_data['functions']:
                code_text = func.get('source')  + f"\n{func["name"]} is defined in the {func["context"]} , {file_path}"
                embedding = self._embed_text(code_text) if code_text else None
                session.run("""
                    MATCH (f:File {path: $file_path})
                    MERGE (fn:Function {name: $name, file_path: $file_path, line_number: $line_number})
                    SET fn.end_line = $end_line,
                        fn.args = $args,
                        fn.source = $source,
                        fn.context = $context,
                        fn.is_dependency = $is_dependency,
                        fn.docstring = $docstring,
                        fn.decorators = $decorators,
                        fn.embedding = $embedding
                    MERGE (f)-[:CONTAINS]->(fn)
                """, file_path=file_path, embedding=embedding, **func)
            
            # Add classes with CONTAINS relationships
            for cls in file_data['classes']:
                code_text = cls.get('source') + f"\n{cls["name"]} is defined in the {cls["context"]} , {file_path}"
                embedding = self._embed_text(code_text) if code_text else None
                session.run("""
                    MATCH (f:File {path: $file_path})
                    MERGE (c:Class {name: $name, file_path: $file_path, line_number: $line_number})
                    SET c.end_line = $end_line,
                        c.bases = $bases,
                        c.source = $source,
                        c.context = $context,
                        c.is_dependency = $is_dependency,
                        c.docstring = $docstring,
                        c.decorators = $decorators,
                        c.embedding = $embedding
                    MERGE (f)-[:CONTAINS]->(c)
                """, file_path=file_path, embedding=embedding, **cls)
            
            # Create INHERITS_FROM relationships for classes
            for cls in file_data['classes']:
                for base_class in cls.get('bases', []):
                    if base_class and not base_class.startswith('('):  # Filter out complex expressions
                        session.run("""
                            MATCH (child:Class {name: $child_name, file_path: $file_path})
                            MATCH (parent:Class {name: $parent_name})
                            MERGE (child)-[:INHERITS_FROM]->(parent)
                        """, child_name=cls['name'], file_path=file_path, parent_name=base_class)
            
            # Add variables with CONTAINS relationships
            for var in file_data['variables']:
                session.run("""
                    MATCH (f:File {path: $file_path})
                    MERGE (v:Variable {name: $name, file_path: $file_path, line_number: $line_number})
                    SET v.value = $value,
                        v.context = $context,
                        v.is_dependency = $is_dependency
                    MERGE (f)-[:CONTAINS]->(v)
                """, file_path=file_path, **var)
                
                # Create CONTAINS relationships from functions/classes to variables
                if var.get('context'):
                    # Check if context is a function
                    session.run("""
                        MATCH (v:Variable {name: $var_name, file_path: $file_path, line_number: $var_line})
                        MATCH (fn:Function {name: $context_name, file_path: $file_path})
                        MERGE (fn)-[:CONTAINS]->(v)
                    """, var_name=var['name'], file_path=file_path, 
                        var_line=var['line_number'], context_name=var['context'])
                    
                    # Check if context is a class
                    session.run("""
                        MATCH (v:Variable {name: $var_name, file_path: $file_path, line_number: $var_line})
                        MATCH (c:Class {name: $context_name, file_path: $file_path})
                        MERGE (c)-[:CONTAINS]->(v)
                    """, var_name=var['name'], file_path=file_path,
                        var_line=var['line_number'], context_name=var['context'])
            
            # Add modules and IMPORTS relationships
            for imp in file_data['imports']:
                session.run("""
                    MATCH (f:File {path: $file_path})
                    MERGE (m:Module {name: $name})
                    SET m.alias = $alias
                    MERGE (f)-[:IMPORTS]->(m)
                """, file_path=file_path, **imp)
            
            # Create CALLS relationships between functions
            self._create_function_calls(session, file_data)
            
            # Create CONTAINS relationships for class methods
            self._create_class_method_relationships(session, file_data)
    
    def add_js_file_to_graph(self, file_data: Dict, repo_name: str):
        """Add file and its elements to Neo4j graph with proper relationships"""
        file_path = file_data['file_path']
        file_name = Path(file_path).name
        relative_path = str(Path(file_path).relative_to(Path(file_path).parent.parent))
        is_dependency = file_data.get('is_dependency', False)
        
        with self.driver.session() as session:
            # Create file node and CONTAINS relationship from Repository
            session.run("""
                MATCH (r:Repository {name: $repo_name})
                MERGE (f:File {path: $path})
                SET f.name = $name, 
                    f.relative_path = $relative_path,
                    f.is_dependency = $is_dependency
                MERGE (r)-[:CONTAINS]->(f)
            """, repo_name=repo_name, path=file_path, name=file_name, 
                relative_path=relative_path, is_dependency=is_dependency)
            
            # Add functions with CONTAINS relationships
            for func in file_data['functions']:
                code_text = f"Function docstring :- {func.get('docstring')}\n" + func.get('source') + f"\n{func["name"]} is defined in the {func["context"]} , {func["file_path"]}"
                embedding = self._embed_text(code_text) if code_text else None
                session.run("""
                    MATCH (f:File {path: $file_path})
                    MERGE (fn:Function {name: $name, file_path: $file_path, line_number: $line_number})
                    SET fn.end_line = $end_line,
                        fn.args = $args,
                        fn.source = $source,
                        fn.context = $context,
                        fn.class_context = $class_context,
                        fn.is_dependency = $is_dependency,
                        fn.docstring = $docstring,
                        fn.startColumn = $startColumn,
                        fn.endColumn = $endColumn,
                        fn.export = $is_Export,
                        fn.embedding = $embedding
                    MERGE (f)-[:CONTAINS]->(fn)
                """, file_path=file_path, embedding=embedding, **func)
            
            # Add classes with CONTAINS relationships
            for cls in file_data['classes']:
                code_text = f"Class docstring :- {cls.get('docstring')}\n" + cls.get('source') + f"\n{cls["name"]} is defined in the {cls["context"]} , {cls["file_path"]}"
                embedding = self._embed_text(code_text) if code_text else None
                session.run("""
                    MATCH (f:File {path: $file_path})
                    MERGE (c:Class {name: $name, file_path: $file_path, line_number: $line_number})
                    SET c.end_line = $end_line,
                        c.bases = $bases,
                        c.source = $source,
                        c.context = $context,
                        c.is_dependency = $is_dependency,
                        c.startColumn = $startColumn,
                        c.endColumn = $endColumn,
                        c.export = $is_Export,
                        c.embedding = $embedding
                    MERGE (f)-[:CONTAINS]->(c)
                """, file_path=file_path, embedding=embedding, **cls)
            
            # Create INHERITS_FROM relationships for classes
            for cls in file_data['classes']:
                base_class = cls.get('bases', None)
                if base_class :  # Filter out complex expressions
                    session.run("""
                        MATCH (child:Class {name: $child_name, file_path: $file_path})
                        MATCH (parent:Class {name: $parent_name})
                        MERGE (child)-[:INHERITS_FROM]->(parent)
                    """, child_name=cls['name'], file_path=file_path, parent_name=base_class)
            
            # Add variables with CONTAINS relationships
            for var in file_data['variables']:
                session.run("""
                    MATCH (f:File {path: $file_path})
                    MERGE (v:Variable {name: $name, file_path: $file_path, line_number: $line_number})
                    SET v.value = $value,
                        v.context = $context,
                        v.is_dependency = $is_dependency,
                        v.class_context = $class_context,
                        v.startColumn = $startColumn,
                        v.endColumn = $endColumn
                    MERGE (f)-[:CONTAINS]->(v)
                """, file_path=file_path, **var)
                
                # Create CONTAINS relationships from functions/classes to variables
                if var.get('context'):
                    # Check if context is a function
                    session.run("""
                        MATCH (v:Variable {name: $var_name, file_path: $file_path, line_number: $var_line})
                        MATCH (fn:Function {name: $context_name, file_path: $file_path , class_context : $class_context})
                        MERGE (fn)-[:CONTAINS]->(v)
                    """, var_name=var['name'], file_path=file_path, 
                        var_line=var['line_number'], context_name=var['context'] , class_context=var["class_context"])
                    
                    # Check if context is a class
                    session.run("""
                        MATCH (v:Variable {name: $var_name, file_path: $file_path, line_number: $var_line})
                        MATCH (c:Class {name: $context_name, file_path: $file_path})
                        MERGE (c)-[:CONTAINS]->(v)
                    """, var_name=var['name'], file_path=file_path,
                        var_line=var['line_number'], context_name=var['context'])
            
            # Add modules and IMPORTS relationships
            for imp in file_data['imports']:
                session.run("""
                    MATCH (f:File {path: $file_path})
                    MERGE (m:Module {name: $name})
                    SET m.alias = $alias,
                    m.moduleSource = $moduleSource
                    MERGE (f)-[:IMPORTS]->(m)
                """, file_path=file_path, **imp)
            
            # Create CALLS relationships between functions
            self._create_function_calls(session, file_data)
            
            # Create CONTAINS relationships for class methods
            self._create_class_method_relationships(session, file_data)
    
    def _create_function_calls(self, session, file_data: Dict):
        """Create CALLS relationships between functions based on function_calls data with improved matching"""
        file_path = file_data['file_path']
        
        
        for call in file_data.get('function_calls', []):
            caller_context = call.get('context')
            called_name = call['name']
            full_call_name = call.get('full_name', called_name)
            line_number = call['line_number']
            call_type = call.get("type" , None)
            startColumn = None
            
            if file_path.endswith(".js"):
                startColumn = call["startColumn"]
            
            # Skip built-in functions and common patterns
            if file_path.endswith(".py") and called_name in ['print', 'len', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple']:
                continue
                
            if file_path.endswith(".py") and caller_context:
                # Strategy 1: Try exact match with the called function name
                session.run("""
                    MATCH (caller:Function {name: $caller_name, file_path: $file_path})
                    MATCH (called:Function {name: $called_name})
                    MERGE (caller)-[:CALLS {line_number: $line_number, args: $args, full_call_name: $full_call_name}]->(called)
                """, 
                caller_name=caller_context,
                file_path=file_path,
                called_name=called_name,
                line_number=line_number,
                args=call.get('args', []),
                full_call_name=full_call_name)
                
                # Strategy 2: For method calls like "self.import_extractor.extract_python_imports"
                # Also try to match against static methods in classes
                if '.' in full_call_name:
                    parts = full_call_name.split('.')
                    if len(parts) >= 2:
                        # Try to find the method in any class
                        method_name = parts[-1]  # Last part is the method name
                        
                        session.run("""
                            MATCH (caller:Function {name: $caller_name, file_path: $file_path})
                            MATCH (called:Function {name: $method_name})
                            WHERE called.name = $method_name
                            MERGE (caller)-[:CALLS {line_number: $line_number, args: $args, full_call_name: $full_call_name, call_type: 'method'}]->(called)
                        """, 
                        caller_name=caller_context,
                        file_path=file_path,
                        method_name=method_name,
                        line_number=line_number,
                        args=call.get('args', []),
                        full_call_name=full_call_name)
            
            elif file_path.endswith(".js") and caller_context:
                # Strategy 1: Try exact match with the called function name
                session.run("""
                    MATCH (caller:Function {name: $caller_name, file_path: $file_path})
                    MATCH (called:Function {name: $called_name})
                    MERGE (caller)-[:CALLS {line_number: $line_number, args: $args, full_call_name: $full_call_name, call_type: $call_type , startColumn : $startColumn}]->(called)
                """, 
                caller_name=caller_context,
                file_path=file_path,
                called_name=called_name,
                line_number=line_number,
                args=call.get('args', []),
                full_call_name=full_call_name,
                call_type = call_type,
                startColumn = startColumn)
                
    def _create_class_method_relationships(self, session, file_data: Dict):
        """Create CONTAINS relationships from classes to their methods"""
        file_path = file_data['file_path']
        
        for func in file_data.get('functions', []):
            class_context = func.get('class_context')
            if class_context:
                session.run("""
                    MATCH (c:Class {name: $class_name, file_path: $file_path})
                    MATCH (fn:Function {name: $func_name, file_path: $file_path, line_number: $func_line})
                    MERGE (c)-[:CONTAINS]->(fn)
                """, 
                class_name=class_context,
                file_path=file_path,
                func_name=func['name'],
                func_line=func['line_number'])
    
    async def build_graph_from_path_async(self, path: Path, is_dependency: bool = False, job_manager: JobManager = None, job_id: str = None) -> Dict[str, Any]:
        """Build Neo4j graph from a given path asynchronously with progress tracking"""
        try:
            # Update job status to running
            if job_manager and job_id:
                job_manager.update_job(job_id, status=JobStatus.RUNNING)
            
            # Add repository
            self.add_repository_to_graph(path, is_dependency)
            
            # Find files
            python_files = []
            js_files = []
            
            if path.is_file() and path.suffix == '.py':
                python_files = [path]
            elif path.is_file() and path.suffix == '.js':
                js_files = [path]
            else:
                python_files = list(path.glob("**/*.py"))
                js_files = list(path.glob("**/*.js"))
            
            total_files = len(python_files) + len(js_files)
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
                    self.add_python_file_to_graph(file_data, path.name)
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
                    
            for file_path in js_files:
                try:
                    # Update current file being processed
                    if job_manager and job_id:
                        job_manager.update_job(job_id, current_file=str(file_path))
                    
                    file_data = self.parse_js_file(file_path, is_dependency)
                    self.add_js_file_to_graph(file_data, path.name)
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
    """Module for finding relevant code snippets using proper relationship traversal"""
    
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
    
    def who_calls_function(self, function_name: str, file_path: str = None) -> List[Dict]:
        """Find what functions call a specific function using CALLS relationships with improved matching"""
        with self.driver.session() as session:
            if file_path:
                # First try exact match
                result = session.run("""
                    MATCH (target:Function {name: $function_name, file_path: $file_path})
                    MATCH (caller:Function)-[call:CALLS]->(target)
                    OPTIONAL MATCH (caller_file:File)-[:CONTAINS]->(caller)
                    RETURN DISTINCT
                        caller.name as caller_function,
                        caller.file_path as caller_file_path,
                        caller.line_number as caller_line_number,
                        caller.docstring as caller_docstring,
                        caller.is_dependency as caller_is_dependency,
                        call.line_number as call_line_number,
                        call.args as call_args,
                        call.full_call_name as full_call_name,
                        call.call_type as call_type,
                        target.file_path as target_file_path
                    ORDER BY caller.is_dependency ASC, caller.file_path, caller.line_number
                    LIMIT 20
                """, function_name=function_name, file_path=file_path)
                
                # If no results, try without file path restriction
                results = [dict(record) for record in result]
                if not results:
                    result = session.run("""
                        MATCH (target:Function {name: $function_name})
                        MATCH (caller:Function)-[call:CALLS]->(target)
                        OPTIONAL MATCH (caller_file:File)-[:CONTAINS]->(caller)
                        RETURN DISTINCT
                            caller.name as caller_function,
                            caller.file_path as caller_file_path,
                            caller.line_number as caller_line_number,
                            caller.docstring as caller_docstring,
                            caller.is_dependency as caller_is_dependency,
                            call.line_number as call_line_number,
                            call.args as call_args,
                            call.full_call_name as full_call_name,
                            call.call_type as call_type,
                            target.file_path as target_file_path
                        ORDER BY caller.is_dependency ASC, caller.file_path, caller.line_number
                        LIMIT 20
                    """, function_name=function_name)
                    results = [dict(record) for record in result]
            else:
                result = session.run("""
                    MATCH (target:Function {name: $function_name})
                    MATCH (caller:Function)-[call:CALLS]->(target)
                    OPTIONAL MATCH (caller_file:File)-[:CONTAINS]->(caller)
                    RETURN DISTINCT
                        caller.name as caller_function,
                        caller.file_path as caller_file_path,
                        caller.line_number as caller_line_number,
                        caller.docstring as caller_docstring,
                        caller.is_dependency as caller_is_dependency,
                        call.line_number as call_line_number,
                        call.args as call_args,
                        call.full_call_name as full_call_name,
                        call.call_type as call_type,
                        target.file_path as target_file_path
                    ORDER BY caller.is_dependency ASC, caller.file_path, caller.line_number
                    LIMIT 20
                """, function_name=function_name)
                results = [dict(record) for record in result]
            
            return results
    
    def what_does_function_call(self, function_name: str, file_path: str = None) -> List[Dict]:
        """Find what functions a specific function calls using CALLS relationships"""
        with self.driver.session() as session:
            if file_path:
                result = session.run("""
                    MATCH (caller:Function {name: $function_name, file_path: $file_path})
                    MATCH (caller)-[call:CALLS]->(called:Function)
                    OPTIONAL MATCH (called_file:File)-[:CONTAINS]->(called)
                    RETURN DISTINCT
                        called.name as called_function,
                        called.file_path as called_file_path,
                        called.line_number as called_line_number,
                        called.docstring as called_docstring,
                        called.is_dependency as called_is_dependency,
                        call.line_number as call_line_number,
                        call.args as call_args,
                        call.full_call_name as full_call_name,
                        call.call_type as call_type
                    ORDER BY called.is_dependency ASC, called.name
                    LIMIT 20
                """, function_name=function_name, file_path=file_path)
            else:
                result = session.run("""
                    MATCH (caller:Function {name: $function_name})
                    MATCH (caller)-[call:CALLS]->(called:Function)
                    OPTIONAL MATCH (called_file:File)-[:CONTAINS]->(called)
                    RETURN DISTINCT
                        called.name as called_function,
                        called.file_path as called_file_path,
                        called.line_number as called_line_number,
                        called.docstring as called_docstring,
                        called.is_dependency as called_is_dependency,
                        call.line_number as call_line_number,
                        call.args as call_args,
                        call.full_call_name as full_call_name,
                        call.call_type as call_type
                    ORDER BY called.is_dependency ASC, called.name
                    LIMIT 20
                """, function_name=function_name)
            
            return [dict(record) for record in result]
    
    def who_imports_module(self, module_name: str) -> List[Dict]:
        """Find what files import a specific module using IMPORTS relationships"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (file:File)-[imp:IMPORTS]->(module:Module)
                WHERE module.name CONTAINS $module_name OR module.name = $module_name
                OPTIONAL MATCH (repo:Repository)-[:CONTAINS]->(file)
                RETURN DISTINCT
                    file.name as file_name,
                    file.path as file_path,
                    file.relative_path as file_relative_path,
                    module.name as imported_module,
                    module.alias as import_alias,
                    file.is_dependency as file_is_dependency,
                    repo.name as repository_name
                ORDER BY file.is_dependency ASC, file.path
                LIMIT 20
            """, module_name=module_name)
            
            return [dict(record) for record in result]
    
    def who_modifies_variable(self, variable_name: str) -> List[Dict]:
        """Find what functions contain or modify a specific variable"""
        with self.driver.session() as session:
            # Find functions that contain the variable
            result = session.run("""
                MATCH (var:Variable {name: $variable_name})
                MATCH (container)-[:CONTAINS]->(var)
                WHERE container:Function OR container:Class OR container:File
                OPTIONAL MATCH (file:File)-[:CONTAINS]->(container)
                RETURN DISTINCT
                    CASE 
                        WHEN container:Function THEN container.name
                        WHEN container:Class THEN container.name
                        ELSE 'file_level'
                    END as container_name,
                    CASE 
                        WHEN container:Function THEN 'function'
                        WHEN container:Class THEN 'class'
                        ELSE 'file'
                    END as container_type,
                    COALESCE(container.file_path, file.path) as file_path,
                    container.line_number as container_line_number,
                    var.line_number as variable_line_number,
                    var.value as variable_value,
                    var.context as variable_context,
                    COALESCE(container.is_dependency, file.is_dependency, false) as is_dependency
                ORDER BY is_dependency ASC, file_path, variable_line_number
                LIMIT 20
            """, variable_name=variable_name)
            
            return [dict(record) for record in result]
    
    def find_class_hierarchy(self, class_name: str) -> Dict[str, Any]:
        """Find class inheritance relationships using INHERITS_FROM relationships"""
        with self.driver.session() as session:
            # Find parent classes (what this class inherits from)
            parents_result = session.run("""
                MATCH (child:Class {name: $class_name})-[:INHERITS_FROM]->(parent:Class)
                OPTIONAL MATCH (parent_file:File)-[:CONTAINS]->(parent)
                RETURN DISTINCT
                    parent.name as parent_class,
                    parent.file_path as parent_file_path,
                    parent.line_number as parent_line_number,
                    parent.docstring as parent_docstring,
                    parent.is_dependency as parent_is_dependency
                ORDER BY parent.is_dependency ASC, parent.name
            """, class_name=class_name)
            
            # Find child classes (what inherits from this class)
            children_result = session.run("""
                MATCH (child:Class)-[:INHERITS_FROM]->(parent:Class {name: $class_name})
                OPTIONAL MATCH (child_file:File)-[:CONTAINS]->(child)
                RETURN DISTINCT
                    child.name as child_class,
                    child.file_path as child_file_path,
                    child.line_number as child_line_number,
                    child.docstring as child_docstring,
                    child.is_dependency as child_is_dependency
                ORDER BY child.is_dependency ASC, child.name
            """, class_name=class_name)
            
            # Find methods of this class
            methods_result = session.run("""
                MATCH (class:Class {name: $class_name})-[:CONTAINS]->(method:Function)
                RETURN DISTINCT
                    method.name as method_name,
                    method.file_path as method_file_path,
                    method.line_number as method_line_number,
                    method.args as method_args,
                    method.docstring as method_docstring,
                    method.is_dependency as method_is_dependency
                ORDER BY method.is_dependency ASC, method.line_number
            """, class_name=class_name)
            
            return {
                "class_name": class_name,
                "parent_classes": [dict(record) for record in parents_result],
                "child_classes": [dict(record) for record in children_result],
                "methods": [dict(record) for record in methods_result]
            }
    
    def find_function_overrides(self, function_name: str) -> List[Dict]:
        """Find all implementations of a function across different classes"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (class:Class)-[:CONTAINS]->(func:Function {name: $function_name})
                OPTIONAL MATCH (file:File)-[:CONTAINS]->(class)
                RETURN DISTINCT
                    class.name as class_name,
                    class.file_path as class_file_path,
                    func.name as function_name,
                    func.line_number as function_line_number,
                    func.args as function_args,
                    func.docstring as function_docstring,
                    func.is_dependency as is_dependency,
                    file.name as file_name
                ORDER BY func.is_dependency ASC, class.name
                LIMIT 20
            """, function_name=function_name)
            
            return [dict(record) for record in result]
    
    def find_dead_code(self) -> Dict[str, Any]:
        """Find potentially unused functions (not called by other functions in the project)"""
        with self.driver.session() as session:
            # Find functions that have no incoming CALLS relationships from non-dependency code
            result = session.run("""
                MATCH (func:Function)
                WHERE func.is_dependency = false
                  AND NOT func.name IN ['main', '__init__', '__main__', 'setup', 'run', '__new__', '__del__']
                  AND NOT func.name STARTS WITH '_test'
                  AND NOT func.name STARTS WITH 'test_'
                WITH func
                OPTIONAL MATCH (caller:Function)-[:CALLS]->(func)
                WHERE caller.is_dependency = false
                WITH func, count(caller) as caller_count
                WHERE caller_count = 0
                OPTIONAL MATCH (file:File)-[:CONTAINS]->(func)
                RETURN
                    func.name as function_name,
                    func.file_path as file_path,
                    func.line_number as line_number,
                    func.docstring as docstring,
                    func.context as context,
                    file.name as file_name
                ORDER BY func.file_path, func.line_number
                LIMIT 50
            """)
            
            return {
                "potentially_unused_functions": [dict(record) for record in result],
                "note": "These functions might be unused, but could be entry points, callbacks, or called dynamically"
            }
    
    def find_function_call_chain(self, start_function: str, end_function: str, max_depth: int = 5) -> List[Dict]:
        """Find call chains between two functions"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH path = shortestPath(
                    (start:Function {name: $start_function})-[:CALLS*1..$max_depth]->(end:Function {name: $end_function})
                )
                WITH path, nodes(path) as func_nodes, relationships(path) as call_rels
                RETURN 
                    [node in func_nodes | {
                        name: node.name,
                        file_path: node.file_path,
                        line_number: node.line_number,
                        is_dependency: node.is_dependency
                    }] as function_chain,
                    [rel in call_rels | {
                        call_line: rel.line_number,
                        args: rel.args,
                        full_call_name: rel.full_call_name
                    }] as call_details,
                    length(path) as chain_length
                ORDER BY chain_length ASC
                LIMIT 10
            """, start_function=start_function, end_function=end_function, max_depth=max_depth)
            
            return [dict(record) for record in result]
    
    def find_module_dependencies(self, module_name: str) -> Dict[str, Any]:
        """Find all dependencies and dependents of a module"""
        with self.driver.session() as session:
            # Find what files import this module
            importers_result = session.run("""
                MATCH (file:File)-[:IMPORTS]->(module:Module {name: $module_name})
                OPTIONAL MATCH (repo:Repository)-[:CONTAINS]->(file)
                RETURN DISTINCT
                    file.name as file_name,
                    file.path as file_path,
                    file.is_dependency as file_is_dependency,
                    repo.name as repository_name
                ORDER BY file.is_dependency ASC, file.path
                LIMIT 20
            """, module_name=module_name)
            
            # Find what modules are imported by files that import this module
            related_imports_result = session.run("""
                MATCH (file:File)-[:IMPORTS]->(target_module:Module {name: $module_name})
                MATCH (file)-[:IMPORTS]->(other_module:Module)
                WHERE other_module <> target_module
                RETURN DISTINCT
                    other_module.name as related_module,
                    other_module.alias as module_alias,
                    count(file) as usage_count
                ORDER BY usage_count DESC
                LIMIT 20
            """, module_name=module_name)
            
            return {
                "module_name": module_name,
                "imported_by_files": [dict(record) for record in importers_result],
                "frequently_used_with": [dict(record) for record in related_imports_result]
            }
    
    def find_variable_usage_scope(self, variable_name: str) -> Dict[str, Any]:
        """Find the scope and usage patterns of a variable"""
        with self.driver.session() as session:
            # Find all instances of the variable
            variable_instances = session.run("""
                MATCH (var:Variable {name: $variable_name})
                OPTIONAL MATCH (container)-[:CONTAINS]->(var)
                WHERE container:Function OR container:Class OR container:File
                OPTIONAL MATCH (file:File)-[:CONTAINS]->(var)
                RETURN DISTINCT
                    var.name as variable_name,
                    var.value as variable_value,
                    var.line_number as line_number,
                    var.context as context,
                    COALESCE(var.file_path, file.path) as file_path,
                    CASE 
                        WHEN container:Function THEN 'function'
                        WHEN container:Class THEN 'class'
                        ELSE 'module'
                    END as scope_type,
                    CASE 
                        WHEN container:Function THEN container.name
                        WHEN container:Class THEN container.name
                        ELSE 'module_level'
                    END as scope_name,
                    var.is_dependency as is_dependency
                ORDER BY var.is_dependency ASC, file_path, line_number
            """, variable_name=variable_name)
            
            return {
                "variable_name": variable_name,
                "instances": [dict(record) for record in variable_instances]
            }
    
    def analyze_code_relationships(self, query_type: str, target: str, context: str = None) -> Dict[str, Any]:
        """Main method to analyze different types of code relationships with fixed return types"""
        
        query_type = query_type.lower().strip()
        
        try:
            if query_type in ["who_calls", "what_calls", "callers"]:
                results = self.who_calls_function(target, context)
                return {
                    "query_type": "who_calls",
                    "target": target,
                    "context": context,
                    "results": results,
                    "summary": f"Found {len(results)} functions that call '{target}'"
                }
            
            elif query_type in ["what_calls", "calls_what", "dependencies"]:
                results = self.what_does_function_call(target, context)
                return {
                    "query_type": "what_calls",  # Fixed: was incorrectly "who_calls"
                    "target": target,
                    "context": context,
                    "results": results,
                    "summary": f"Function '{target}' calls {len(results)} other functions"
                }
            
            elif query_type in ["who_imports", "imports", "importers"]:
                results = self.who_imports_module(target)
                return {
                    "query_type": "who_imports",
                    "target": target,
                    "results": results,
                    "summary": f"Found {len(results)} files that import '{target}'"
                }
            
            elif query_type in ["who_modifies", "modifies", "mutations", "changes", "variable_usage"]:
                results = self.who_modifies_variable(target)
                return {
                    "query_type": "who_modifies",
                    "target": target,
                    "results": results,
                    "summary": f"Found {len(results)} containers that hold variable '{target}'"
                }
            
            elif query_type in ["class_hierarchy", "inheritance", "extends"]:
                results = self.find_class_hierarchy(target)
                return {
                    "query_type": "class_hierarchy",
                    "target": target,
                    "results": results,
                    "summary": f"Class '{target}' has {len(results['parent_classes'])} parents, {len(results['child_classes'])} children, and {len(results['methods'])} methods"
                }
            
            elif query_type in ["overrides", "implementations", "polymorphism"]:
                results = self.find_function_overrides(target)
                return {
                    "query_type": "overrides",
                    "target": target,
                    "results": results,
                    "summary": f"Found {len(results)} implementations of function '{target}'"
                }
            
            elif query_type in ["dead_code", "unused", "unreachable"]:
                results = self.find_dead_code()
                return {
                    "query_type": "dead_code",
                    "results": results,
                    "summary": f"Found {len(results['potentially_unused_functions'])} potentially unused functions"
                }
            
            elif query_type in ["call_chain", "path", "chain"]:
                # For call chain, target should be "start_func->end_func"
                if '->' in target:
                    start_func, end_func = target.split('->', 1)
                    results = self.find_function_call_chain(start_func.strip(), end_func.strip())
                    return {
                        "query_type": "call_chain",
                        "target": target,
                        "results": results,
                        "summary": f"Found {len(results)} call chains from '{start_func.strip()}' to '{end_func.strip()}'"
                    }
                else:
                    return {
                        "error": "For call_chain queries, use format 'start_function->end_function'",
                        "example": "main->process_data"
                    }
            
            elif query_type in ["module_deps", "module_dependencies", "module_usage"]:
                results = self.find_module_dependencies(target)
                return {
                    "query_type": "module_dependencies",
                    "target": target,
                    "results": results,
                    "summary": f"Module '{target}' is imported by {len(results['imported_by_files'])} files"
                }
            
            elif query_type in ["variable_scope", "var_scope", "variable_usage_scope"]:
                results = self.find_variable_usage_scope(target)
                return {
                    "query_type": "variable_scope",
                    "target": target,
                    "results": results,
                    "summary": f"Variable '{target}' has {len(results['instances'])} instances across different scopes"
                }
            
            else:
                return {
                    "error": f"Unknown query type: {query_type}",
                    "supported_types": [
                        "who_calls", "what_calls", "who_imports", "who_modifies",
                        "class_hierarchy", "overrides", "dead_code", "call_chain",
                        "module_deps", "variable_scope"
                    ]
                }
        
        except Exception as e:
            return {
                "error": f"Error executing relationship query: {str(e)}",
                "query_type": query_type,
                "target": target
            }

    def close(self):
        """Close Neo4j driver"""
        self.driver.close()

class SemanticCodeSearcher:
    """Class to handle semantic code search"""

    # Edge weights for propagation (do not include CONTAINS)
    EDGE_WEIGHTS = {
        "CALLS": 0.2
        # Add more edge types as needed
    }

    def __init__(self, embedding_model, embedding_tokenizer, code_finder):
        self.embedding_model = embedding_model
        self.embedding_tokenizer = embedding_tokenizer
        self.code_finder = code_finder

    def embed_text(self, text: str):
        inputs = self.embedding_tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            if hasattr(outputs, 'last_hidden_state'):
                emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            else:
                emb = outputs[0][:, 0, :].cpu().numpy().flatten()
        return emb

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def semantic_search_with_propagation(
        self, 
        query: str, 
        repo: Optional[str] = None, 
        file_names: Optional[List[str]] = None, 
        class_contexts: Optional[List[str]] = None, 
        function_names: Optional[List[str]] = None, 
        top_k: int = 10
    ):
        """
        Perform semantic similarity search and propagate scores through the graph.
        Args:
            query (str): The natural language search query.
            repo (str): The repository to search in.
            file_names (list, optional): Filter by file names. Defaults to None.
            class_contexts (list, optional): Filter by class contexts. Defaults to None.
            function_names (list, optional): Filter by function names. Defaults to None.
            top_k (int, optional): The total number of results to return. Defaults to 10.
            
        Returns:
            list: A list of top-k matching nodes with their details.
        """
        if not self.code_finder.driver:
            raise ConnectionError("Neo4j driver not initialized.")

        query_embedding = self.embed_text(query)

        cypher_query = """
        MATCH (n)
        WHERE (n:Function OR n:Class OR n:Variable)
          AND n.embedding IS NOT NULL
        """
        params = {}

        if file_names:
            cypher_query += " AND n.file_path IN $file_names"
            params["file_names"] = file_names
        if class_contexts:
            cypher_query += " AND n.class_name IN $class_contexts"
            params["class_contexts"] = class_contexts
        if function_names:
            cypher_query += " AND n.function_name IN $function_names"
            params["function_names"] = function_names
            
        cypher_query += " RETURN n, n.embedding AS embedding, elementId(n) AS node_id"

        with self.code_finder.driver.session() as session:
            result = session.run(cypher_query, params)
            
            nodes_with_embeddings = []
            for record in result:
                node_data = dict(record["n"])
                embedding = record["embedding"]
                node_id = record["node_id"]
                
                node_data["node_id"] = node_id
                if embedding:
                    nodes_with_embeddings.append((node_data, np.array(embedding)))

        if not nodes_with_embeddings:
            return []

        # Calculate initial similarity scores
        scored_nodes = {}
        for node_data, embedding in nodes_with_embeddings:
            node_id = node_data['node_id']
            similarity = self.cosine_similarity(query_embedding, embedding)
            scored_nodes[node_id] = {
                "score": similarity,
                "node": node_data
            }
            
        # Sort by initial score to get seed nodes
        seed_nodes = sorted(scored_nodes.values(), key=lambda x: x['score'], reverse=True)

        # Score propagation with edge weights (excluding CONTAINS)
        propagated_scores = scored_nodes.copy()
        with self.code_finder.driver.session() as session:
            for seed in seed_nodes:
                seed_id = seed['node']['node_id']
                seed_score = seed['score']
                # Query all outgoing and incoming relationships except CONTAINS
                rel_query = """
                    MATCH (n)-[r]->(m)
                    WHERE elementId(n) = $id AND type(r) <> 'CONTAINS'
                    RETURN type(r) AS rel_type, elementId(m) AS neighbor_id
                    UNION
                    MATCH (m)-[r]->(n)
                    WHERE elementId(n) = $id AND type(r) <> 'CONTAINS'
                    RETURN type(r) AS rel_type, elementId(m) AS neighbor_id
                """
                rel_result = session.run(rel_query, id=seed_id)
                for record in rel_result:
                    rel_type = record['rel_type']
                    neighbor_id = record['neighbor_id']
                    weight = self.EDGE_WEIGHTS.get(rel_type, 0.5)  # Default weight if not specified
                    if neighbor_id in propagated_scores:
                        propagated_scores[neighbor_id]['score'] += seed_score * weight

        # Rank all nodes by final score
        final_ranked_nodes = sorted(propagated_scores.values(), key=lambda x: x['score'], reverse=True)
        for entry in final_ranked_nodes[:top_k]:
            entry["node"].pop("embedding", None)
        
        return final_ranked_nodes[:top_k]

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
            
        # Embedding model setup
        self._load_embedding_model()
        
        self.graph_builder = GraphBuilder(embedding_model=self.embedding_model , embedding_tokenizer=self.embedding_tokenizer)
        self.code_finder = CodeFinder()
        self.import_extractor = ImportExtractor()
        self.job_manager = JobManager()
        self.semantic_searcher = SemanticCodeSearcher(
            self.embedding_model, self.embedding_tokenizer, self.code_finder
        )
    
    def _load_embedding_model(self):
        """Load the code embedding model (StarEncoder or fallback)"""
        try:
            from transformers import AutoTokenizer, AutoModel
            self.embedding_model_name = "Salesforce/SFR-Embedding-Code-400M_R"
            hf_token = HF_TOKEN
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name, token=hf_token, trust_remote_code=True)
            self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name, token=hf_token, trust_remote_code=True)
        except ImportError:
            raise ImportError("transformers library is required for semantic search. Please install it via pip.")

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
            "analyze_code_relationships": {
                "name": "analyze_code_relationships",
                "description": "Analyze code relationships like 'who calls this function', 'what does this call', 'who imports this', etc.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query_type": {
                            "type": "string",
                            "description": "Type of relationship query",
                            "enum": [
                                "who_calls", "what_calls", "who_imports", "who_modifies",
                                "class_hierarchy", "overrides", "dead_code"
                            ]
                        },
                        "target": {
                            "type": "string",
                            "description": "The function, class, module, or variable name to analyze"
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional: specific file path for more precise results",
                            "default": None
                        }
                    },
                    "required": ["query_type", "target"]
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
            },
            "code_search": {
                "name": "code_search",
                "description": "Efficiently finds code snippets related to the given query from the codebase. Ideal for natural language queries, fuzzy intent discovery, or finding code behavior across functions, classes, or variables. Eg query `How graph is generated for the python codebase`",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query (natural language or code)"},
                        "file_paths": {"type": "array", "items": {"type": "string"}, "description": "Optional: list of file path to filter"},
                        "class_contexts": {"type": "array", "items": {"type": "string"}, "description": "Optional: list of class contexts to filter"},
                        "function_names": {"type": "array", "items": {"type": "string"}, "description": "Optional: list of function names to filter"},
                        "top_k": {"type": "integer", "description": "Number of results to return", "default": 10}
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
    
    def analyze_code_relationships_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool to analyze code relationships"""
        query_type = args.get('query_type', '')
        target = args.get('target', '')
        context = args.get('context')
        
        if not query_type or not target:
            return {
                "error": "Both 'query_type' and 'target' are required",
                "supported_query_types": [
                    "who_calls", "what_calls", "who_imports", "who_modifies",
                    "class_hierarchy", "overrides", "dead_code"
                ]
            }
        
        try:
            debug_log(f"Analyzing relationships: {query_type} for {target}")
            results = self.code_finder.analyze_code_relationships(query_type, target, context)
            
            return {
                "success": True,
                "query_type": query_type,
                "target": target,
                "context": context,
                "results": results
            }
        
        except Exception as e:
            debug_log(f"Error analyzing relationships: {str(e)}")
            return {"error": f"Failed to analyze relationships: {str(e)}"}

    def find_code_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool to find code by name or content"""
        query = args.get('query')
        if not query:
            return {"error": "Query parameter is required"}
        
        results = self.code_finder.find_related_code(query)
        return {"success": True, "results": results}

    def semantic_code_search_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool for semantic code search with propagation"""
        try:
            query = args['query']
            repo = args.get('repo')
            file_names = args.get('file_paths')
            class_contexts = args.get('class_contexts')
            function_names = args.get('function_names')

            top_k = args.get('top_k', 10)

            results = self.semantic_searcher.semantic_search_with_propagation(
                query=query,
                repo=repo,
                file_names=file_names,
                class_contexts=class_contexts,
                function_names=function_names,
                top_k=top_k
            )
            return {"success": True, "results": results}
        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            return {"error": f"Failed to perform semantic search: {str(e)}"}

    async def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming tool calls"""
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
        elif tool_name == "analyze_code_relationships":
            return self.analyze_code_relationships_tool(args)
        elif tool_name == "code_search":
            return self.semantic_code_search_tool(args)
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