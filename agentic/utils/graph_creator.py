#!/usr/bin/env python3

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
        
        self._load_embedding_model()
        
        # For embedding
        self.embedding_tokenizer = embedding_tokenizer
        self.embedding_model = embedding_model

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
                code_text = func.get('source')  + f"\n{func['name']} is defined in the {func['context']} , {file_path}"
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
                code_text = cls.get('source') + f"\n{cls['name']} is defined in the {cls['context']} , {file_path}"
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
                code_text = f"Function docstring :- {func.get('docstring')}\n" + func.get('source') + f"\n{func['name']} is defined in the {func['context']} , {func['file_path']}"
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
    
    async def build_graph_from_path_async(self, path: Path, is_dependency: bool = False) -> Dict[str, Any]:
        """Build Neo4j graph from a given path asynchronously with progress tracking"""
        try:
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
            
            for file_path in python_files:
                try:
                    file_data = self.parse_python_file(file_path, is_dependency)
                    self.add_python_file_to_graph(file_data, path.name)
                    processed_files += 1
                    # Small delay to allow other operations
                    await asyncio.sleep(0.001)
                    
                except Exception as e:
                    error_msg = f"Error processing {file_path}: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    
            for file_path in js_files:
                try:
                    file_data = self.parse_js_file(file_path, is_dependency)
                    self.add_js_file_to_graph(file_data, path.name)
                    processed_files += 1

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
            return result
        
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "repository": path.name if path else "unknown"
            }
            
            return error_result
    
    def build_graph_from_path(self, path: Path, is_dependency: bool = False) -> Dict[str, Any]:
        """Build Neo4j graph from a given path (synchronous version for backward compatibility)"""
        return asyncio.run(self.build_graph_from_path_async(path, is_dependency))
    
    def close(self):
        """Close Neo4j driver"""
        self.driver.close()