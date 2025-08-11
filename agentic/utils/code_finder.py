from typing import List, Dict, Any
import re
from typing import Any
import torch
import numpy as np
from neo4j import GraphDatabase
import os

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', "Only@2025")

driver = None
def create_schema():
    """Create constraints and indexes in Neo4j"""
    with driver.session() as session:
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
            pass
   

class CodeFinder:
    """Module for finding relevant code snippets using proper relationship traversal"""
    
    def __init__(self, driver):
        self.driver = driver
        
    def _validate_and_compile(self, query: str, case_sensitive: bool):
        """Validate and compile regex pattern."""
        try:
            return re.compile(query, 0 if case_sensitive else re.IGNORECASE), True
        except re.error as e:
            print(f"Invalid regex pattern: {str(e)}")
            return None, False
    
    def _file_match_conditions(self, include_pattern=None, exclude_pattern=None):
        """Convert glob patterns to regex for Cypher query."""
        include_regexes = _glob_to_regex(include_pattern) if include_pattern else None
        exclude_regexes = _glob_to_regex(exclude_pattern) if exclude_pattern else None
        return include_regexes, exclude_regexes
    
    def find_by_class_name(self, query: str, case_sensitive=False, include_pattern=None, exclude_pattern=None) -> List[Dict]:
        """Find classes by name using regex with file filters."""
        pattern, valid = self._validate_and_compile(query, case_sensitive)
        if not valid:
            return []

        include_regexes, exclude_regexes = self._file_match_conditions(include_pattern, exclude_pattern)

        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Class)
                WHERE (c.name =~ $regex_pattern OR c.docstring =~ $regex_pattern)
                  AND ($include_regexes IS NULL OR any(re IN $include_regexes WHERE c.file_path =~ re))
                  AND ($exclude_regexes IS NULL OR all(re IN $exclude_regexes WHERE NOT c.file_path =~ re))
                RETURN c.name as name, c.file_path as file_path, c.line_number as line_number,
                       c.docstring as docstring, c.is_dependency as is_dependency
                ORDER BY c.is_dependency ASC, c.name
                LIMIT 3
            """,  regex_pattern=pattern.pattern,
                 include_regexes=include_regexes, exclude_regexes=exclude_regexes)
            
            return [dict(record) for record in result]

    def find_by_content(self, query: str, case_sensitive=False, include_pattern=None, exclude_pattern=None) -> List[Dict]:
        """Find functions by regex in source or docstring with file filters."""
        pattern, valid = self._validate_and_compile(query, case_sensitive)
        if not valid:
            return []

        include_regexes, exclude_regexes = self._file_match_conditions(include_pattern, exclude_pattern)
        with self.driver.session() as session:
            func_result = session.run("""
                MATCH (f:Function)
                WHERE (f.source =~ $regex_pattern OR f.docstring =~ $regex_pattern)
                  AND ($include_regexes IS NULL OR any(re IN $include_regexes WHERE f.file_path =~ re))
                  AND ($exclude_regexes IS NULL OR all(re IN $exclude_regexes WHERE NOT f.file_path =~ re))
                RETURN 'function' as type, f.name as name, f.file_path as file_path, 
                       f.line_number as line_number, f.source as source, 
                       f.is_dependency as is_dependency
                ORDER BY f.is_dependency ASC, f.name
                LIMIT 15
            """, regex_pattern=pattern.pattern,
                 include_regexes=include_regexes, exclude_regexes=exclude_regexes)
            
            return [dict(record) for record in func_result]

    def find_related_code(self, user_query: str, case_sensitive=False, include_pattern=None, exclude_pattern=None) -> Dict[str, Any]:
        """Find code related to a query using regex + file filters."""
        user_query = "(?is).*" + user_query + ".*"
        results = {
            "query": user_query,
            "classes_by_name": self.find_by_class_name(user_query, case_sensitive, include_pattern, exclude_pattern),
            "content_matches": self.find_by_content(user_query, case_sensitive, include_pattern, exclude_pattern)
        }

        # Merge + limit
        all_results = results["classes_by_name"] + results["content_matches"]
        print(all_results)
        return all_results[:15]
    
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
        
def _glob_to_regex(pattern_str: str) -> List[re.Pattern]:
    """Convert comma-separated glob patterns to regex patterns."""
    patterns = []
    
    for glob in pattern_str.split(','):
        glob = glob.strip()
        if not glob:
            continue
        
        # Convert glob syntax to regex
        regex = (glob
                .replace('.', r'\.')  # Escape dots
                .replace('*', r'.*')  # * becomes .*
                .replace('?', r'.'))  # ? becomes .
        
        try:
            patterns.append(re.compile(f"^{regex}$"))
        except re.error:
            # Skip invalid patterns
            continue
    
    return patterns

if __name__ == "__main__":
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    create_schema() 
    
    fnd = CodeFinder(driver)
    print(fnd.find_related_code("deploy_new_config" , False, None, None))