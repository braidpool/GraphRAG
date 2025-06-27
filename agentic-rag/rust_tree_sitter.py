import os
from pathlib import Path
from connections.neo4j_login import connect_to_neo4j
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from tree_sitter_language_pack import get_binding, get_language, get_parser
from neo4j import GraphDatabase

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

driver = connect_to_neo4j()
NORMALIZED_BASE = Path("normalized_repos")

# Setup Rust parser
def setup_rust_parser():
    # Get the path to the Rust language library
        
    binding = get_binding("rust")  # this is a pycapsule object pointing to the C binding
    lang = get_language("rust")  # this is an instance of tree_sitter.Language
    parser = get_parser("rust") 
    return parser

# Initialize parser
RUST_PARSER = setup_rust_parser()

# ─── RUST AST PARSING ────────────────────────────────────────────────────────

@dataclass  
class RustFunction:
    name: str
    visibility: str
    is_async: bool
    is_unsafe: bool
    return_type: Optional[str]
    parameters: List[Tuple[str, str]]  # (name, type)
    start_line: int
    end_line: int

@dataclass
class RustStruct:
    name: str
    visibility: str
    fields: List[Tuple[str, str]]  # (name, type)
    start_line: int
    end_line: int

@dataclass
class RustTrait:
    name: str
    visibility: str
    start_line: int
    end_line: int

@dataclass
class RustImpl:
    trait_name: Optional[str]  # None for inherent impls
    type_name: str
    start_line: int
    end_line: int

@dataclass
class RustModule:
    name: str
    visibility: str
    start_line: int
    end_line: int

def collect_rust_files(normalized_base: Path):
    """
    Walk through each repo folder and return a list of Rust files with their module paths.
    Returns: List[Tuple[repo_name, file_path, module_name]]
    """
    all_files = []
    for repo_dir in normalized_base.iterdir():
        if not repo_dir.is_dir():
            continue
            
        repo_name = repo_dir.name
        for root, _, files in os.walk(repo_dir):
            root_path = Path(root)
            for fname in files:
                if fname.lower().endswith('.rs'):
                    fpath = root_path / fname
                    # Convert file path to module path
                    rel_path = fpath.relative_to(repo_dir).with_suffix('')
                    module_parts = [repo_name] + list(rel_path.parts)
                    # Handle mod.rs specially
                    if fname == 'mod.rs':
                        module_parts = module_parts[:-1]  # Remove 'mod' from path
                    module_name = '.'.join(module_parts)
                    all_files.append((repo_name, fpath, module_name))
    return all_files

def parse_rust_file(file_path: Path):
    """Parse a Rust file and return the syntax tree."""
    with open(file_path, 'rb') as f:
        source = f.read()
    return RUST_PARSER.parse(source)

def extract_rust_items(tree, source: bytes, file_path: None):
    """Extract functions, structs, traits, and impls from a Rust syntax tree."""
    root = tree.root_node
    
    functions = {}
    structs = {}
    traits = {}
    impls = {}
    modules = {}
    imports = set()
    
    def get_import_path(node):
        """Recursively extract the full import path from a use declaration node."""
        if node.type == 'identifier':
            return get_text(node)
        elif node.type == 'scoped_identifier':
            left = get_import_path(node.child_by_field_name('path'))
            right = get_import_path(node.child_by_field_name('name'))
            return f"{left}::{right}" if left else right
        elif node.type == 'scoped_use_list':
            path = get_import_path(node.child_by_field_name('path'))
            items = []
            for child in node.child_by_field_name('list').named_children:
                if child.type == 'identifier':
                    items.append(get_text(child))
                elif child.type == 'use_as_clause':
                    name = get_text(child.child_by_field_name('name'))
                    alias = get_text(child.child_by_field_name('alias'))
                    items.append(f"{name} as {alias}")
            return f"{path}::{{{', '.join(items)}}}"
        elif node.type == 'use_as_clause':
            name = get_import_path(node.child_by_field_name('path'))
            alias = get_text(node.child_by_field_name('alias'))
            return f"{name} as {alias}"
        return ""
    
    # Helper to get source code for a node
    def get_text(node):
        return source[node.start_byte:node.end_byte].decode('utf-8')
    
    # Helper to process type
    def process_type(node):
        if not node:
            return None
        return get_text(node).strip()
    
    # Process all items in the file
    for item in root.named_children:
        # Function definitions
        
        if item.type == 'function_item':
            name_node = item.child_by_field_name('name')
            if not name_node:
                continue
                
            name = get_text(name_node)
            visibility = 'pub' if item.prev_named_sibling and item.prev_named_sibling.type == 'visibility_modifier' else ''
            
            # Check if function is async
            is_async = bool(item.child_by_field_name('async_token'))
            
            # Check if function is unsafe
            is_unsafe = bool(item.child_by_field_name('unsafe_token'))
            
            # Get return type
            return_type = None
            if item.child_by_field_name('return_type'):
                return_type_node = item.child_by_field_name('return_type')
                # Get all named children and find the type (skip '->' token)
                named_children = [c for c in return_type_node.named_children]
                # The type is typically the last named child
                if named_children:
                    return_type = process_type(named_children[-1])
                else:
                    return_type = None
            else:
                return_type = None
            # Get parameters
            parameters = []
            params_node = item.child_by_field_name('parameters')
            if params_node:
                for param in params_node.named_children:
                    if param.type == 'parameter':
                        param_name = param.child_by_field_name('pattern')
                        param_type = param.child_by_field_name('type')
                        if param_name and param_type:
                            parameters.append((
                                get_text(param_name).strip(),
                                get_text(param_type).strip()
                            ))
            
            functions[name] = RustFunction(
                name=name,
                visibility=visibility,
                is_async=is_async,
                is_unsafe=is_unsafe,
                return_type=return_type,
                parameters=parameters,
                start_line=item.start_point[0] + 1,  # 1-based
                end_line=item.end_point[0] + 1
            )
        
        # Struct definitions
        elif item.type == 'struct_item':
            name_node = item.child_by_field_name('name')
            if not name_node:
                continue
                
            name = get_text(name_node)
            visibility = 'pub' if item.prev_named_sibling and item.prev_named_sibling.type == 'visibility_modifier' else ''
            
            # Get fields
            fields = []
            field_list = item.child_by_field_name('body')
            if field_list:
                for field in field_list.named_children:
                    if field.type == 'field_declaration':
                        field_name = field.child_by_field_name('name')
                        field_type = field.child_by_field_name('type')
                        if field_name and field_type:
                            fields.append((
                                get_text(field_name).strip(),
                                get_text(field_type).strip()
                            ))
            
            structs[name] = RustStruct(
                name=name,
                visibility=visibility,
                fields=fields,
                start_line=item.start_point[0] + 1,
                end_line=item.end_point[0] + 1
            )
        
        # Trait definitions
        elif item.type == 'trait_item':
            name_node = item.child_by_field_name('name')
            if not name_node:
                continue
                
            name = get_text(name_node)
            visibility = 'pub' if item.prev_named_sibling and item.prev_named_sibling.type == 'visibility_modifier' else ''
            
            traits[name] = RustTrait(
                name=name,
                visibility=visibility,
                start_line=item.start_point[0] + 1,
                end_line=item.end_point[0] + 1
            )
        
        # Impl blocks
        elif item.type == 'impl_item':
            trait_node = item.child_by_field_name('trait')
            type_node = item.child_by_field_name('type')
            
            if not type_node:
                continue
                
            trait_name = get_text(trait_node) if trait_node else None
            type_name = get_text(type_node)
            
            impls[f"{trait_name}::{type_name}" if trait_name else type_name] = RustImpl(
                trait_name=trait_name,
                type_name=type_name,
                start_line=item.start_point[0] + 1,
                end_line=item.end_point[0] + 1
            )
        
        # Module declarations
        elif item.type == 'mod_item':
            name_node = item.child_by_field_name('name')
            if not name_node:
                continue
                
            name = get_text(name_node)
            visibility = 'pub' if item.prev_named_sibling and item.prev_named_sibling.type == 'visibility_modifier' else ''
            
            modules[name] = RustModule(
                name=name,
                visibility=visibility,
                start_line=item.start_point[0] + 1,
                end_line=item.end_point[0] + 1
            )
        
        # Use declarations (imports)
        elif item.type == 'use_declaration':
            try:
                # Get the argument of the use declaration
                arg = item.child_by_field_name('argument')
                if arg:
                    import_path = get_import_path(arg)
                    if import_path:
                        print(f"Found import: {import_path} in {file_path}")
                        imports.add(import_path)
                    else:
                        print(f"Warning: Could not parse import: {get_text(item)}")
            except Exception as e:
                print(f"Error processing import: {e}")
                import traceback
                traceback.print_exc()
    
    return {
        'functions': functions,
        'structs': structs,
        'traits': traits,
        'impls': impls,
        'modules': modules,
        'imports': list(imports)
    }

def collect_call_edges(tree, source: bytes) -> Set[str]:
    """Collect function calls within the given syntax tree."""
    calls = set()
    root = tree.root_node
    
    def get_text(node):
        return source[node.start_byte:node.end_byte].decode('utf-8')
    
    # Simple visitor to collect function calls
    def visit(node):
        if node.type == 'call_expression':
            # Get the function being called
            func_node = node.child_by_field_name('function')
            if func_node:
                calls.add(get_text(func_node).split('.').pop())  # Get just the function name
        
        for child in node.children:
            visit(child)
    
    visit(root)
    return calls

# ─── NEO4J INTEGRATION ───────────────────────────────────────────────────────

class Neo4jGraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def run(self, query, parameters=None, read_only=False):
        session = self.driver.session()
        try:
            result = session.run(query, parameters or {})
            if read_only:
                # For read queries, consume and return all records
                records = list(result)  # Consume all records
                return records
            else:
                # For write queries, consume the result but don't return anything
                result.consume()
                return None
        except Exception as e:
            print(f"Error executing query: {e}")
            raise
        finally:
            session.close()  # Always close the session
            
    def create_repo_node(self, repo_name):
        self.run(
            "MERGE (r:Repository {name: $name})",
            {"name": repo_name}
        )
        
    def create_file_node(self, repo_name, file_path, module_name):
        self.run(
            """
            MERGE (f:File {path: $path})
            SET f.module = $module_name
            WITH f
            MATCH (r:Repository {name: $repo_name})
            MERGE (r)-[:CONTAINS]->(f)
            """,
            {
                "path": file_path,
                "module_name": module_name,
                "repo_name": repo_name
            }
        )
        
    def close(self):
        self.driver.close()

def process_rust_repositories():
    """Main function to process all Rust repositories and populate Neo4j."""
    graph = Neo4jGraphBuilder(os.getenv("NEO4J_URI"), os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    
    try:
        # Collect all Rust files
        rust_files = collect_rust_files(NORMALIZED_BASE)
        
        for repo_name, file_path, module_name in rust_files:
            # Create repo and file nodes
            graph.create_repo_node(repo_name)
            graph.create_file_node(repo_name, str(file_path), module_name)
            
            # Parse the Rust file
            tree = parse_rust_file(file_path)
            with open(file_path, 'rb') as f:
                source = f.read()
            
            # Extract items
            items = extract_rust_items(tree, source,file_path)
            
            # Process functions
            for func_name, func in items['functions'].items():
                qualified_name = f"{module_name}::{func_name}"
                graph.run(
                    """
                    MERGE (f:RustFunction {qualified_name: $qualified_name})
                    ON CREATE SET 
                        f.name = $name,
                        f.visibility = $visibility,
                        f.is_async = $is_async,
                        f.is_unsafe = $is_unsafe,
                        f.return_type = $return_type,
                        f.start_line = $start_line,
                        f.end_line = $end_line,
                        f.file = $file_path,
                        f.repo = $repo_name
                    WITH f
                    MATCH (file:File {path: $file_path})
                    MERGE (file)-[:DECLARES_FUNCTION]->(f)
                    """,
                    {
                        "qualified_name": qualified_name,
                        "name": func_name,
                        "visibility": func.visibility,
                        "is_async": func.is_async,
                        "is_unsafe": func.is_unsafe,
                        "return_type": func.return_type or "",
                        "start_line": func.start_line,
                        "end_line": func.end_line,
                        "file_path": str(file_path),
                        "repo_name": repo_name
                    }
                )
            
            # Process structs
            for struct_name, struct in items['structs'].items():
                qualified_name = f"{module_name}::{struct_name}"
                graph.run(
                    """
                    MERGE (s:RustStruct {qualified_name: $qualified_name})
                    ON CREATE SET 
                        s.name = $name,
                        s.visibility = $visibility,
                        s.start_line = $start_line,
                        s.end_line = $end_line,
                        s.file = $file_path,
                        s.repo = $repo_name
                    WITH s
                    MATCH (file:File {path: $file_path})
                    MERGE (file)-[:DECLARES_STRUCT]->(s)
                    """,
                    {
                        "qualified_name": qualified_name,
                        "name": struct_name,
                        "visibility": struct.visibility,
                        "start_line": struct.start_line,
                        "end_line": struct.end_line,
                        "file_path": str(file_path),
                        "repo_name": repo_name
                    }
                )
                
                # Add fields
                for field_name, field_type in struct.fields:
                    graph.run(
                        """
                        MATCH (s:RustStruct {qualified_name: $struct_name})
                        MERGE (f:RustField {name: $field_name, type: $field_type})
                        MERGE (s)-[:HAS_FIELD]->(f)
                        """,
                        {
                            "struct_name": qualified_name,
                            "field_name": field_name,
                            "field_type": field_type
                        }
                    )
            
            # Process traits
            for trait_name, trait in items['traits'].items():
                qualified_name = f"{module_name}::{trait_name}"
                graph.run(
                    """
                    MERGE (t:RustTrait {qualified_name: $qualified_name})
                    ON CREATE SET 
                        t.name = $name,
                        t.visibility = $visibility,
                        t.start_line = $start_line,
                        t.end_line = $end_line,
                        t.file = $file_path,
                        t.repo = $repo_name
                    WITH t
                    MATCH (file:File {path: $file_path})
                    MERGE (file)-[:DECLARES_TRAIT]->(t)
                    """,
                    {
                        "qualified_name": qualified_name,
                        "name": trait_name,
                        "visibility": trait.visibility,
                        "start_line": trait.start_line,
                        "end_line": trait.end_line,
                        "file_path": str(file_path),
                        "repo_name": repo_name
                    }
                )
            
            # Process impls
            for impl_name, impl in items['impls'].items():
                graph.run(
                    """
                    MERGE (i:RustImpl {name: $name})
                    ON CREATE SET 
                        i.trait_name = $trait_name,
                        i.type_name = $type_name,
                        i.start_line = $start_line,
                        i.end_line = $end_line,
                        i.file = $file_path,
                        i.repo = $repo_name
                    WITH i
                    MATCH (file:File {path: $file_path})
                    MERGE (file)-[:DECLARES_IMPL]->(i)
                    """,
                    {
                        "name": f"{module_name}::{impl_name}",
                        "trait_name": impl.trait_name or "",
                        "type_name": impl.type_name,
                        "start_line": impl.start_line,
                        "end_line": impl.end_line,
                        "file_path": str(file_path),
                        "repo_name": repo_name
                    }
                )
   
            if 'imports' in items and items['imports']:
                print(f"\nProcessing {len(items['imports'])} imports for {file_path}")
                for import_path in items['imports']:
                    print(f"  - Import: {import_path}")
                    try:
                        # Create or update the import node and relationship in a single query
                        query = """
                            MERGE (f:File {path: $file_path})
                            MERGE (imp:RustImport {path: $import_path})
                            MERGE (f)-[r:IMPORTS]->(imp)
                            RETURN id(r) as rel_id
                        """
                        # Just execute the query, don't try to access the result
                        graph.run(query, {
                            "file_path": str(file_path),
                            "import_path": import_path
                        })
                        print(f"  ✓ Created/updated import for {import_path}")
                            
                    except Exception as e:
                        print(f"  ✗ Error processing import '{import_path}': {str(e)}")
                        import traceback
                        traceback.print_exc()
                        
            # Process call edges
            calls = collect_call_edges(tree, source)
            for call in calls:
                graph.run(
                    """
                    MATCH (caller:File {path: $file_path})
                    MATCH (callee:RustFunction {name: $func_name})
                    WHERE callee.file = $file_path
                    MERGE (caller)-[:CALLS]->(callee)
                    """,
                    {
                        "file_path": str(file_path),
                        "func_name": call
                    }
                )
                
    
    finally:
        graph.close()

def check_imports_in_db():
    with driver.session() as session:
        result = session.run("""
            MATCH (f:File)-[r:IMPORTS]->(i)
            RETURN f.path, i.path, count(*) as count
            LIMIT 10
        """)
        print("Import relationships in database:")
        for record in result:
            print(f"File: {record['f.path']} -> Import: {record['i.path']}")

if __name__ == "__main__":
    process_rust_repositories()
    # check_imports_in_db()

# if __name__ == "__main__":
#     # Test with a sample Rust file first

#     parser = get_parser("rust") 
#     test_file = "normalized_repos/rust-bitcoin/bitcoin/src/taproot/mod.rs"
#     with open(test_file, 'rb') as f:
#         source = f.read()
#     tree = parser.parse(source)
#     items = extract_rust_items(tree, source,test_file)
#     print("\nFound imports:")
#     cnt=0
#     for imp in items.get('imports', []):
#         cnt+=1
#         print(f"- {imp}")
#         print(cnt)
    
  