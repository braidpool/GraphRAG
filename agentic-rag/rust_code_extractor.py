# def extract_rust_snippet(file_path, node_name, node_type):
#     """
#     Extract a code snippet from a Rust file using tree-sitter.
    
#     Args:
#         file_path: Path to the Rust file
#         node_name: Name of the function/struct/trait to extract
#         node_type: Type of node ('function', 'struct', 'trait', 'impl', 'module', 'import')
        
#     Returns:
#         str: The extracted code snippet or None if not found
#     """
#     from pathlib import Path
#     from tree_sitter_language_pack import get_parser
    
#     # Convert to string if it's a Path object
#     file_path = str(file_path)
    
#     # Handle path normalization
#     file_path = file_path.replace('\\', '/')
#     parts = file_path.split('/')
#     try:
#         idx = parts.index('normalized_repos')
#         file_path = '/'.join(parts[idx:])
#     except ValueError:
#         pass
    
#     # Construct the full path
#     base_dir = Path("/home/keshav/Downloads/graph_rag_braidpool")
#     full_path = base_dir / file_path.lstrip('/')
    
#     if not full_path.exists():
#         print(f"File not found: {full_path}")
#         return None
    
#     try:
#         with open(full_path, "rb") as f:
#             source = f.read()
            
#         # Initialize Rust parser
#         parser = get_parser("rust")
#         tree = parser.parse(source)
        
#         def get_node_text(node):
#             return source[node.start_byte:node.end_byte].decode('utf-8')
        
#         def find_node(node, target_name, target_type):
#             if node.type == target_type and node.child_by_field_name('name'):
#                 current_name = get_node_text(node.child_by_field_name('name')).strip()
#                 if current_name == target_name:
#                     return node
            
#             for child in node.children:
#                 result = find_node(child, target_name, target_type)
#                 if result:
#                     return result
#             return None
        
#         # Handle different node types
#         if node_type == "function":
#             target_node = find_node(tree.root_node, node_name, "function_item")
#             if target_node:
#                 return get_node_text(target_node)
                
#         elif node_type == "struct":
#             target_node = find_node(tree.root_node, node_name, "struct_item")
#             if target_node:
#                 return get_node_text(target_node)
                
#         elif node_type == "trait":
#             target_node = find_node(tree.root_node, node_name, "trait_item")
#             if target_node:
#                 return get_node_text(target_node)
                
#         elif node_type == "impl":
#             target_node = find_node(tree.root_node, node_name, "impl_item")
#             if target_node:
#                 return get_node_text(target_node)
                
#         elif node_type == "module":
#             # Return the entire file content for module
#             return source.decode('utf-8')
            
#         elif node_type == "import":
#             # Collect all use declarations
#             imports = []
#             def collect_imports(node):
#                 if node.type == "use_declaration":
#                     imports.append(get_node_text(node))
#                 for child in node.children:
#                     collect_imports(child)
            
#             collect_imports(tree.root_node)
#             return "\n".join(imports) if imports else None
            
#     except Exception as e:
#         print(f"Error processing {full_path}: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return None
    
#     print(f"No matching {node_type} '{node_name}' found in {full_path}")
#     return None
from pathlib import Path
from tree_sitter_language_pack import get_parser
import os

def extract_rust_snippet(source_str, item_name, item_type):
    """
    Extract a code snippet for a Rust item from its source file.
    
    Args:
        source_str: The source code as a string
        item_name: Name of the item to extract
        item_type: Type of the item ('RustFunction', 'RustStruct', 'RustTrait', 'RustImpl')
        
    Returns:
        str: The extracted code snippet, or None if not found
    """
    try:
        def get_text(node):
            return source_str[node.start_byte:node.end_byte]
        
        def print_node(node, depth=0):
            indent = "  " * depth
            # print(f"{indent}Node type: {node.type}")
            # print(f"{indent}Node text: {get_text(node)[:50]}...")
            # print(f"{indent}Node children: {len(node.children)}")
        
        # Get the parser and parse the source
        parser = get_parser("rust")
        tree = parser.parse(source_str.encode('utf-8'))
        
        # Find the item in the AST
        def find_item(node, depth=0):
            # print_node(node, depth)
            
            # Handle imports first since they're special
            if item_type == "RustImport":
                if node.type == "use_declaration":
                    return get_text(node)
                for child in node.children:
                    result = find_item(child, depth + 1)
                    if result:
                        return result
                return None

            # For other node types, map Rust types to tree-sitter types
            tree_sitter_type = {
                'RustFunction': 'function_item',
                'RustStruct': 'struct_item',
                'RustTrait': 'trait_item',
                'RustImpl': 'impl_item',
                'RustModule': 'module_item'
            }.get(item_type)
            
            if not tree_sitter_type:
                return None
            
            if node.type == tree_sitter_type:
                name_node = node.child_by_field_name('name')
                if name_node and get_text(name_node) == item_name:
                    return node
            
            # For impls, we need to check both trait and type names
            if tree_sitter_type == 'impl_item':
                trait_node = node.child_by_field_name('trait')
                type_node = node.child_by_field_name('type')
                if type_node and get_text(type_node) == item_name:
                    return node
                elif trait_node and get_text(trait_node) == item_name:
                    return node
            
            # Recursively check children for all node types
            for child in node.children:
                found = find_item(child, depth + 1)
                if found:
                    return found
            return None
        
        item_node = find_item(tree.root_node)
        if item_type == "RustImport":
            return item_node if isinstance(item_node, str) else None
        elif item_node:
            return get_text(item_node)
            
    except Exception as e:
        print(f"Error extracting code from {item_name}: {e}")
        import traceback
        traceback.print_exc()
    
    return None
# Example usage:
snippet = extract_rust_snippet(
    "/home/keshav/Downloads/graph_rag_braidpool/normalized_repos/rust-miniscript/examples/big.rs",
    "main",
    "import"
)
print(snippet)