import os
import re
from typing import List, Dict, Any, Tuple, Optional

def grep_search(
    query: str,
    case_sensitive: bool = True,
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
    working_dir: str = ""
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Search through files for specific patterns using regex.
    
    Args:
        query: Regex pattern to find
        case_sensitive: Whether the search is case sensitive
        include_pattern: Glob pattern for files to include (e.g., "*.py")
        exclude_pattern: Glob pattern for files to exclude
        working_dir: Directory to search in (defaults to current directory if empty)
        
    Returns:
        Tuple of (list of matches, success status)
        Each match contains:
        {
            "file": file path,
            "line_number": line number (1-indexed),
            "content": matched line content
        }
    """
    results = []
    search_dir = working_dir if working_dir else "."
    
    try:
        # Compile the regex pattern
        try:
            pattern = re.compile(query, 0 if case_sensitive else re.IGNORECASE)
        except re.error as e:
            print(f"Invalid regex pattern: {str(e)}")
            return [], False
        
        # Convert glob patterns to regex for file matching
        include_regexes = _glob_to_regex(include_pattern) if include_pattern else None
        exclude_regexes = _glob_to_regex(exclude_pattern) if exclude_pattern else None
        
        # Walk through the directory and search files
        for root, _, files in os.walk(search_dir):
            for filename in files:
                # Skip files that don't match inclusion pattern
                if include_regexes and not any(r.match(filename) for r in include_regexes):
                    continue
                
                # Skip files that match exclusion pattern
                if exclude_regexes and any(r.match(filename) for r in exclude_regexes):
                    continue
                
                file_path = os.path.join(root, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for i, line in enumerate(f, 1):
                            if pattern.search(line):
                                results.append({
                                    "file": file_path,
                                    "line_number": i,
                                    "content": line.rstrip()
                                })
                                
                                # Limit to 50 results
                                if len(results) >= 50:
                                    break
                except Exception:
                    # Skip files that can't be read
                    continue
                
                if len(results) >= 50:
                    break
            
            if len(results) >= 50:
                break
        
        return results, True
    
    except Exception as e:
        print(f"Search error: {str(e)}")
        return [], False

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

def list_code_elements(
    element_type=None, 
    include_pattern: Optional[str] = None, 
    exclude_pattern: Optional[str] = None,
    driver = None,
    limit=50
):
    """
    List functions, classes, and/or methods with flexible filters, including file include/exclude patterns.
    
    Args:
        element_type: The type of element to find ('function', 'class', 'method').
        include_pattern: A regex pattern for files to include (e.g., ".*\.py").
        exclude_pattern: A regex pattern for files to exclude (e.g., ".*_test\.py").
        limit: The maximum number of results to return.
    
    Returns:
        A list of dictionaries with brief element details.
    """
    if driver is None:
        return [], False
    
    with driver.session() as session:
        queries = []
        params = {"limit": limit}
        results = []
        
        # Normalize element_type
        if element_type is None:
            element_type = ["function", "class", "method"]
        elif isinstance(element_type, str):
            element_type = [element_type]
        element_type = set([e.lower() for e in element_type])

        # Build the file filtering clause
        file_filter_clauses = []
        if include_pattern:
            file_filter_clauses.append(f"{{file_path_prop}} =~ $include_pattern")
            params["include_pattern"] = f"(?i).*{include_pattern}"
        if exclude_pattern:
            file_filter_clauses.append(f"NOT {{file_path_prop}} =~ $exclude_pattern")
            params["exclude_pattern"] = f"(?i).*{exclude_pattern}"

        where_clause = ""
        if file_filter_clauses:
            where_clause = "WHERE " + " AND ".join(file_filter_clauses)
            
        # List classes
        if "class" in element_type:
            q = f"""
                MATCH (c:Class)
                {where_clause.replace('{file_path_prop}', 'c.file_path')}
                RETURN 'class' as type, c.name as name, c.file_path as file_path, c.line_number as line_number, c.docstring as docstring, null as args
                ORDER BY c.file_path, c.line_number
                LIMIT $limit
            """
            queries.append((q, dict(params)))

        # List functions (top-level only)
        if "function" in element_type:
            function_where = where_clause.replace('{file_path_prop}', 'f.file_path')
            
            # This logic correctly handles combining the NOT and file filters
            if function_where:
                 function_where = f"WHERE NOT ( (:Class)-[:CONTAINS]->(f) ) AND ({function_where[6:]})"
            else:
                 function_where = "WHERE NOT ( (:Class)-[:CONTAINS]->(f) )"

            q = f"""
                MATCH (f:Function)
                {function_where}
                RETURN 'function' as type, f.name as name, f.file_path as file_path, f.line_number as line_number, f.docstring as docstring, f.args as args
                ORDER BY f.file_path, f.line_number
                LIMIT $limit
            """
            queries.append((q, dict(params)))

        # List methods (functions contained in a class)
        if "method" in element_type:
            method_where = where_clause.replace('{file_path_prop}', 'm.file_path')
            
            q = f"""
                MATCH (c:Class)-[:CONTAINS]->(m:Function)
                {method_where}
                RETURN 'method' as type, m.name as name, m.file_path as file_path, m.line_number as line_number, m.docstring as docstring, m.args as args, c.name as class_name
                ORDER BY m.file_path, m.line_number
                LIMIT $limit
            """
                
            queries.append((q, dict(params)))

        for q, p in queries:
            for record in session.run(q, **p):
                d = dict(record)
                # Truncate docstring for brevity
                if d.get("docstring"):
                    d["docstring"] = d["docstring"][:200] + ("..." if len(d["docstring"]) > 200 else "")
                results.append(d)
        
        print(results)
        
        return results[:limit], True