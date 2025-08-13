import torch
import numpy as np
from typing import List, Optional, Dict, Any
import os

class SemanticCodeSearcher:
    """Class to handle semantic code search"""

    # Edge weights for propagation (do not include CONTAINS)
    EDGE_WEIGHTS = {
        "CALLS": 0.2
        # Add more edge types as needed
    }

    def __init__(self, embedding_model, embedding_tokenizer, driver):
        self.embedding_model = embedding_model
        self.embedding_tokenizer = embedding_tokenizer
        self.driver = driver

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

    def _convert_to_regex_pattern(self, pattern: str) -> str:
        """Convert a pattern to regex format, handling glob-like patterns"""
        if not pattern:
            return ""
        
        # If it's already a regex pattern (contains regex special chars), use as-is
        if any(char in pattern for char in ['^', '$', '[', ']', '(', ')', '+', '?', '*', '|']):
            return pattern
        
        # Convert glob-like patterns to regex
        regex_pattern = pattern.replace('.', r'\.').replace('*', '.*').replace('?', '.')
        return regex_pattern

    def semantic_search_with_propagation(
        self, 
        working_dir,
        query: str,  
        file_names: Optional[List[str]] = None, 
        class_contexts: Optional[List[str]] = None, 
        top_k: int = 10
    ):
        """
        Perform semantic similarity search and propagate scores through the graph.
        Args:
            query (str): The natural language search query.
            file_names (list, optional): List of regex patterns for file path filtering. 
                                       Supports glob-like patterns (*, ?) and regex patterns.
                                       Defaults to None.
            class_contexts (list, optional): List of regex patterns for class context filtering.
                                           Supports glob-like patterns (*, ?) and regex patterns.
                                           Defaults to None.
            top_k (int, optional): The total number of results to return. Defaults to 10.
            
        Returns:
            list: A list of top-k matching nodes with their details.
        """
        if not self.driver:
            raise ConnectionError("Neo4j driver not initialized.")

        query_embedding = self.embed_text(query)

        cypher_query = """
        MATCH (n)
        WHERE (n:Function OR n:Class)
          AND n.embedding IS NOT NULL
        """
        params = {}

        if file_names:
            # Convert file_names to regex patterns for flexible matching
            file_patterns = []
            for pattern in file_names:
                if pattern:
                    regex_pattern = self._convert_to_regex_pattern(pattern)
                    if regex_pattern:
                        file_patterns.append(f"n.file_path =~ '.*{regex_pattern}.*'")
            
            if file_patterns:
                cypher_query += f" AND ({' OR '.join(file_patterns)})"
        
        if class_contexts:
            # Convert class_contexts to regex patterns for flexible matching
            class_patterns = []
            for pattern in class_contexts:
                if pattern:
                    regex_pattern = self._convert_to_regex_pattern(pattern)
                    if regex_pattern:
                        class_patterns.append(f"n.context =~ '.*{regex_pattern}.*'")
            
            if class_patterns:
                cypher_query += f" AND ({' OR '.join(class_patterns)})"
        
        # Restrict results to files under the working directory, if provided
        if working_dir:
            cypher_query += " AND n.file_path STARTS WITH $working_dir"
            params["working_dir"] = working_dir
            
        cypher_query += " RETURN n, n.embedding AS embedding, elementId(n) AS node_id"

        with self.driver.session() as session:
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
        with self.driver.session() as session:
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
            # Truncate code to first 7 lines if 'source' exists
            code = entry["node"].get("source")
            if code:
                lines = code.splitlines()
                short_code = "\n".join(lines[:7])
                if len(lines) > 7:
                    short_code += "\n... (truncated)"
                entry["node"]["source"] = short_code

            # Include working directory and full file path
            file_path = entry["node"].get("file_path")
            if file_path is not None:
                if working_dir:
                    full_path = file_path if os.path.isabs(file_path) else os.path.join(working_dir, file_path)
                else:
                    full_path = file_path
                entry["node"]["working_dir"] = working_dir
                entry["node"]["full_file_path"] = full_path
        
        return final_ranked_nodes[:top_k]
