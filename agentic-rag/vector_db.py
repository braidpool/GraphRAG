from pymilvus import (
    connections,
    utility,
    FieldSchema, 
    CollectionSchema, 
    DataType,
    Collection,
)

# Connect to local Milvus server
def connect_to_milvus(host='localhost', port='19530'):
    """Connect to a local Milvus server."""
    try:
        connections.connect(
            "default",
            host=host,
            port=port
        )
        print(f"Successfully connected to Milvus at {host}:{port}")
        return True
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return False

def create_code_collection(collection_name="code_nodes", dim=3584):
    """Create a collection for code nodes with appropriate schema and indexes."""
    if utility.has_collection(collection_name):
        print(f"Collection {collection_name} already exists. Dropping it first.")
        utility.drop_collection(collection_name)
    
    # Define the schema
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="qualified_name", dtype=DataType.VARCHAR, max_length=1500),
        FieldSchema(name="node_type", dtype=DataType.VARCHAR, max_length=1000),  # function, class, struct, trait, etc.
        FieldSchema(name="code", dtype=DataType.VARCHAR, max_length=65000),
        FieldSchema(name="docstring", dtype=DataType.VARCHAR, max_length=65000),
        FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=500),  # rust, python, etc.
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="imported_from", dtype=DataType.VARCHAR, max_length=1500),
        FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=1500),
    ]
    
    schema = CollectionSchema(
        fields=fields,
        description="Code nodes with vector embeddings for semantic search",
        enable_dynamic_field=True  # Allow additional fields
    )
    
    # Create collection
    collection = Collection(
        name=collection_name,
        schema=schema
    )
    
    print(f"Created collection: {collection_name}")
    
    # Create index for vector search
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 1024}
    }
    
    collection.create_index("embedding", index_params)
    
    return collection

def main():
    # Connect to local Milvus
    if not connect_to_milvus():
        return
    
    # Create the collection
    collection = create_code_collection()
    
    # # Example of inserting data
    # # Note: In a real scenario, you'd generate embeddings using a model
    # example_data = [
    #     {
    #         "id": "func1",
    #         "qualified_name": "module.function_name",
    #         "node_type": "function",
    #         "code": "def example(): pass",
    #         "file_path": "/path/to/file.py",
    #         "language": "python",
    #         "embedding": [0.1] * 768  # Replace with real embeddings
    #     }
    # ]
    
    # # # Insert data
    # try:
    #     collection.insert(example_data)
    #     print("Successfully inserted example data")
    # except Exception as e:
    #     print(f"Error inserting data: {e}")

if __name__ == "__main__":
    main()