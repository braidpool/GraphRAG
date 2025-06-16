from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

def connect_to_neo4j():
    """Connect to Neo4j instance."""
    NEO4J_URI=os.getenv("NEO4J_URI")
    NEO4J_USERNAME=os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD=os.getenv("NEO4J_PASSWORD")
    NEO4J_DATABASE=os.getenv("NEO4J_DATABASE")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    return driver

if __name__ == "__main__":
    driver = connect_to_neo4j()
    print("Connected to Neo4j")