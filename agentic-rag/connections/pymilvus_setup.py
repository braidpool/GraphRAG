from pymilvus import MilvusClient
import os
from dotenv import load_dotenv

load_dotenv()

def connect_to_milvus():
    MILVUS_URI = os.getenv("MILVUS_URI")
    MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")

    # Connect to Milvus Cloud
    client = MilvusClient(
    uri=MILVUS_URI,
    token=MILVUS_TOKEN
)
    return client

if __name__ == "__main__":
    client = connect_to_milvus()
    print("Successfully connected to Milvus Cloud!")