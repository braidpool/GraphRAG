import requests
from requests.auth import HTTPBasicAuth
import os
from dotenv import load_dotenv

load_dotenv()

def use_ollama_llm(text: str):
    url = "https://french.braidpool.net:11435/api/embed"
    auth = HTTPBasicAuth(os.getenv("USER"), os.getenv("TOKEN"))
    data = {
        "model": "manutic/nomic-embed-code",
        "input": text,
        "stream": False
    }
    response = requests.post(url, json=data, auth=auth, verify="/home/keshav/Downloads/rootCA.crt")
    # print(response.json())
    return response.json()['embeddings'][0]

def use_ollama_llm_normal(system_prompt,context, model: str = "qwen3:4b"):
    url = "https://french.braidpool.net:11435/api/chat"
    auth = HTTPBasicAuth(os.getenv("USER"), os.getenv("TOKEN"))
    data = {
        "model": model,
        "messages": [
            {
            "role": "system",
            "content": system_prompt
            },
            {
            "role": "user", 
            "content": context 
            }
        ],
        "stream": False
        }
    response = requests.post(url, json=data, auth=auth, verify="/home/keshav/Downloads/rootCA.crt")
    # print(response.json())
    return response.json()['message']['content']

if __name__ == "__main__":
    print(use_ollama_llm_normal("Hello World","Hello World"))
    