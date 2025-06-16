import requests
import os
from dotenv import load_dotenv

load_dotenv()

response = requests.get("https://french.braidpool.net:11435",
                        verify="/home/keshav/Downloads/rootCA.crt",
                        auth=(os.getenv("USER"), os.getenv("TOKEN")))
                        
print(response.content)