from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv
import os
import json
import re
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Define the router agent configuration once
ROUTER_AGENT = Agent(
    name="Query Router",
    role="""
    You are an expert router that decides which agent is best suited to handle a user's query.
    Follow these guidelines:

    1. Use the Vector DB Agent (semantic search) for:
       - Conceptual questions about programming (e.g., 'how to', 'what is', 'explain')
       - Error messages and exceptions
       - Requesting code examples or tutorials
       - Natural language queries about code functionality

    2. Use the Cypher Query Agent (graph database) for:
       - Questions about code structure and relationships
       - Dependency and import relationships
       - Function/method call hierarchies
       - Class inheritance and interface implementations

    3. Use the MCP Based Search Agent:
       - When it seems like a general query about codes, errors, or concepts.
       - It seems more likely to be less specific and very generic which can be solved via using stackoverflow, github issues, etc   

    4. Its not compulsory to use only one agent, you can use multiple agents if required.
    
    5. List of output for agent field:
       - "VectorDB"
       - "Cypher"
       - "MCP Based Search"
       - "VectorDB and MCP Based Search"
       - "Cypher and MCP Based Search"
       - "VectorDB and Cypher"
       - "VectorDB and Cypher and MCP Based Search"
       
    6. Always provide a brief explanation of your routing decision.

    Output format should be in JSON:

    ```json
    {
        "agent": "agent",
        "explanation": "Brief explanation of your routing decision"
    }
    ```
    """,
    model = Groq(id="qwen/qwen3-32b", temperature=0.3, max_tokens=1024, top_p=0.95),
)



def query_router(query: str) -> dict:
    """
    Sends the user query to the router agent and returns a parsed JSON decision.
    """
    raw = ROUTER_AGENT.run(query)
    response = raw.content
    
    # Extract JSON content using regex
    json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON found in agent response: {response}")
    
    try:
        # Parse the JSON string into a Python dictionary
        json_str = json_match.group(1).strip()
        decision = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse routing decision: {e}\nResponse: {response}")
    
    return decision


def interactive_router():
    """
    Runs an interactive loop: gets user query, provides routing, and asks for feedback.
    """
    user_query = "how to fix pydantic error"   
    decision = query_router(user_query)
    print("\nRouting Decision:")
    print(decision)


if __name__ == "__main__":
    interactive_router()  # Start the interactive feedback-enabled router
