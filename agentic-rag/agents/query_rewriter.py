from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv
import os
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
def create_rewriter_agent(query: str):
    response = Agent(
        name="Query Rewriter",
        role="Takes ambiguous, grammatically incorrect, or shorthand user queries and rewrites them for clarity and precision",
        model=Groq(id="qwen/qwen3-32b"),
        show_tool_calls=True,
        # debug_mode=True
    )
    output = response.run(query)
    return output.content.split("</think>")[1].strip()

if __name__ == "__main__":
    test_query = "how 2 fix err in pythn code getting none type errorrr?"
    # We run the agent with the test query and expect a rewritten query
    rewriter = create_rewriter_agent(f"Rewrite the following query to be more clear, grammatically correct, and unambiguous: {test_query}. AND STRICTLY DO NO PRINT ANYTHING ELSE OTHER THAN RE-WRITTEN QUERY")
    print(rewriter)
