import asyncio
import os

from agno.agent import Agent
from agno.tools.mcp import MultiMCPTools
from agno.models.openai import OpenAIChat
from dotenv import load_dotenv

load_dotenv()

class SequentialMemory:
    def __init__(self):
        """Initializes the memory with an empty list to store conversation history."""
        self.history = []

    def add_message(self, user_input: str, ai_response: str):
        """
        Adds a new user-AI interaction to the history.
        Each interaction is stored as two dictionary entries in the list.
        """
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": ai_response})

    def get_context(self, query: str = None) -> str:
        """
        Retrieves the entire conversation history and formats it into a single
        string to be used as context for the LLM. The 'query' parameter is ignored
        as this strategy always returns the full history.
        """
        # Join all messages into a single newline-separated string.
        return "\n".join([f"{turn['role'].capitalize()}: {turn['content']}" for turn in self.history])

    def clear(self):
        """Resets the conversation history by clearing the list."""
        self.history = []
        print("Sequential memory cleared.")


# Create a global memory instance to persist across function calls
memory = SequentialMemory()

async def run_agent(message: str) -> None:
    global memory
    
    # Get conversation context from memory
    context = memory.get_context()
    
    # Add context to agent instructions if there's conversation history
    agent_instructions = """You are a coding assistant provided with tools for codegraph(which has access to knowledge graph of code repository) and context7 documentation search tool (to help fix depreciation and related syntax errors) 
    Your task is use these tools appropriately to help user code effectively and resolve any errors"""
    
    if context:
        agent_instructions += f"\n\nHere is the conversation history so far:\n{context}"
    
    env = {
        **os.environ,
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    }

    async with MultiMCPTools(
        [
        "npx -y mcp-code-graph@latest tensorzero/tensorzero",
        "npx -y @upstash/context7-mcp",
        # "npx -y @mnhlt/WebSearch-MCP",
        ],
        env=env,
    ) as mcp_tools:
        agent = Agent(
            model=OpenAIChat(id="gpt-4.1-mini", api_key=os.getenv("OPENAI_API_KEY")),
            tools=[mcp_tools],
            markdown=True,
            instructions=agent_instructions,
            show_tool_calls=True,
        )

        import io
        import sys

        # Redirect stdout to capture the agent's response
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        try:
            # This will now print into our captured_output buffer
            await agent.aprint_response(message, stream=True)
        finally:
            # Restore stdout
            sys.stdout = old_stdout

        # Get the full response from the buffer
        full_response = captured_output.getvalue()

        # Store the interaction in memory
        memory.add_message(message, full_response)
        # print(memory.get_context())
        # Finally, print the captured response to the real console
        print(full_response, end="")


# Example usage
if __name__ == "__main__":
    # Interactive conversation loop
    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\nExiting conversation.")
                break
            elif user_input.lower() == "clear memory":
                memory.clear()
                continue
                
            asyncio.run(run_agent(user_input))
    except KeyboardInterrupt:
        print("\n\nConversation ended by user.")
