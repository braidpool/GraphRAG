import asyncio
from fastmcp import Client

async def test():
    # Use async context manager to handle client lifecycle
    async with Client("agents/mcp_search.py") as client:
        try:
            # Call the search_all tool and await the result
            result = await client.call_tool("search_all", {
                "query": "how to kill process in linux",
                "limit": 3
            })
            print("Search results:")
            print(result)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test())
