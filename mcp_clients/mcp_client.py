from fastmcp import Client
from langchain.tools import tool

@tool
async def add_numbers():
    """Tool to add two numbers using the Math Server."""
    
    client = Client("http://localhost:8000/mcp")

    async with client:
        result = await client.call_tool(
            "add",
            {
                "a": 10,
                "b": 20
            }
        )

        print(result)

