from mcp_clients.mcp_client import add_numbers
import asyncio

async def main():
    await add_numbers()

asyncio.run(main())