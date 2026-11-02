from fastmcp import FastMCP
import traceback

mcp = FastMCP("Math Server")

@mcp.tool
def add(a: int, b: int) -> int:
    return a + b

if __name__ == "__main__":
    try:
        mcp.run(
            transport="http",
            host="0.0.0.0",
            port=8000
        )
    except Exception:
        traceback.print_exc()