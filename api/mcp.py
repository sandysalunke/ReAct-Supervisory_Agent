from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["mcp"])


@router.post("/mcp/add-numbers")
async def run_add_numbers():
    """Run the sample MCP client against the local Math Server."""
    from mcp_clients.mcp_client import add_numbers

    try:
        result = await add_numbers.ainvoke({})
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Could not call the Math MCP server: {error}",
        ) from error
    return {"result": result}
