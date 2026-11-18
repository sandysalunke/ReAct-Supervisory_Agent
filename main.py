"""Starts the GenAI Copilot API. Run from anywhere with `python main.py`."""
import os
from pathlib import Path

import uvicorn

if __name__ == "__main__":
    # Agents read and write ./data relative to the project root.
    os.chdir(Path(__file__).resolve().parent)
    uvicorn.run(
        "api.app:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        # 8001 because the sample Math MCP server listens on 8000.
        port=int(os.getenv("API_PORT", "8001")),
    )
