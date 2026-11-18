from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api import chat, dashboards, files, mcp
from api.paths import ROOT

FRONTEND_DIST = ROOT / "frontend" / "dist" / "frontend" / "browser"

app = FastAPI(title="GenAI Copilot API")

# The Angular dev server proxies /api to this app, so CORS is only needed
# when the frontend calls the API directly from another local origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")
app.include_router(dashboards.router, prefix="/api")
app.include_router(files.router, prefix="/api")
app.include_router(mcp.router, prefix="/api")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# After `npm run build` in frontend/, the API also serves the Angular app, so a
# single process is enough outside development.
if FRONTEND_DIST.is_dir():

    @app.get("/{path:path}", include_in_schema=False)
    async def frontend(path: str):
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found.")
        dist_dir = FRONTEND_DIST.resolve()
        asset = (dist_dir / path).resolve()
        if asset.is_relative_to(dist_dir) and asset.is_file():
            return FileResponse(asset)
        # Client-side routes (e.g. /usage) are resolved by the Angular router.
        return FileResponse(dist_dir / "index.html")
