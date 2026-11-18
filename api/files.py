from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.paths import DATA_DIR

router = APIRouter(tags=["files"])

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def image_url(image_path: Path) -> str | None:
    """Return the API URL for a generated image, or None if it is not servable."""
    try:
        relative_path = image_path.resolve().relative_to(DATA_DIR.resolve())
    except ValueError:
        return None
    if image_path.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    return "/api/files/" + quote(relative_path.as_posix())


@router.get("/files/{file_path:path}")
async def get_file(file_path: str):
    data_dir = DATA_DIR.resolve()
    resolved_path = (data_dir / file_path).resolve()
    if (
        not resolved_path.is_relative_to(data_dir)
        or resolved_path.suffix.lower() not in IMAGE_SUFFIXES
        or not resolved_path.is_file()
    ):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(resolved_path)
