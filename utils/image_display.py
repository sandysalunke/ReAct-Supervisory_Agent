import re
from pathlib import Path


_LOCAL_IMAGE_PATH_PATTERN = re.compile(
    r"(?<![\w])(?:[A-Za-z]:)?(?:[\\/]|\.{1,2}[\\/])?"
    r"(?:[\w.-]+[\\/])*[\w.-]+\.(?:png|jpe?g|gif|webp)(?![\w])",
    re.IGNORECASE,
)


def split_local_image_paths(message: str) -> list[tuple[str, Path | None]]:
    """Split a message into text and existing local image paths."""
    segments: list[tuple[str, Path | None]] = []
    text_start = 0

    for match in _LOCAL_IMAGE_PATH_PATTERN.finditer(message):
        image_path = Path(match.group())
        if not image_path.is_file():
            continue

        if match.start() > text_start:
            segments.append((message[text_start : match.start()], None))
        segments.append(("", image_path))
        text_start = match.end()

    if text_start < len(message):
        segments.append((message[text_start:], None))

    return segments or [(message, None)]
