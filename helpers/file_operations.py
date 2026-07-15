from datetime import datetime
from pathlib import Path

UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def save_uploaded_file(uploaded_file):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    extension = Path(uploaded_file.name).suffix
    filename = f"{timestamp}{extension}"

    file_path = UPLOAD_DIR / filename

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(file_path)

