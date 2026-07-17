from datetime import datetime
from pathlib import Path
import hashlib
from langchain_core.documents import Document
from unstructured.partition.auto import partition

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


# Read the document and return content
def load_document(file_path):

    elements = partition(filename=file_path)

    # join all elements together to create single document
    full_text = "\n".join(
        element.text.strip()
        for element in elements
        if getattr(element, "text", "").strip()
    )

    return [
        Document(
            page_content=full_text,
            metadata={
                "source": file_path
            }
        )
    ]

# Create hash for the file binary content
# To prevent duplicate file collections in vector DB 
def get_file_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()
