import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Sanitize the metadat in chunks
def sanitize_metadata(chunks):

    for chunk in chunks:

        cleaned = {}

        for k, v in chunk.metadata.items():

            if isinstance(v, (str, int, float, bool, list)) or v is None:
                cleaned[k] = v

            else:
                cleaned[k] = json.dumps(v, default=str)

        chunk.metadata = cleaned

    return chunks

# Chunk the document and return chunks
def chunk_document(
        documents,
        chunk_size: int = 1000,
        chunk_overlap: int = 150
    ):
        """
        Split documents into chunks.
        """

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        chunks = splitter.split_documents(documents)
        
        return sanitize_metadata(chunks)
