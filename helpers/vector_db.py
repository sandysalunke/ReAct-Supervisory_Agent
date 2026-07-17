from langchain_chroma import Chroma
from chromadb import PersistentClient
from helpers.embeddings import AzureEmbeddingWrapper

persist_directory = "./data/vectorDB"
COLLECTION_NAME = "rag_docs"

# Check if the collection aready exist in vector DB
def collection_exists(collection_name: str):

    client = PersistentClient(path=persist_directory)

    try:
        client.get_collection(collection_name)
        return True
    except Exception:
        return False

# Create embedings and store in vector SB
def create_vector_store(chunks):
    """
    Create embeddings and store them in Chroma.
    """
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=AzureEmbeddingWrapper(),
        persist_directory=persist_directory,
        collection_name = COLLECTION_NAME
    )

    return vector_store

# Load the vector store
def load_vector_store():
    """
    Load an existing vector store.
    """

    return Chroma(
        persist_directory=persist_directory,
        collection_name = COLLECTION_NAME,
        embedding_function=AzureEmbeddingWrapper()
    )

# Retrive similar chunks using Similarity search 
# Those chunks will be used as context to answer user query
def retrieve_documents(
    query: str,
    vector_store,
    k: int = 5
):
    """
    Retrieve relevant chunks.
    """

    return vector_store.similarity_search(
        query=query,
        k=k
    )

# Retrive contnet from the document using Similarity search along with match score
def retrieve_documents_with_scores(
    query: str,
    vector_store,
    k: int = 5
):
    return vector_store.similarity_search_with_score(
        query=query,
        k=k
    )
