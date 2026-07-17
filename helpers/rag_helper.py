import json
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from unstructured.partition.auto import partition
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.azure_config import llm, EMBED_MODEL
from helpers.embeddings import AzureEmbeddingWrapper

persist_directory = "./data/vectorDB"
COLLECTION_NAME = "rag_docs"

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

# Retrive contnet from the document using Similarity search 
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

# Build the context
def build_context(
    retrieved_docs
):
    """
    Convert retrieved documents into a prompt context.
    """

    return "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

# Generate the response
# Uses LLM along with context and user prompt  
def generate_response(
    query: str,
    retrieved_docs
):
    """
    Generate an answer using retrieved chunks.
    """

    context = build_context(retrieved_docs)

    prompt = f"""
        Use the provided context to answer the question.

        Context:
        {context}

        Question:
        {query}
    """

    response = llm.invoke(prompt)

    return response.content