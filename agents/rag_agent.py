from agent_state.agent_state import AgentState
from helpers.file_operations import load_document, get_file_hash
from helpers.chunk_helper import chunk_document
from helpers.vector_db import create_vector_store, load_vector_store, retrieve_documents, collection_exists
from helpers.rag_helper import generate_response

# Injest document
# - Read the content
# - Create chunks
# - Create embedings
# - Store in vector DB
def ingest_document(file_path: str, collection_name):
    """
    Full ingestion pipeline.
    """
    documents = load_document(file_path)
    chunks = chunk_document(documents)
    vector_store = create_vector_store(collection_name,chunks)

    return vector_store

# LLM to answer user query
# - Read the vectore DB
# - Retrive content using similarity search, returns context
# - Answer user query with the help of context 
def answer_question(collection_name, query: str):
    """
    Full RAG pipeline.
    """

    vector_store = load_vector_store(collection_name)

    retrieved_docs = retrieve_documents(
        query,
        vector_store
    )
    
    return generate_response(
        query,
        retrieved_docs
    )

# rag_agent - Added as node in supervisor agent graph
def rag_agent(state: AgentState) -> AgentState:
    """Use this agent for RAG - reasoning based on file content"""
    print("\n===== RAG AGENT =====")

    file_path = state["uploaded_file"]
    user_input = state["user_input"]
    result = "I can not find a document, please upload a document"

    if file_path:
        collection_name = get_file_hash(file_path)

        if not collection_exists(collection_name):
            ingest_document(file_path, collection_name)

        result = answer_question(collection_name, user_input)
        
    print("\n===== result =====", result)

    return {
        "agent_result": [result],
        "completed_steps": ["file_reasoning"]
    }