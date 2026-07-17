from agent_state.agent_state import AgentState
from helpers.file_operations import load_document
from helpers.chunk_helper import chunk_document
from helpers.vector_db import create_vector_store, load_vector_store, retrieve_documents
from helpers.rag_helper import generate_response

# Injest document
# - Read the content
# - Create chunks
# - Create embedings
# - Store in vector DB
def ingest_document(file_path: str):
    """
    Full ingestion pipeline.
    """
    documents = load_document(file_path)
    chunks = chunk_document(documents)
    vector_store = create_vector_store(chunks)

    return vector_store

# LLM to answer user query
# - Read the vectore DB
# - Retrive content using similarity search, returns context
# - Answer user query with the help of context 
def answer_question(query: str):
    """
    Full RAG pipeline.
    """

    vector_store = load_vector_store()

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

    if file_path:
        ingest_document(file_path)
        result = answer_question(user_input)
    else:
        result = "I can not find a document, please upload a document"

    return {
        "agent_result": [result],
        "completed_steps": ["file_reasoning"]
    }