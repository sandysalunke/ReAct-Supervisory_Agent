from agent_state.agent_state import AgentState
from helpers.file_operations import load_document, get_file_hash
from helpers.chunk_helper import chunk_document
from helpers.vector_db import create_vector_store, load_vector_store, retrieve_documents, collection_exists
from helpers.rag_helper import generate_response
from pathlib import Path

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

    user_input = state.get("user_input")
    task_prompt = state.get("task_prompt", user_input)
    current_task = state.get("current_task", {})
    result = "I can not find a document, please upload a document"

    file_extensions = { ".txt", ".pdf", ".docx", ".doc", ".ppt", ".pptx"}
    file_path = state.get("uploaded_file", "")
    file_extension = Path(file_path).suffix

    if file_path and file_extension in file_extensions:
        collection_name = get_file_hash(file_path)

        if not collection_exists(collection_name):
            ingest_document(file_path, collection_name)

        result = answer_question(collection_name, task_prompt)
        
    return {
        "agent_result": [result],
        "completed_steps": ["file_reasoning"],
        "task_results": {
            current_task["id"] : {
                "intent": current_task["intent"],
                "result": result
            }
        }
    }