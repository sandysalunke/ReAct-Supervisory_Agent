import os
from dotenv import load_dotenv
from agent_state.agent_state import AgentState
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from helpers.hybrid_rag_helper import generate_response
from utils.tracing import tracer
from opentelemetry import trace
from helpers.llm_cache import get_cached_response, get_semantic_cached_response
from helpers.embeddings import AzureEmbeddingWrapper
from helpers.llm_cache import set_cached_response


load_dotenv()

# RBAC - Filter by allowed user groups
# user_groups = "Business,Executives"
# user_groups = "HR,Manager"
user_groups = "Employees"
# user_groups = "Engineering,Architect,Developers"

# Configuration
endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
index_name = os.getenv("AZURE_SEARCH_INDEX")
api_key = os.getenv("AZURE_SEARCH_KEY")

search_client = SearchClient(
    endpoint=endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(api_key),
    connection_verify=False
)

def hybrid_search(question: str, top: int = 5):
    # Trace the chunking
    with tracer.start_as_current_span("agent.hybrid_rag.chunks", attributes={"chunk_count": top}):
        vector_query = VectorizableTextQuery(
            text=question,
            k_nearest_neighbors=10,
            fields="content_vector"
        )

        results = search_client.search(
            search_text=question,
            vector_queries=[vector_query],
            filter= f"allowed_user_groups eq '{user_groups}'",
            query_type="semantic",
            semantic_configuration_name="knowledge-base-search-semantic-configuration",
            select=[
                "chunk",
                "title",
                "metadata_storage_path"
            ],
            top=top
        )
        results = list(results)

        trace.get_current_span().set_attribute("agent.hybrid_rag.chunks", len(results))
        trace.get_current_span().set_attribute(
            "agent.hybrid_rag.sources",
            len(set(r.get("title") for r in results if r.get("title")))
        )

    return results

# rag_agent - Added as node in supervisor agent graph
def hybrid_rag_agent(state: AgentState) -> AgentState:
    """Use this agent for RAG - reasoning based on file content"""
    print("\n===== AZURE AI SEARCH AGENT =====")

    user_input = state.get("user_input")
    task_prompt = state.get("task_prompt", user_input)
    current_task = state.get("current_task", {})

    # Trace the overall agent execution
    with tracer.start_as_current_span("agent.hybrid_rag", attributes={
        "thread_id": state.get("thread_id"),
        "task_id": current_task.get("id"),
        "intent": current_task.get("intent")
    }):

        # ===========     
           
        # Check cache for the exact query match
        # Sample prompts to test caching:
        # what is the eligibility criteria for maternity leave according to the our org leave policy?
        cached_result = get_cached_response(task_prompt, user_groups)
        
        # If exact match in cache is not found then check for semantic search match
        # If the query is semantically similar to a cached query, return the cached response
        # Sample prompts to test semantic caching:
        # what is the eligibility criteria for maternity leave according to our org leave policy?
        # who is eligible for maternity leave according to the our org leave policy?
        # Check cache for semantic search match
        if cached_result is not None:
            result = cached_result
        else:
            query_embedding = AzureEmbeddingWrapper().embed_query(task_prompt)
            semantic_cached_result = get_semantic_cached_response(task_prompt, query_embedding, user_groups)

            # If semantic search match in cache is not found then use AzureAI Search and generate response
            if semantic_cached_result is not None:
                result = semantic_cached_result
            else:
                try:
                    top5chunks = hybrid_search(task_prompt)
                    result = generate_response(task_prompt, top5chunks, user_groups)
                    # Cache the LLM output for future identical requests
                    set_cached_response(task_prompt, query_embedding, result, user_groups)
                except Exception:
                    pass

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