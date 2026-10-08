import os
from dotenv import load_dotenv
from agent_state.agent_state import AgentState
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from helpers.hybrid_rag_helper import generate_response

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

    return list(results)

# rag_agent - Added as node in supervisor agent graph
def hybrid_rag_agent(state: AgentState) -> AgentState:
    """Use this agent for RAG - reasoning based on file content"""
    print("\n===== AZURE AI SEARCH AGENT =====")

    user_input = state.get("user_input")
    task_prompt = state.get("task_prompt", user_input)
    current_task = state.get("current_task", {})

    top5chunks = hybrid_search(task_prompt)

    # what is the eligibility criteria for maternity leave according to the our org leave policy?
    # what criteria makes employee to be eligible for maternity leave according to the our org leave policy?
    result = generate_response(task_prompt, top5chunks, user_groups)

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