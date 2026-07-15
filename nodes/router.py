from agent_state.agent_state import AgentState
from langgraph.graph import END

def route_agent(state: AgentState) -> str:

    intent = state["intent"].strip()

    mapping = {
        "chat": "chat_agent",
        "image_generation": "image_agent",
        "file_reasoning": "rag_agent",
        "database_search": "sql_agent",
        "image_to_text": "ocr_agent",
        "meeting_assistant": "meeting_agent",
        "excel_analysis": "excel_agent",
        "END": END,
    }

    return mapping[intent]