from config.azure_config import llm
from agent_state.agent_state import AgentState
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage

# summary_agent - Added as node in supervisor agent graph
def summary_agent(state: AgentState) -> AgentState:
    """Use this agent to summarize the final result of original goal"""
    print("\n===== SUMMARY AGENT =====")

    human_message = HumanMessage(content=state["user_input"])
    ai_response_message = AIMessage(state["agent_result"][-1])

    return {
        "messages": [human_message, ai_response_message],
        "completed_steps": ["summarize_result"]
    }
    