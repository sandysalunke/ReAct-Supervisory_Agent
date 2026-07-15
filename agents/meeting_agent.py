from agent_state.agent_state import AgentState
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage

def meeting_agent(state: AgentState) -> AgentState:
    """Use this agent for meeting assistance"""
    print("\n===== MEETING AGENT =====")

    response = "Here are the meeting insights"

    # Add Agent result to agent_result list in agentState
    agent_result_AIMessage = AIMessage(content=response)
    
    return {
        "agent_result": [response],
        "completed_steps": ["meeting_assistant"]
    }