from agent_state.agent_state import AgentState
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage

# excel_agent - Added as node in supervisor agent graph
def excel_agent(state: AgentState) -> AgentState:
    """Use this agent for excel analytics"""
    print("\n===== EXCEL AGENT =====")

    response = "The average revenue according to the excel is $10287624"

    # Add Agent result to agent_result list in agentState
    agent_result_AIMessage = AIMessage(content=response)
    
    return {
        "agent_result": [response],
        "completed_steps": ["excel_analysis"]
    }