from agent_state.agent_state import AgentState
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage

def sql_agent(state: AgentState) -> AgentState:
    """Use this agent for Database operations, database search, SQL generation"""
    print("\n===== SQL AGENT =====")

    response = "The average revenue according to the database records is $10287624"
    
    # Add Agent result to agent_result list in agentState
    agent_result_AIMessage = AIMessage(content=response)
    
    return {
        "agent_result": [response],
        "completed_steps": ["database_search"]
    }
