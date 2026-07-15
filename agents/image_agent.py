from agent_state.agent_state import AgentState
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage

def image_agent(state: AgentState) -> AgentState:
    """Use this agent for Image generation"""
    print("\n===== IMAGE GENERATION AGENT =====")

    response = "Here is your generated image"
    
    # Add Agent result to agent_result list in agentState
    agent_result_AIMessage = AIMessage(content=response)
    
    return {
        "agent_result": [response],
        "completed_steps": ["image_generation"]
    }