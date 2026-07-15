from agent_state.agent_state import AgentState
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage

def ocr_agent(state: AgentState) -> AgentState:
    """Use this agent for OCR - Image to text conversion"""
    print("\n===== OCR AGENT =====")

    response = "Here is the text extracted from image"
        
    # Add Agent result to agent_result list in agentState
    agent_result_AIMessage = AIMessage(content=response)
    
    return {
        "agent_result": [response],
        "completed_steps": ["image_to_text"]
    }