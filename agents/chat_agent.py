from config.azure_config import llm
from agent_state.agent_state import AgentState
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage

# chat_agent - Added as node in supervisor agent graph
def chat_agent(state: AgentState):
    """Use this agent for General conversational AI tasks"""
    print("\n===== CHAT AGENT =====")

    prompt = f"""
        User Request:
        {state["user_input"]}
        
        Results from previous steps:
        {state.get("agent_result",[])}
        
        Generate the next response considering all previous results.
    """
    
    response = llm.invoke(prompt)

    return {
        "agent_result": [response.content],
        "completed_steps": ["chat"]
    }
    