import json
from config.azure_config import llm
from agent_state.agent_state import AgentState
from langgraph.types import interrupt
from helpers.dependency_manager import get_dependency_results

# chat_agent - Added as node in supervisor agent graph
def chat_agent(state: AgentState):
    """Use this agent for General conversational AI tasks"""
    print("\n===== CHAT AGENT =====")

    user_input = state.get("user_input")
    task_prompt = state.get("task_prompt", user_input)
    current_task = state.get("current_task", {})
    task_results = state.get("task_results",{})
    chat_history = state.get("chat_history",[])

    dependency_results = get_dependency_results(current_task, task_results)

    prompt = f"""
        User Request:
        {task_prompt}
        
        Results from previous steps:
        {json.dumps(dependency_results, indent=2)}

        Chat history:
        {chat_history}
        
        Generate the next response considering all previous results and chat history.
    """

    response = llm.invoke(prompt)

    if state.get("human_comment"):
        comment = state["human_comment"]
    else:
        comment = interrupt("Should I continue?")
        
    return {
        "human_comment": comment,
        "agent_result": [response.content],
        "completed_steps": ["chat"],
        "task_results": {
            current_task["id"]: {
                "intent": current_task["intent"],
                "result": response.content
            }
        }
    }
    