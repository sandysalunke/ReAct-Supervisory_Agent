from config.azure_config import llm
from agent_state.agent_state import AgentState
from helpers.ppt_helper import create_presentation
from helpers.dependency_manager import get_dependency_results
import json

# Sample ppt content
slides_data_structure = {
    "slides": [
        {
            "slide_type": "bullet/summary/table/chart/image",
            "title": "",
            "subtitle": "", # Optional
            "description": "",  # Optional
            "bullets": [],  # Optional
            "table": {  # Optional
                "columns": [],
                "rows": []
            },
            "images": ["file_path"],    # Optional
            "charts": ["file_path"],    # Optional
        }
    ]
}

# chat_agent - Added as node in supervisor agent graph
def ppt_agent(state: AgentState):
    """Use this agent to create power point (ppt) presentation"""
    print("\n===== PPT AGENT =====")

    user_input = state.get("user_input")
    task_prompt = state.get("task_prompt", user_input)
    current_task = state.get("current_task", {})
    task_results = state.get("task_results",{})
    chat_history = state.get("chat_history",[])

    dependency_results = get_dependency_results(current_task, task_results)

    prompt = f"""
        Create content for power point presentation.
        {task_prompt}
        
        Results from previous steps:
        {dependency_results}

        Chat history:
        {chat_history}

        Rules:
        - No markdown (no ``` blocks)
        - Use results from previous steps
        - Use images and charts from previous results
        - Use chat history to decide content, format, structue
        - return data in this format: {slides_data_structure}
        - Use double quotes for all properties and values
        
    """

    response = llm.invoke(prompt)

    if response.content:
        slides = json.loads(response.content)
        file_name = create_presentation(slides['slides'])

    return {
        "agent_result": [f"Here is your power point presentation: {file_name}" ],
        "completed_steps": ["chat"],
        "task_results": {
            current_task["id"] : {
                "intent": current_task["intent"],
                "result": response.content
            }
        }
    }
    