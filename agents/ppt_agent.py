from config.azure_config import llm
from agent_state.agent_state import AgentState
from helpers.ppt_helper import create_presentation
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

    prompt = f"""
        Create content for power point presentation.
        {state["user_input"]}
        
        Results from previous steps:
        {state.get("agent_result",[])}

        Chat history:
        {state.get("chat_history",[])}

        Rules:
        - No markdown (no ``` blocks)
        - Use results from previous steps
        - Use images and charts from previous results
        - Use chat history to decide content, format, structue
        - return data inthis format: {slides_data_structure}
        - Use double quotes for all properties and values
        
    """

    response = llm.invoke(prompt)

    slides = json.loads(response.content)

    if response.content:
        file_name = create_presentation(slides['slides'])

    return {
        "agent_result": [f"Here is your power point presentation: {file_name}" ],
        "completed_steps": ["chat"]
    }
    