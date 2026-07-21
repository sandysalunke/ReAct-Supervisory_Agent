from agent_state.agent_state import AgentState
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from helpers.file_operations import read_excel_file
from tools.excel_tools import create_excel_tools
from config.azure_config import llm
from pathlib import Path

# excel_agent - Added as node in supervisor agent graph
def excel_agent(state: AgentState) -> AgentState:
    """Use this agent for excel analytics"""
    print("\n===== EXCEL AGENT =====")

    user_input = state["user_input"]
    file_path = state["uploaded_file"]
    file = Path(file_path)
    
    if file.is_file():
        data_frame = read_excel_file(file_path)
        tools = create_excel_tools(data_frame)

        system_prompt = f"""
            You are an Excel analytics assistant.

            Available columns:
            {list(data_frame.columns)}

            Use:
            - get_schema for structure questions
            - analyze_data for calculations
            - create_chart for visualizations

            Always use a tool before answering.
            """

        agent = create_agent(
            llm,
            tools=tools,
            context_schema=data_frame,
            system_prompt=system_prompt,
        )

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            },
            config={
                "configurable": {
                    "thread_id": str(uuid7())
                }
            }
        )

        response = result["messages"][-1].content
    else:
        response = "File does not exist"
        
    return {
        "agent_result": [response],
        "completed_steps": ["excel_analysis"]
    }