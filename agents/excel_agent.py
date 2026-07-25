from agent_state.agent_state import AgentState
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from helpers.file_operations import read_excel_file
from tools.excel_tools import create_excel_tools
from config.azure_config import llm
from pathlib import Path
from helpers.dependency_manager import get_dependency_results

# excel_agent - Added as node in supervisor agent graph
def excel_agent(state: AgentState) -> AgentState:
    """Use this agent for excel analytics"""
    print("\n===== EXCEL AGENT =====")

    user_input = state.get("user_input")
    task_prompt = state.get("task_prompt", user_input)
    current_task = state.get("current_task", {})
    task_results = state.get("task_results",{})

    file_path = state.get("uploaded_file", "")
    file_extension = Path(file_path).suffix
    file_extensions = {".xls", ".xlsx", ".xlm", ".xlsb"}
    file = Path(file_path)

    if file.is_file() and file_extension in file_extensions:
        dependency_results = get_dependency_results(current_task, task_results)
        data_frame = read_excel_file(file_path)
        tools = create_excel_tools(data_frame, dependency_results)

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
                        "content": task_prompt,
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
        "completed_steps": ["excel_analysis"],
        "task_results": {
            current_task["id"] : {
                "intent": current_task["intent"],
                "result": response
            }
        }
    }