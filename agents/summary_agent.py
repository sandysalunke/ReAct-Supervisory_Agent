import json

from agent_state.agent_state import AgentState
from langchain_core.messages import AIMessage, HumanMessage

INTENT_DISPLAY_NAMES = {
    "chat": "Response",
    "image_generation": "Generated image",
    "file_reasoning": "Document analysis",
    "knowledge_search": "Organization knowledge",
    "database_search": "Database analysis",
    "image_to_text": "Image text extraction",
    "meeting_assistant": "Meeting analysis",
    "excel_analysis": "Spreadsheet analysis",
    "ppt_creation": "Presentation",
    "azure_devops": "Azure DevOps",
}

# summary_agent - Added as node in supervisor agent graph
def summary_agent(state: AgentState) -> AgentState:
    """Use this agent to summarize the final result of original goal"""
    print("\n===== SUMMARY AGENT =====")

    tasks = state["execution_plan"]["tasks"]
    task_results = state.get("task_results", {})
    formatted_results = []

    for task in tasks:
        task_id = task["id"]
        task_result = task_results.get(task_id)
        if not isinstance(task_result, dict) or "result" not in task_result:
            raise ValueError(
                f"Workflow task {task_id!r} completed without a recorded result."
            )

        result = task_result["result"]
        result_text = (
            result if isinstance(result, str) else json.dumps(result, default=str)
        )
        display_name = INTENT_DISPLAY_NAMES.get(task["intent"])
        if display_name is None:
            raise ValueError(
                "A user-facing title is not configured for a workflow task."
            )
        formatted_results.append((display_name, result_text))

    if len(formatted_results) == 1:
        final_result = formatted_results[0][1]
    else:
        title_counts = {
            title: sum(1 for candidate, _ in formatted_results if candidate == title)
            for title, _ in formatted_results
        }
        title_occurrences = {}
        sections = []
        for title, result in formatted_results:
            title_occurrences[title] = title_occurrences.get(title, 0) + 1
            if title_counts[title] > 1:
                title = f"{title} ({title_occurrences[title]})"
            sections.append(f"### {title}\n\n{result}")
        final_result = "\n\n".join(sections)

    human_message = HumanMessage(content=state["user_input"])
    ai_response_message = AIMessage(final_result)

    return {
        "messages": [human_message, ai_response_message],
        "completed_steps": ["summarize_result"],
        "thread_id": state["thread_id"],
        "workflow_status": "COMPLETED",
    }
    