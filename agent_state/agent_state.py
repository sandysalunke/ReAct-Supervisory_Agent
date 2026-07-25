from typing import TypedDict, Optional, Annotated, Sequence, Any
from operator import add

def merge_dicts(left: dict, right: dict) -> dict:
    return {**left, **right}

class AgentState(TypedDict):
    user_input: str
    uploaded_file: Optional[str]
    execution_plan: dict
    completed_steps: Annotated[list, add]
    agent_result: Annotated[list, add]
    messages: Annotated[list, add]
    chat_history: list
    task_results: Annotated[dict, merge_dicts]