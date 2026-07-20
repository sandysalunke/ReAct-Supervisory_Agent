from typing import TypedDict, Optional, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from operator import add

class AgentState(TypedDict):
    user_input: str
    uploaded_file: Optional[str]
    execution_plan: dict
    completed_steps: Annotated[list, add]
    agent_result: Annotated[list, add]
    messages: Annotated[list, add]
    chat_history: list
    