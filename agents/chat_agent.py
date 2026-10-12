import json
from config.azure_config import llm
from utils.tracing import tracer
from opentelemetry import trace
from agent_state.agent_state import AgentState
from langgraph.types import interrupt
from helpers.dependency_manager import get_dependency_results
import uuid
from datetime import datetime, UTC
from helpers.cosmos_db_client import container

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

    with tracer.start_as_current_span("agent.chat", attributes={"llm_used": True}):
        response = llm.invoke(prompt)

        # Add traces in AI for observability and monitoring
        usage = response.response_metadata["token_usage"]
        trace.get_current_span().set_attribute("llm.model", response.response_metadata["model_name"])
        trace.get_current_span().set_attribute("llm.prompt_tokens", usage["prompt_tokens"])
        trace.get_current_span().set_attribute("llm.completion_tokens", usage["completion_tokens"])
        trace.get_current_span().set_attribute("llm.total_tokens", usage["total_tokens"])
        trace.get_current_span().set_attribute("llm.finish_reason", response.response_metadata["finish_reason"])

        # Add traces in Cosmos DB for cost observability and monitoring
        container.create_item({
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "agent": "chat_agent",
            "thread_id": state.get("thread_id"),
            "model": response.response_metadata["model_name"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "finish_reason": response.response_metadata["finish_reason"],
        })

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
    