import os
import asyncio
import json
import sys
from pathlib import Path
import uuid
from langgraph.types import interrupt
from unittest.mock import AsyncMock, patch
from langchain_core.messages import AIMessage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Use a valid UUID for a dummy instrumentation key so azure monitor config doesn't fail
os.environ.setdefault("APPLICATIONINSIGHTS_CONNECTION_STRING", f"InstrumentationKey={uuid.uuid4()}")

from graphs import supervisor_graph
from nodes import planner


async def test_planner_async():
    plan = {
        "tasks": [
            {
                "id": "planner-task",
                "intent": "chat",
                "prompt": "Say hello",
                "dependencies": [],
            }
        ]
    }
    response = AIMessage(
        content=json.dumps(plan),
        response_metadata={
            "token_usage": {
                "prompt_tokens": 4,
                "completion_tokens": 6,
                "total_tokens": 10,
            },
            "model_name": "test-model",
            "finish_reason": "stop",
        },
    )

    with (
        patch.object(planner, "llm") as mock_llm,
        patch.object(planner.container, "create_item") as mock_create_item,
        patch.object(planner, "save_execution_plan") as mock_save_plan,
    ):
        mock_llm.ainvoke = AsyncMock(return_value=response)
        result = await planner.create_plan("test-thread", "Say hello", None)

    mock_llm.ainvoke.assert_awaited_once()
    mock_create_item.assert_called_once()
    mock_save_plan.assert_called_once_with("test-thread", plan)
    assert result == plan


async def test_chat_agent_async():
    response = AIMessage(
        content="chat response",
        response_metadata={
            "token_usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
            },
            "model_name": "test-model",
            "finish_reason": "stop",
        },
    )
    state = {
        "user_input": "Say hello",
        "task_prompt": "Say hello",
        "current_task": {
            "id": "chat-task",
            "intent": "chat",
            "dependencies": [],
        },
        "thread_id": "test-thread-chat",
        "human_comment": "continue",
    }

    with (
        patch.object(supervisor_graph.chat_agent, "llm") as mock_llm,
        patch.object(supervisor_graph.chat_agent.container, "create_item") as mock_create_item,
    ):
        mock_llm.ainvoke = AsyncMock(return_value=response)
        result = await supervisor_graph.chat_agent.chat_agent(state)

    mock_llm.ainvoke.assert_awaited_once()
    mock_create_item.assert_called_once()
    assert result["agent_result"] == ["chat response"]
    assert result["task_results"]["chat-task"]["result"] == "chat response"


async def test_workflow_awaits_planner():
    plan = {
        "tasks": [
            {
                "id": "planner-task",
                "intent": "chat",
                "prompt": "Say hello",
                "dependencies": [],
            }
        ]
    }
    request = {
        "user_input": "Say hello",
        "uploaded_file": None,
        "chat_history": [],
        "thread_id": "test-thread-planner",
    }

    with patch.object(
        supervisor_graph,
        "create_plan",
        new=AsyncMock(return_value=plan),
    ) as mock_create_plan:
        service = supervisor_graph.WorkflowService(request)
        result = await service.invoke(request)

    mock_create_plan.assert_awaited_once_with(
        "test-thread-planner", "Say hello", None
    )
    assert result["workflow_status"] == "INTERRUPTED"


def fake_chat_agent(state):
    interrupt("Continue?")
    result = "async graph node ok"
    return {
        "agent_result": [result],
        "completed_steps": ["chat"],
        "task_results": {
            state["current_task"]["id"]: {
                "intent": "chat",
                "result": result,
            }
        },
    }


def write_plan(thread_id: str):
    plan = {
        "tasks": [
            {
                "id": "t1",
                "intent": "chat",
                "prompt": "say hello",
                "dependencies": []
            }
        ]
    }

    data_dir = ROOT / "data" / "plan"
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / f"{thread_id}.json"
    with file_path.open("w", encoding="utf-8") as f:
        json.dump({"thread_id": thread_id, "execution_plan": plan}, f, indent=2)


async def main():
    await test_planner_async()
    await test_chat_agent_async()
    supervisor_graph.chat_agent.chat_agent = fake_chat_agent
    await test_workflow_awaits_planner()
    thread_id = "test-thread-1"
    write_plan(thread_id)

    request = {
        "user_input": "test",
        "uploaded_file": None,
        "chat_history": [],
        "workflow_status": "INTERRUPTED",
        "thread_id": thread_id
    }
    print("Instantiating WorkflowService (should initialize tracer and build graph)")
    svc = supervisor_graph.WorkflowService(request)
    interrupted = await svc.invoke(request)
    assert interrupted["workflow_status"] == "INTERRUPTED"

    resume_service = supervisor_graph.WorkflowService(request)
    result = await resume_service.resume("continue", thread_id)
    assert result["workflow_status"] == "COMPLETED"
    assert result["data"]["messages"][-1].content == "async graph node ok"
    print("Async initialization, invocation, interrupt, and resume passed.")


if __name__ == "__main__":
    asyncio.run(main())
