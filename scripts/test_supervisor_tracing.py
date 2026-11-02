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
from agents import rag_agent
from helpers import rag_helper, vector_db
from agents import hybrid_rag_agent
from helpers import embeddings, hybrid_rag_helper


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


async def test_rag_agent_async():
    state = {
        "user_input": "Summarize this document",
        "task_prompt": "Summarize this document",
        "uploaded_file": "document.pdf",
        "current_task": {
            "id": "rag-task",
            "intent": "file_reasoning",
            "dependencies": [],
        },
    }

    with (
        patch.object(rag_agent, "get_file_hash", return_value="file-hash"),
        patch.object(rag_agent, "collection_exists", return_value=True),
        patch.object(
            rag_agent,
            "answer_question",
            new=AsyncMock(return_value="document answer"),
        ) as mock_answer,
    ):
        result = await rag_agent.rag_agent(state)

    mock_answer.assert_awaited_once_with("file-hash", "Summarize this document")
    assert result["agent_result"] == ["document answer"]
    assert result["completed_steps"] == ["file_reasoning"]
    assert result["task_results"]["rag-task"]["result"] == "document answer"


async def test_rag_uses_async_retrieval_and_model():
    document = object()
    vector_store = object()
    with (
        patch.object(
            rag_agent,
            "load_vector_store",
            return_value=vector_store,
        ),
        patch.object(
            rag_agent,
            "retrieve_documents_async",
            new=AsyncMock(return_value=[document]),
        ) as mock_retrieve,
        patch.object(
            rag_agent,
            "generate_response",
            new=AsyncMock(return_value="answer"),
        ) as mock_generate,
    ):
        result = await rag_agent.answer_question("file-hash", "question")

    mock_retrieve.assert_awaited_once_with("question", vector_store)
    mock_generate.assert_awaited_once_with("question", [document])
    assert result == "answer"


async def test_async_vector_retrieval():
    vector_store = type("VectorStore", (), {})()
    vector_store.asimilarity_search = AsyncMock(return_value=["chunk"])

    result = await vector_db.retrieve_documents_async("question", vector_store)

    vector_store.asimilarity_search.assert_awaited_once_with(
        query="question", k=5
    )
    assert result == ["chunk"]


async def test_rag_response_uses_async_model():
    response = AIMessage(content="grounded answer")
    with patch.object(rag_helper, "llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=response)
        result = await rag_helper.generate_response("question", [])

    mock_llm.ainvoke.assert_awaited_once()
    assert result == "grounded answer"


async def test_async_query_embedding():
    response = type(
        "EmbeddingResponse",
        (),
        {
            "model": "embedding-model",
            "usage": type(
                "EmbeddingUsage",
                (),
                {"prompt_tokens": 2, "total_tokens": 2},
            )(),
            "data": [type("EmbeddingData", (), {"embedding": [0.1, 0.2]})()],
        },
    )()

    with (
        patch.object(embeddings.async_client, "embeddings") as mock_embeddings,
        patch.object(embeddings.container, "create_item") as mock_create_item,
    ):
        mock_embeddings.create = AsyncMock(return_value=response)
        result = await embeddings.AzureEmbeddingWrapper().aembed_query("query")

    mock_embeddings.create.assert_awaited_once()
    mock_create_item.assert_called_once()
    assert result == [0.1, 0.2]


async def test_async_hybrid_search():
    mock_client = type("SearchClientStub", (), {})()
    mock_client.search = AsyncMock()

    async def iter_results():
        yield {"title": "Policy", "chunk": "Policy content"}

    mock_client.search.return_value = iter_results()
    class AsyncSearchClientContext:
        async def __aenter__(self):
            return mock_client

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    mock_search_client = AsyncSearchClientContext()

    with patch.object(
        hybrid_rag_agent,
        "SearchClient",
        return_value=mock_search_client,
    ) as search_client_factory:
        results = await hybrid_rag_agent.hybrid_search("policy question")

    search_client_factory.assert_called_once()
    mock_client.search.assert_awaited_once()
    assert results == [{"title": "Policy", "chunk": "Policy content"}]


async def test_hybrid_rag_agent_async():
    state = {
        "user_input": "policy question",
        "task_prompt": "policy question",
        "current_task": {
            "id": "knowledge-task",
            "intent": "knowledge_search",
            "dependencies": [],
        },
        "thread_id": "test-hybrid-thread",
    }

    class FakeEmbeddingWrapper:
        async def aembed_query(self, query):
            return [0.1, 0.2]

    async def run_sync_in_thread(function, *args):
        return function(*args)

    with (
        patch.object(hybrid_rag_agent, "get_cached_response", return_value=None),
        patch.object(
            hybrid_rag_agent,
            "get_semantic_cached_response",
            return_value=None,
        ),
        patch.object(
            hybrid_rag_agent,
            "AzureEmbeddingWrapper",
            return_value=FakeEmbeddingWrapper(),
        ),
        patch.object(
            hybrid_rag_agent,
            "hybrid_search",
            new=AsyncMock(return_value=[{"chunk": "policy"}]),
        ) as mock_search,
        patch.object(
            hybrid_rag_agent,
            "generate_response",
            new=AsyncMock(return_value="policy answer"),
        ) as mock_generate,
        patch.object(
            hybrid_rag_agent.asyncio,
            "to_thread",
            new=AsyncMock(side_effect=run_sync_in_thread),
        ),
    ):
        result = await hybrid_rag_agent.hybrid_rag_agent(state)

    mock_search.assert_awaited_once_with("policy question")
    mock_generate.assert_awaited_once_with(
        "policy question",
        [{"chunk": "policy"}],
        "test-hybrid-thread",
    )
    assert result["agent_result"] == ["policy answer"]


async def test_hybrid_response_and_rating_are_async():
    answer = AIMessage(
        content="Policy answer [1]",
        response_metadata={
            "token_usage": {
                "prompt_tokens": 5,
                "completion_tokens": 4,
                "total_tokens": 9,
            },
            "model_name": "test-model",
            "finish_reason": "stop",
        },
    )
    rating = AIMessage(
        content='{"groundedness": 5, "relevance": 5, "completeness": 4}',
        response_metadata={
            "token_usage": {
                "prompt_tokens": 6,
                "completion_tokens": 3,
                "total_tokens": 9,
            },
            "model_name": "test-model",
            "finish_reason": "stop",
        },
    )

    with (
        patch.object(hybrid_rag_helper, "llm") as mock_llm,
        patch.object(
            hybrid_rag_helper.container,
            "create_item",
        ) as mock_create_item,
    ):
        mock_llm.ainvoke = AsyncMock(side_effect=[answer, rating])
        result = await hybrid_rag_helper.generate_response(
            "policy question",
            [{"title": "Policy", "chunk": "Policy content"}],
            "test-thread",
        )

    assert mock_llm.ainvoke.await_count == 2
    assert mock_create_item.call_count == 2
    assert result == "Policy answer [1]"


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


async def test_workflow_awaits_async_rag_node():
    plan = {
        "tasks": [
            {
                "id": "rag-task",
                "intent": "file_reasoning",
                "prompt": "Summarize this document",
                "dependencies": [],
            }
        ]
    }
    request = {
        "user_input": "Summarize this document",
        "uploaded_file": "document.pdf",
        "chat_history": [],
        "thread_id": "test-thread-rag",
    }
    rag_result = {
        "agent_result": ["RAG answer"],
        "completed_steps": ["file_reasoning"],
        "task_results": {
            "rag-task": {
                "intent": "file_reasoning",
                "result": "RAG answer",
            }
        },
    }

    with (
        patch.object(
            supervisor_graph,
            "create_plan",
            new=AsyncMock(return_value=plan),
        ),
        patch.object(
            supervisor_graph.rag_agent,
            "rag_agent",
            new=AsyncMock(return_value=rag_result),
        ) as mock_rag_agent,
    ):
        service = supervisor_graph.WorkflowService(request)
        result = await service.invoke(request)

    mock_rag_agent.assert_awaited_once()
    assert result["workflow_status"] == "COMPLETED"
    assert result["data"]["messages"][-1].content == "RAG answer"


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
    await test_rag_agent_async()
    await test_rag_uses_async_retrieval_and_model()
    await test_async_vector_retrieval()
    await test_rag_response_uses_async_model()
    await test_async_query_embedding()
    await test_async_hybrid_search()
    await test_hybrid_rag_agent_async()
    await test_hybrid_response_and_rating_are_async()
    supervisor_graph.chat_agent.chat_agent = fake_chat_agent
    await test_workflow_awaits_planner()
    await test_workflow_awaits_async_rag_node()
    thread_id = "test-thread-1"
    write_plan(thread_id)

    request = {
        "user_input": "test",
        "uploaded_file": None,
        "chat_history": [],
        "workflow_status": "INTERRUPTED",
        "thread_id": thread_id
    }
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
