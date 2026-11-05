import os
import asyncio
import json
import sys
from pathlib import Path
import uuid
from langgraph.types import interrupt
from unittest.mock import AsyncMock, Mock, patch
from langchain_core.messages import AIMessage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Use a valid UUID for a dummy instrumentation key so azure monitor config doesn't fail
os.environ.setdefault("APPLICATIONINSIGHTS_CONNECTION_STRING", f"InstrumentationKey={uuid.uuid4()}")

from graphs import supervisor_graph
from nodes import planner
from agents import chat_agent, rag_agent
from helpers import rag_helper, vector_db
from agents import hybrid_rag_agent
from agents import sql_agent
from agents import excel_agent
from agents import ocr_agent
from agents import meeting_agent, image_agent, ppt_agent
from agents import azure_devops_agent
from nodes import classifier
from helpers import ppt_helper
from utils.async_runner import AsyncLoopRunner
from helpers import embeddings, hybrid_rag_helper
from helpers import cosmos_db_client
from tools import excel_tools


async def test_planner_async():
    plan = {
        "tasks": [
            {
                "id": "planner-task",
                "intent": "chat",
                "prompt": "Say hello",
            }
        ]
    }
    normalized_plan = {
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
        patch.object(
            planner.async_container,
            "create_item",
            new=AsyncMock(),
        ) as mock_create_item,
        patch.object(planner, "save_execution_plan") as mock_save_plan,
    ):
        mock_llm.ainvoke = AsyncMock(return_value=response)
        result = await planner.create_plan("test-thread", "Say hello", None)

    mock_llm.ainvoke.assert_awaited_once()
    mock_create_item.assert_awaited_once()
    mock_save_plan.assert_called_once_with("test-thread", normalized_plan)
    assert "Route explicit Azure DevOps or ADO data requests to azure_devops" in (
        mock_llm.ainvoke.await_args.args[0]
    )
    assert result == normalized_plan


async def test_planner_prompt_prioritizes_current_goal_over_attachment():
    plan = {
        "tasks": [
            {
                "id": "knowledge-task",
                "intent": "knowledge_search",
                "prompt": "Who is eligible for maternity leave according to our policy?",
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
        patch.object(
            planner.async_container,
            "create_item",
            new=AsyncMock(),
        ),
        patch.object(planner, "save_execution_plan"),
    ):
        mock_llm.ainvoke = AsyncMock(return_value=response)
        result = await planner.create_plan(
            "test-policy-thread",
            "Who is eligible for maternity leave according to our policy?",
            "data/uploads/unrelated.pdf",
        )

    planner_prompt = mock_llm.ainvoke.await_args.args[0]
    assert "uploaded file path only indicates that a file is available" in planner_prompt
    assert "choose knowledge_search even when an unrelated document is attached" in planner_prompt
    assert result["tasks"][0]["intent"] == "knowledge_search"


async def test_azure_devops_agent_uses_readonly_mcp_tools():
    class MCPTool:
        def __init__(self, name, input_schema):
            self.name = name
            self.description = name
            self.input_schema = input_schema

        def model_copy(self, update):
            return MCPTool(self.name, update["input_schema"])

    tools = [
        MCPTool(
            "core_list_projects",
            {"type": "object", "properties": {}},
        ),
        MCPTool(
            "wit_work_item",
            {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["get", "update"],
                    }
                },
                "required": ["action"],
            },
        ),
        MCPTool(
            "wit_work_item_write",
            {"type": "object", "properties": {}},
        ),
    ]
    tool_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "core_list_projects",
                "args": {},
                "id": "list-projects-call",
                "type": "tool_call",
            }
        ],
    )
    final_response = AIMessage(content="Found project Alpha.")

    class FakeMCPClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def list_tools(self):
            return tools

        async def call_tool(self, name, arguments):
            assert name == "core_list_projects"
            assert arguments == {}
            return type(
                "MCPResult",
                (),
                {"content": [type("Text", (), {"text": "Alpha"})()]},
            )()

    fake_client = FakeMCPClient()
    bound_model = type(
        "BoundModel",
        (),
        {
            "ainvoke": AsyncMock(
                side_effect=[tool_response, final_response]
            ),
        },
    )()
    fake_llm = type("FakeLLM", (), {})()
    fake_llm.bind_tools = Mock(return_value=bound_model)
    state = {
        "user_input": "List my Azure DevOps projects",
        "task_prompt": "List my Azure DevOps projects",
        "thread_id": "ado-thread",
        "current_task": {
            "id": "ado-task",
            "intent": "azure_devops",
            "dependencies": [],
        },
        "task_results": {},
    }

    with (
        patch.object(
            azure_devops_agent,
            "_get_client",
            return_value=("https://mcp.dev.azure.com/example", fake_client),
        ),
        patch.object(azure_devops_agent, "_client_locks", {}),
        patch.object(azure_devops_agent, "get_dependency_results", return_value={}),
        patch.object(azure_devops_agent, "llm", fake_llm),
    ):
        result = await azure_devops_agent.azure_devops_agent(state)

    assert bound_model.ainvoke.await_args_list[0].args[0][0].content.startswith(
        "You are an Azure DevOps read-only assistant."
    )
    fake_llm.bind_tools.assert_called_once()
    supplied_tools = fake_llm.bind_tools.call_args.args[0]
    supplied_by_name = {
        item["function"]["name"]: item["function"] for item in supplied_tools
    }
    assert set(supplied_by_name) == {"core_list_projects", "wit_work_item"}
    assert supplied_by_name["core_list_projects"]["parameters"] == (
        tools[0].input_schema
    )
    assert supplied_by_name["wit_work_item"]["parameters"]["properties"]["action"][
        "enum"
    ] == ["get"]
    assert bound_model.ainvoke.await_count == 2
    assert result["agent_result"] == ["Found project Alpha."]
    assert result["completed_steps"] == ["azure_devops"]
    assert result["task_results"]["ado-task"]["result"] == "Found project Alpha."


async def test_azure_devops_client_uses_local_server_and_azure_cli():
    with (
        patch.dict(os.environ, {"AZURE_DEVOPS_ORG": "example"}),
        patch.object(azure_devops_agent, "_clients", {}),
        patch.object(
            azure_devops_agent.shutil,
            "which",
            side_effect=lambda name: (
                r"C:\Node\npx.cmd" if name == "npx" else r"C:\AzureCLI\az.exe"
            ),
        ),
        patch.object(azure_devops_agent.sys, "platform", "win32"),
        patch.object(azure_devops_agent, "StdioTransport") as transport,
        patch.object(azure_devops_agent, "Client", return_value="client") as client,
    ):
        client_key, result = azure_devops_agent._get_client()

    assert client_key == "stdio:example"
    assert result == "client"
    assert transport.call_args.kwargs["command"].lower().endswith("cmd.exe")
    assert transport.call_args.kwargs["args"] == [
        "/d",
        "/c",
        "npx",
        "-y",
        "@azure-devops/mcp",
        "example",
        "--authentication",
        "azcli",
        "-d",
        "core",
        "work",
        "work-items",
        "repositories",
        "pipelines",
        "wiki",
        "search",
    ]
    client.assert_called_once()


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
        patch.object(chat_agent, "llm") as mock_llm,
        patch.object(
            chat_agent.async_container,
            "create_item",
            new=AsyncMock(),
        ) as mock_create_item,
    ):
        mock_llm.ainvoke = AsyncMock(return_value=response)
        result = await chat_agent.chat_agent(state)

    mock_llm.ainvoke.assert_awaited_once()
    mock_create_item.assert_awaited_once()
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
        patch.object(
            embeddings.async_container,
            "create_item",
            new=AsyncMock(),
        ) as mock_create_item,
    ):
        mock_embeddings.create = AsyncMock(return_value=response)
        result = await embeddings.AzureEmbeddingWrapper().aembed_query("query")

    mock_embeddings.create.assert_awaited_once()
    mock_create_item.assert_awaited_once()
    assert result == [0.1, 0.2]


async def test_cosmos_query_is_async():
    async def items():
        yield {"id": "item-1"}
        yield {"id": "item-2"}

    with patch.object(
        cosmos_db_client.async_container,
        "query_items",
        return_value=items(),
    ) as mock_query:
        result = await cosmos_db_client.query_items("SELECT * FROM c")

    mock_query.assert_called_once_with(
        query="SELECT * FROM c",
    )
    assert result == [{"id": "item-1"}, {"id": "item-2"}]


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
            hybrid_rag_helper.async_container,
            "create_item",
            new=AsyncMock(),
        ) as mock_create_item,
    ):
        mock_llm.ainvoke = AsyncMock(side_effect=[answer, rating])
        result = await hybrid_rag_helper.generate_response(
            "policy question",
            [{"title": "Policy", "chunk": "Policy content"}],
            "test-thread",
        )

    assert mock_llm.ainvoke.await_count == 2
    assert mock_create_item.await_count == 2
    assert result == "Policy answer [1]"


async def test_sql_agent_uses_async_executor():
    sql_database = object()
    executor = type("SQLExecutorStub", (), {})()
    executor.ainvoke = AsyncMock(return_value={"output": "sales answer"})

    async def run_sync_in_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    with (
        patch.object(sql_agent.SQLDatabase, "from_uri", return_value=sql_database) as mock_from_uri,
        patch.object(sql_agent, "create_sql_agent", return_value=executor) as mock_create_agent,
        patch.object(sql_agent.asyncio, "to_thread", new=AsyncMock(side_effect=run_sync_in_thread)),
    ):
        result = await sql_agent.ask_sql_llm("sales question", {}, [])

    assert result == "sales answer"
    mock_from_uri.assert_called_once_with(
        "sqlite:///sales_multi.db",
        include_tables=["customers", "orders", "products"],
    )
    mock_create_agent.assert_called_once()
    executor.ainvoke.assert_awaited_once_with({"input": "sales question"})


async def test_sql_graph_node_is_async():
    state = {
        "user_input": "sales question",
        "task_prompt": "sales question",
        "current_task": {
            "id": "sql-task",
            "intent": "database_search",
            "dependencies": [],
        },
        "task_results": {},
        "chat_history": [],
    }

    with patch.object(
        sql_agent,
        "ask_sql_llm",
        new=AsyncMock(return_value="sales answer"),
    ) as mock_ask:
        result = await sql_agent.sql_agent(state)

    mock_ask.assert_awaited_once_with("sales question", {}, [])
    assert result["agent_result"] == ["sales answer"]
    assert result["task_results"]["sql-task"]["result"] == "sales answer"


async def test_excel_agent_async():
    import pandas as pd

    state = {
        "user_input": "sum amount",
        "task_prompt": "sum amount",
        "uploaded_file": "report.xlsx",
        "thread_id": "excel-thread",
        "current_task": {
            "id": "excel-task",
            "intent": "excel_analysis",
            "dependencies": [],
        },
        "task_results": {},
    }
    agent_graph = type("ExcelAgentGraph", (), {})()
    agent_graph.ainvoke = AsyncMock(
        return_value={"messages": [AIMessage(content="Total: 42")]}
    )
    dataframe = pd.DataFrame({"amount": [20, 22]})

    async def run_sync_in_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    with (
        patch.object(excel_agent.Path, "is_file", return_value=True),
        patch.object(excel_agent, "read_excel_file", return_value=dataframe) as mock_read,
        patch.object(excel_agent, "create_excel_tools", return_value=[]) as mock_tools,
        patch.object(excel_agent, "create_agent", return_value=agent_graph) as mock_create_agent,
        patch.object(excel_agent.asyncio, "to_thread", new=AsyncMock(side_effect=run_sync_in_thread)),
    ):
        result = await excel_agent.excel_agent(state)

    mock_read.assert_called_once_with("report.xlsx")
    mock_tools.assert_called_once()
    mock_create_agent.assert_called_once()
    agent_graph.ainvoke.assert_awaited_once()
    assert result["agent_result"] == ["Total: 42"]
    assert result["task_results"]["excel-task"]["result"] == "Total: 42"


async def test_excel_tools_are_async():
    import pandas as pd

    dataframe = pd.DataFrame({"amount": [20, 22]})
    tools = {
        excel_tool.name: excel_tool
        for excel_tool in excel_tools.create_excel_tools(dataframe, {})
    }
    responses = [
        AIMessage(content="result = int(data_frame['amount'].sum())"),
        AIMessage(content="chart_path = 'chart.png'"),
    ]

    with patch.object(excel_tools, "llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(side_effect=responses)
        analysis = await tools["analyze_data"].ainvoke(
            {"question": "sum amount"}
        )
        chart = await tools["create_chart"].ainvoke(
            {"question": "plot amount"}
        )

    assert analysis == 42
    assert chart["status"] == "success"
    assert chart["chart_path"] == "chart.png"
    assert mock_llm.ainvoke.await_count == 2


async def test_ocr_uses_async_client_and_file_io():
    response = type(
        "OCRResponse",
        (),
        {
            "choices": [
                type(
                    "OCRChoice",
                    (),
                    {
                        "message": type(
                            "OCRMessage",
                            (),
                            {"content": "recognized text"},
                        )()
                    },
                )()
            ]
        },
    )()

    async def run_sync_in_thread(function, *args, **kwargs):
        assert function is ocr_agent._encode_image
        return "ZmFrZS1pbWFnZQ=="

    with (
        patch.object(ocr_agent.async_client.chat.completions, "create", new=AsyncMock(return_value=response)) as mock_create,
        patch.object(ocr_agent.asyncio, "to_thread", new=AsyncMock(side_effect=run_sync_in_thread)) as mock_to_thread,
    ):
        result = await ocr_agent.image_to_text("image.png", "Read the text")

    mock_to_thread.assert_awaited_once_with(ocr_agent._encode_image, "image.png")
    mock_create.assert_awaited_once()
    request = mock_create.await_args.kwargs
    assert request["messages"][0]["content"][0]["text"] == "Read the text"
    assert request["messages"][0]["content"][1]["image_url"].startswith(
        "data:image/png;base64,"
    )
    assert result == "recognized text"


async def test_ocr_agent_async():
    state = {
        "user_input": "read image",
        "task_prompt": "read image",
        "uploaded_file": "image.png",
        "current_task": {
            "id": "ocr-task",
            "intent": "image_to_text",
        },
    }

    with (
        patch.object(ocr_agent.Path, "is_file", return_value=True),
        patch.object(
            ocr_agent,
            "image_to_text",
            new=AsyncMock(return_value="recognized text"),
        ) as mock_image_to_text,
    ):
        result = await ocr_agent.ocr_agent(state)

    mock_image_to_text.assert_awaited_once_with("image.png", "read image")
    assert result["agent_result"] == ["recognized text"]
    assert result["task_results"]["ocr-task"]["result"] == "recognized text"


async def test_meeting_transcription_and_summary_async():
    transcription = type(
        "Transcription",
        (),
        {"text": "Meeting transcript"},
    )()
    summary = type(
        "MeetingSummary",
        (),
        {
            "choices": [
                type(
                    "MeetingChoice",
                    (),
                    {
                        "message": type(
                            "MeetingMessage",
                            (),
                            {"content": "Meeting summary"},
                        )()
                    },
                )()
            ]
        },
    )()

    with (
        patch.object(
            meeting_agent,
            "chunk_audio_file",
            new=AsyncMock(return_value=["meeting.mp3"]),
        ),
        patch.object(
            meeting_agent.async_client.audio.transcriptions,
            "create",
            new=AsyncMock(return_value=transcription),
        ) as mock_transcribe,
        patch.object(
            meeting_agent.async_client.chat.completions,
            "create",
            new=AsyncMock(return_value=summary),
        ) as mock_chat,
        patch.object(
            meeting_agent.asyncio,
            "to_thread",
            new=AsyncMock(return_value=b"audio-bytes"),
        ) as mock_to_thread,
    ):
        transcript = await meeting_agent.transcribe_audio("meeting.mp3")
        result = await meeting_agent.process_meeting(transcript)

    mock_to_thread.assert_awaited_once()
    mock_transcribe.assert_awaited_once_with(
        model=meeting_agent.WHISPER_MODEL,
        file=("meeting.mp3", b"audio-bytes"),
    )
    mock_chat.assert_awaited_once()
    assert transcript == "Meeting transcript"
    assert result == "Meeting summary"


async def test_meeting_agent_async():
    state = {
        "uploaded_file": "meeting.mp3",
        "current_task": {
            "id": "meeting-task",
            "intent": "meeting_assistant",
        },
    }

    with (
        patch.object(meeting_agent.Path, "is_file", return_value=True),
        patch.object(
            meeting_agent,
            "transcribe_audio",
            new=AsyncMock(return_value="transcript"),
        ) as mock_transcribe,
        patch.object(
            meeting_agent,
            "process_meeting",
            new=AsyncMock(return_value="summary"),
        ) as mock_process,
    ):
        result = await meeting_agent.meeting_agent(state)

    mock_transcribe.assert_awaited_once_with("meeting.mp3")
    mock_process.assert_awaited_once_with("transcript")
    assert result["agent_result"] == ["summary"]


async def test_image_generation_is_async():
    prompt_response = AIMessage(
        content="refined image prompt",
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
    image_result = type(
        "ImageResult",
        (),
        {
            "usage": type(
                "ImageUsage",
                (),
                {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            )(),
            "data": [
                type(
                    "ImageData",
                    (),
                    {"b64_json": "aW1hZ2UtYnl0ZXM="},
                )()
            ],
        },
    )()

    async def run_in_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    with (
        patch.object(image_agent, "llm") as mock_llm,
        patch.object(image_agent, "async_client") as mock_async_client,
        patch.object(
            image_agent.async_container,
            "create_item",
            new=AsyncMock(),
        ) as mock_create_item,
        patch.object(image_agent.asyncio, "to_thread", new=AsyncMock(side_effect=run_in_thread)),
        patch.object(image_agent, "_write_image") as mock_write,
    ):
        mock_llm.ainvoke = AsyncMock(return_value=prompt_response)
        mock_async_client.images.generate = AsyncMock(return_value=image_result)
        generated_prompt = await image_agent.generate_prompt(
            "image-thread", "Create a graphic", {}, []
        )
        image_path = await image_agent.generate_image(
            "image-thread", generated_prompt
        )

    mock_llm.ainvoke.assert_awaited_once()
    mock_async_client.images.generate.assert_awaited_once()
    assert mock_create_item.await_count == 2
    mock_write.assert_called_once()
    assert image_path.endswith(".png")


async def test_image_agent_async():
    state = {
        "user_input": "Create a graphic",
        "task_prompt": "Create a graphic",
        "current_task": {
            "id": "image-task",
            "intent": "image_generation",
            "dependencies": [],
        },
        "task_results": {},
        "chat_history": [],
        "thread_id": "image-thread",
    }

    with (
        patch.object(
            image_agent,
            "generate_prompt",
            new=AsyncMock(return_value="refined prompt"),
        ) as mock_prompt,
        patch.object(
            image_agent,
            "generate_image",
            new=AsyncMock(return_value="image.png"),
        ) as mock_generate,
    ):
        result = await image_agent.image_agent(state)

    mock_prompt.assert_awaited_once()
    mock_generate.assert_awaited_once_with("image-thread", "refined prompt")
    assert result["agent_result"] == ["image.png"]


async def test_ppt_agent_and_helper_async_boundary():
    slides = {"slides": [{"slide_type": "summary", "title": "Summary"}]}
    response = AIMessage(content=json.dumps(slides))

    async def run_in_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    state = {
        "user_input": "Create a presentation",
        "task_prompt": "Create a presentation",
        "current_task": {
            "id": "ppt-task",
            "intent": "ppt_creation",
            "dependencies": [],
        },
        "task_results": {},
        "chat_history": [],
    }

    with (
        patch.object(ppt_agent, "llm") as mock_llm,
        patch.object(ppt_agent, "create_presentation", return_value="deck.pptx") as mock_create,
        patch.object(ppt_agent.asyncio, "to_thread", new=AsyncMock(side_effect=run_in_thread)),
    ):
        mock_llm.ainvoke = AsyncMock(return_value=response)
        result = await ppt_agent.ppt_agent(state)

    mock_llm.ainvoke.assert_awaited_once()
    mock_create.assert_called_once_with(slides["slides"])
    assert result["agent_result"] == ["Here is your power point presentation: deck.pptx"]


def test_presentation_paths_are_unique_per_call():
    with (
        patch.object(ppt_helper, "Presentation") as mock_presentation,
        patch.object(ppt_helper, "uuid7", side_effect=["first-deck", "second-deck"]),
    ):
        first_path = ppt_helper.create_presentation([])
        second_path = ppt_helper.create_presentation([])

    assert first_path == "./data/uploads/first-deck.pptx"
    assert second_path == "./data/uploads/second-deck.pptx"
    assert mock_presentation.return_value.save.call_args_list[0].args == (
        first_path,
    )
    assert mock_presentation.return_value.save.call_args_list[1].args == (
        second_path,
    )


def test_async_loop_runner_reuses_loop():
    runner = AsyncLoopRunner()

    async def current_loop_id():
        return id(asyncio.get_running_loop())

    try:
        first_loop_id = runner.run(current_loop_id())
        second_loop_id = runner.run(current_loop_id())
        assert first_loop_id == second_loop_id
    finally:
        runner.close()


async def test_classifier_uses_async_model():
    state = {"user_input": "look up customer orders"}
    with patch.object(classifier, "llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="database_search"))
        result = await classifier.classify_intent(state)

    mock_llm.ainvoke.assert_awaited_once()
    assert result["intent"] == "database_search"


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
        planner,
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
            planner,
            "create_plan",
            new=AsyncMock(return_value=plan),
        ),
        patch.object(
            rag_agent,
            "rag_agent",
            new=AsyncMock(return_value=rag_result),
        ) as mock_rag_agent,
    ):
        service = supervisor_graph.WorkflowService(request)
        result = await service.invoke(request)

    mock_rag_agent.assert_awaited_once()
    assert result["workflow_status"] == "COMPLETED"
    assert result["data"]["messages"][-1].content == "RAG answer"


async def test_workflow_routes_azure_devops_to_agent():
    plan = {
        "tasks": [
            {
                "id": "ado-task",
                "intent": "azure_devops",
                "prompt": "List open work items",
            }
        ]
    }
    request = {
        "user_input": "List open work items",
        "uploaded_file": None,
        "chat_history": [],
        "thread_id": "test-thread-ado",
    }
    ado_result = {
        "agent_result": ["Found 2 open work items."],
        "completed_steps": ["azure_devops"],
        "task_results": {
            "ado-task": {
                "intent": "azure_devops",
                "result": "Found 2 open work items.",
            }
        },
    }

    with (
        patch.object(
            planner,
            "create_plan",
            new=AsyncMock(return_value=plan),
        ),
        patch.object(
            azure_devops_agent,
            "azure_devops_agent",
            new=AsyncMock(return_value=ado_result),
        ) as mock_ado_agent,
    ):
        service = supervisor_graph.WorkflowService(request)
        result = await service.invoke(request)

    mock_ado_agent.assert_awaited_once()
    assert result["workflow_status"] == "COMPLETED"
    assert result["data"]["messages"][-1].content == "Found 2 open work items."


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
    await test_planner_prompt_prioritizes_current_goal_over_attachment()
    await test_azure_devops_agent_uses_readonly_mcp_tools()
    await test_azure_devops_client_uses_local_server_and_azure_cli()
    await test_chat_agent_async()
    await test_rag_agent_async()
    await test_rag_uses_async_retrieval_and_model()
    await test_async_vector_retrieval()
    await test_rag_response_uses_async_model()
    await test_async_query_embedding()
    await test_cosmos_query_is_async()
    await test_async_hybrid_search()
    await test_hybrid_rag_agent_async()
    await test_hybrid_response_and_rating_are_async()
    await test_sql_agent_uses_async_executor()
    await test_sql_graph_node_is_async()
    await test_excel_agent_async()
    await test_excel_tools_are_async()
    await test_ocr_uses_async_client_and_file_io()
    await test_ocr_agent_async()
    await test_meeting_transcription_and_summary_async()
    await test_meeting_agent_async()
    await test_image_generation_is_async()
    await test_image_agent_async()
    await test_ppt_agent_and_helper_async_boundary()
    test_presentation_paths_are_unique_per_call()
    test_async_loop_runner_reuses_loop()
    await test_classifier_uses_async_model()
    chat_agent.chat_agent = fake_chat_agent
    await test_workflow_awaits_planner()
    await test_workflow_awaits_async_rag_node()
    await test_workflow_routes_azure_devops_to_agent()
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

    (ROOT / "data" / "plan" / f"{thread_id}.json").unlink()
    resume_service = supervisor_graph.WorkflowService(request)
    result = await resume_service.resume("continue", thread_id)
    assert result["workflow_status"] == "COMPLETED"
    assert result["data"]["messages"][-1].content == "async graph node ok"
    print("Async initialization, invocation, interrupt, and resume passed.")


if __name__ == "__main__":
    asyncio.run(main())
