"""API tests with the workflow and Cosmos DB mocked. Run: python scripts\\test_api.py"""
import json
import os
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Use a valid UUID for a dummy instrumentation key so azure monitor config doesn't fail
os.environ.setdefault(
    "APPLICATIONINSIGHTS_CONNECTION_STRING", f"InstrumentationKey={uuid.uuid4()}"
)

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from api import chat, dashboards
from api.app import app
from api.paths import DATA_DIR
from graphs import supervisor_graph
from helpers import file_operations

client = TestClient(app)


class FakeWorkflowService:
    """Interrupts when asked to, otherwise echoes the prompt."""

    requests: list[dict] = []

    def __init__(self, request):
        self.request = request
        FakeWorkflowService.requests.append(request)

    async def invoke(self, request):
        prompt = request["user_input"]
        if prompt == "interrupt me":
            return {
                "thread_id": "thread-1",
                "workflow_status": "INTERRUPTED",
                "data": {
                    "messages": [
                        HumanMessage(content=prompt),
                        SystemMessage(
                            content="Should I continue?",
                            additional_kwargs={
                                "thread_id": "thread-1",
                                "workflow_status": "INTERRUPTED",
                                "interrupt_ids": ["interrupt-1"],
                            },
                        ),
                    ]
                },
            }
        if prompt == "fail":
            raise RuntimeError("planner exploded")
        return self._completed(prompt, f"echo: {prompt}")

    async def resume(self, user_response, thread_id):
        return self._completed(user_response, f"resumed {thread_id}")

    @staticmethod
    def _completed(prompt, answer):
        return {
            "thread_id": "thread-1",
            "workflow_status": "COMPLETED",
            "data": {
                "messages": [HumanMessage(content=prompt), AIMessage(content=answer)]
            },
        }


def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


def test_chat_keeps_history_and_resumes_interrupts():
    FakeWorkflowService.requests = []
    with patch.object(supervisor_graph, "WorkflowService", FakeWorkflowService):
        first = client.post("/api/chat", data={"message": "hello"})
        assert first.status_code == 200, first.text
        body = first.json()
        session_id = body["session_id"]
        assert body["workflow_status"] == "COMPLETED"
        assert [message["role"] for message in body["messages"]] == ["user", "assistant"]
        assert body["messages"][1]["segments"] == [{"type": "text", "text": "echo: hello"}]

        interrupted = client.post(
            "/api/chat", data={"message": "interrupt me", "session_id": session_id}
        ).json()
        assert interrupted["workflow_status"] == "INTERRUPTED"
        assert interrupted["messages"][1]["role"] == "system"
        assert interrupted["messages"][1]["interrupted"] is True
        # The second request saw the first turn as chat history.
        assert len(FakeWorkflowService.requests[1]["chat_history"]) == 2

        resumed = client.post(
            "/api/chat", data={"message": "yes", "session_id": session_id}
        ).json()
        resume_request = FakeWorkflowService.requests[2]
        assert resume_request["workflow_status"] == "INTERRUPTED"
        assert resume_request["thread_id"] == "thread-1"
        assert resume_request["interrupt_ids"] == ["interrupt-1"]
        assert resumed["messages"][1]["segments"][0]["text"] == "resumed thread-1"

    history = client.get(f"/api/chat/{session_id}").json()
    assert len(history["messages"]) == 6

    assert client.delete(f"/api/chat/{session_id}").status_code == 204
    assert client.get(f"/api/chat/{session_id}").status_code == 404


def _parse_sent_at(message: dict) -> datetime:
    """The message's sent_at as a datetime; fails unless it is ISO-8601 in UTC."""
    assert isinstance(message.get("sent_at"), str), message
    parsed = datetime.fromisoformat(message["sent_at"])
    assert parsed.utcoffset() == timedelta(0), message["sent_at"]
    return parsed


def test_chat_messages_carry_sent_at():
    FakeWorkflowService.requests = []
    started = datetime.now(UTC)
    with patch.object(supervisor_graph, "WorkflowService", FakeWorkflowService):
        first = client.post("/api/chat", data={"message": "hello"}).json()
        session_id = first["session_id"]
        interrupted = client.post(
            "/api/chat", data={"message": "interrupt me", "session_id": session_id}
        ).json()
        resumed = client.post(
            "/api/chat", data={"message": "yes", "session_id": session_id}
        ).json()
    finished = datetime.now(UTC)

    posted = first["messages"] + interrupted["messages"] + resumed["messages"]
    assert [message["role"] for message in posted] == [
        "user",
        "assistant",
        "user",
        "system",
        "user",
        "assistant",
    ]
    assert posted[3]["interrupted"] is True

    # Every message, whatever its role, has a UTC time, in conversation order.
    times = [_parse_sent_at(message) for message in posted]
    assert times == sorted(times)
    assert started <= times[0] and times[-1] <= finished

    # Loading the session returns the same times as the sends did.
    history = client.get(f"/api/chat/{session_id}").json()["messages"]
    assert [message["role"] for message in history] == [
        message["role"] for message in posted
    ]
    assert [message["sent_at"] for message in history] == [
        message["sent_at"] for message in posted
    ]

    assert client.delete(f"/api/chat/{session_id}").status_code == 204


def test_messages_without_a_recorded_time_omit_sent_at():
    for message in (
        HumanMessage(content="hi"),
        AIMessage(content="hello"),
        SystemMessage(content="note"),
    ):
        assert "sent_at" not in chat.serialize_message(message)
        assert "sent_at" not in chat.serialize_message(message, None)

    recorded = "2026-05-04T09:15:30.123456+00:00"
    assert chat.serialize_message(HumanMessage(content="hi"), recorded)["sent_at"] == recorded

    # A session whose sent_at list has a gap and is shorter than its history.
    session_id = f"untimed-{uuid.uuid4().hex}"
    chat._sessions[session_id] = chat.ChatSession(
        chat_history=[
            HumanMessage(content="first"),
            AIMessage(content="second"),
            HumanMessage(content="third"),
            AIMessage(content="fourth"),
        ],
        sent_at=[recorded, None],
    )
    try:
        history = client.get(f"/api/chat/{session_id}").json()["messages"]
        assert len(history) == 4
        assert history[0]["sent_at"] == recorded
        for untimed in history[1:]:
            assert "sent_at" not in untimed

        # A new turn is timed without shifting a time onto the untimed messages.
        with patch.object(supervisor_graph, "WorkflowService", FakeWorkflowService):
            sent = client.post(
                "/api/chat", data={"message": "hello", "session_id": session_id}
            ).json()
        assert sent["session_id"] == session_id
        for timed in sent["messages"]:
            _parse_sent_at(timed)

        history = client.get(f"/api/chat/{session_id}").json()["messages"]
        assert len(history) == 6
        assert history[0]["sent_at"] == recorded
        for untimed in history[1:4]:
            assert "sent_at" not in untimed
        assert [message["sent_at"] for message in history[4:]] == [
            message["sent_at"] for message in sent["messages"]
        ]
    finally:
        chat._sessions.pop(session_id, None)


def test_chat_upload_is_saved_and_remembered():
    FakeWorkflowService.requests = []
    with (
        patch.object(supervisor_graph, "WorkflowService", FakeWorkflowService),
        patch.object(
            file_operations, "save_uploaded_file", return_value="./data/uploads/x.csv"
        ) as mock_save,
    ):
        first = client.post(
            "/api/chat",
            data={"message": "analyze"},
            files={"file": ("sales.csv", b"a,b\n1,2\n", "text/csv")},
        ).json()
        assert mock_save.call_args.args[0] == "sales.csv"
        assert first["uploaded_file_name"] == "sales.csv"
        # A follow-up without a file still references the earlier upload.
        client.post(
            "/api/chat", data={"message": "again", "session_id": first["session_id"]}
        )
    assert [request["uploaded_file"] for request in FakeWorkflowService.requests] == [
        "./data/uploads/x.csv",
        "./data/uploads/x.csv",
    ]


def test_save_uploaded_file_copies_stream():
    import io

    saved_path = Path(file_operations.save_uploaded_file("notes.txt", io.BytesIO(b"hi")))
    try:
        assert saved_path.suffix == ".txt"
        assert saved_path.read_bytes() == b"hi"
    finally:
        saved_path.unlink()


def test_chat_errors():
    assert client.post("/api/chat", data={"message": "   "}).status_code == 422
    with patch.object(supervisor_graph, "WorkflowService", FakeWorkflowService):
        failed = client.post("/api/chat", data={"message": "fail"})
    assert failed.status_code == 500
    assert "planner exploded" in failed.json()["detail"]


def test_generated_images_become_file_urls():
    image_dir = DATA_DIR / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"api-test-{uuid.uuid4().hex}.png"
    image_path.write_bytes(b"png-bytes")
    try:
        message = chat.serialize_message(
            AIMessage(content=f"Here it is:\n\n./data/images/{image_path.name}\n\nDone.")
        )
        assert [segment["type"] for segment in message["segments"]] == [
            "text",
            "image",
            "text",
        ]
        url = message["segments"][1]["url"]
        assert url == f"/api/files/images/{image_path.name}"
        assert client.get(url).content == b"png-bytes"
    finally:
        image_path.unlink()

    # Path traversal and non-image files are refused.
    assert client.get("/api/files/..%2F.env").status_code == 404
    assert client.get("/api/files/%2E%2E/sales_multi.db").status_code == 404
    assert client.get("/api/files/images/missing.png").status_code == 404


def test_usage_dashboard():
    items = [
        {"agent": "chat_agent", "model": "gpt", "thread_id": "t1", "total_tokens": 100,
         "timestamp": "2026-11-01T10:00:00+00:00"},
        {"agent": "chat_agent", "model": "gpt", "thread_id": "t2", "total_tokens": 50,
         "timestamp": "2026-11-02T10:00:00+00:00"},
        {"agent": "sql_agent", "model": "gpt-mini", "thread_id": "t1", "total_tokens": 50,
         "timestamp": "2026-11-02T11:00:00+00:00"},
    ]
    with patch.object(dashboards, "query_items", AsyncMock(return_value=items)):
        usage = client.get("/api/usage").json()
    assert usage["total_calls"] == 3
    assert usage["total_tokens"] == 200
    assert usage["avg_tokens_per_call"] == 66.67
    assert usage["tokens_by_agent"] == [
        {"agent": "chat_agent", "total_tokens": 150, "percent": 75},
        {"agent": "sql_agent", "total_tokens": 50, "percent": 25},
    ]
    assert usage["calls_by_agent"][0] == {"agent": "chat_agent", "calls": 2}
    assert usage["daily_tokens"] == [
        {"date": "2026-11-01", "total_tokens": 100},
        {"date": "2026-11-02", "total_tokens": 100},
    ]
    assert usage["top_threads"][0] == {"thread_id": "t1", "total_tokens": 150}

    with patch.object(dashboards, "query_items", AsyncMock(return_value=[])):
        empty = client.get("/api/usage").json()
    assert empty["total_calls"] == 0 and empty["avg_tokens_per_call"] == 0

    with patch.object(
        dashboards, "query_items", AsyncMock(side_effect=RuntimeError("no cosmos"))
    ):
        assert client.get("/api/usage").status_code == 502


def test_performance_dashboard():
    items = [
        {"thread_id": "t1", "groundedness": 5, "relevance": 5, "completeness": 4,
         "timestamp": "2026-11-01T10:00:00+00:00"},
        {"thread_id": "t2", "groundedness": 3, "relevance": 5, "completeness": 4,
         "timestamp": "2026-11-01T12:00:00+00:00"},
    ]
    with patch.object(dashboards, "query_items", AsyncMock(return_value=items)):
        performance = client.get("/api/performance").json()
    assert performance["averages"] == {
        "groundedness": 4,
        "relevance": 5,
        "completeness": 4,
    }
    assert performance["daily"] == [
        {"date": "2026-11-01", "groundedness": 4, "relevance": 5, "completeness": 4}
    ]
    assert [row["thread_id"] for row in performance["low_scores"]] == ["t2"]

    with patch.object(dashboards, "query_items", AsyncMock(return_value=[])):
        empty = client.get("/api/performance").json()
    assert empty["averages"]["groundedness"] is None and empty["daily"] == []


def test_evaluation_dashboard():
    reports = client.get("/api/evaluation/reports").json()["reports"]
    assert any(report["name"] == "planner-results.json" for report in reports)

    view = client.get("/api/evaluation/reports/planner-results.json").json()
    saved = json.loads((ROOT / "evals" / "planner-results.json").read_text("utf-8"))
    assert view["source"] == "evals/planner-results.json"
    assert view["summary"]["sample_count"] == saved["sample_count"]
    assert len(view["per_intent"]) == len(saved["per_intent"])
    assert len(view["mismatches"]) == len(saved["mismatches"])
    if view["mismatches"]:
        assert view["mismatches"][0]["prompt"]

    assert client.get("/api/evaluation/reports/planner_routing.jsonl").status_code == 404
    assert client.get("/api/evaluation/reports/..%2F.env").status_code == 404

    uploaded = client.post(
        "/api/evaluation/reports",
        files={"file": ("mine.json", json.dumps(saved).encode(), "application/json")},
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["source"] == "mine.json"

    invalid = client.post(
        "/api/evaluation/reports",
        files={"file": ("bad.json", b'{"sample_count": 1}', "application/json")},
    )
    assert invalid.status_code == 422
    assert "missing required fields" in invalid.json()["detail"]


def main():
    test_health()
    test_chat_keeps_history_and_resumes_interrupts()
    test_chat_messages_carry_sent_at()
    test_messages_without_a_recorded_time_omit_sent_at()
    test_chat_upload_is_saved_and_remembered()
    test_save_uploaded_file_copies_stream()
    test_chat_errors()
    test_generated_images_become_file_urls()
    test_usage_dashboard()
    test_performance_dashboard()
    test_evaluation_dashboard()
    print("API tests passed.")


if __name__ == "__main__":
    main()
