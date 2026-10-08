import os
import json
import sys
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Use a valid UUID for a dummy instrumentation key so azure monitor config doesn't fail
os.environ.setdefault("APPLICATIONINSIGHTS_CONNECTION_STRING", f"InstrumentationKey={uuid.uuid4()}")

from graphs.supervisor_graph import WorkflowService


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


def main():
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
    svc = WorkflowService(request)
    print("WorkflowService created. Graph compiled and ready.")


if __name__ == "__main__":
    main()
