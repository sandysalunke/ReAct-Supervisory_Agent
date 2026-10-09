import os
import sys
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Ensure tracing initialization doesn't fail during tests
os.environ.setdefault("APPLICATIONINSIGHTS_CONNECTION_STRING", f"InstrumentationKey={uuid.uuid4()}")

from agents import hybrid_rag_agent as hra


def fake_hybrid_search(question: str, top: int = 5):
    # Return a couple of fake chunks resembling search results
    return [
        {"title": "Org Leave Policy", "metadata_storage_path": "/policies/leave.pdf", "chunk": "Employees are eligible for maternity leave after 12 months of service."},
        {"title": "Maternity Details", "metadata_storage_path": "/policies/maternity.pdf", "chunk": "Maternity leave is granted for pregnancy-related medical needs and childbirth."}
    ]


def fake_generate_response(prompt, chunks, user_groups):
    # Simple deterministic response that includes the prompt and number of chunks
    return f"[FAKE RESPONSE] Prompt: {prompt} | Chunks: {len(chunks)} | Groups: {user_groups}"


def main():
    prompt = "what is the eligibility criteria for maternity leave according to the our org leave policy?"

    # Monkeypatch the hybrid_search and generate_response used by the agent
    hra.hybrid_search = fake_hybrid_search
    hra.generate_response = fake_generate_response

    state = {
        "user_input": prompt,
        "task_prompt": prompt,
        "current_task": {"id": "t1", "intent": "file_reasoning"},
        "thread_id": "test-hybrid-1"
    }

    print("Running hybrid_rag_agent with prompt:")
    print(prompt)

    result = hra.hybrid_rag_agent(state)

    print("\nAgent result:\n", result)


if __name__ == "__main__":
    main()
