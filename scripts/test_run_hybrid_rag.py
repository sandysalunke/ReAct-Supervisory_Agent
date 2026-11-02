import asyncio
import os
import sys
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Ensure tracing initialization doesn't fail during tests
os.environ.setdefault("APPLICATIONINSIGHTS_CONNECTION_STRING", f"InstrumentationKey={uuid.uuid4()}")

from agents import hybrid_rag_agent as hra


async def fake_hybrid_search(question: str, top: int = 5):
    # Return a couple of fake chunks resembling search results
    return [
        {"title": "Org Leave Policy", "metadata_storage_path": "/policies/leave.pdf", "chunk": "Employees are eligible for maternity leave after 12 months of service."},
        {"title": "Maternity Details", "metadata_storage_path": "/policies/maternity.pdf", "chunk": "Maternity leave is granted for pregnancy-related medical needs and childbirth."}
    ]


async def fake_generate_response(prompt, chunks, thread_id):
    # Simple deterministic response that includes the prompt and number of chunks
    return f"[FAKE RESPONSE] Prompt: {prompt} | Chunks: {len(chunks)} | Thread: {thread_id}"


class FakeEmbeddingWrapper:
    async def aembed_query(self, query):
        return [0.1, 0.2]


async def main():
    prompt = "what is the eligibility criteria for maternity leave according to the our org leave policy?"

    hra.get_cached_response = lambda query, groups: None
    hra.get_semantic_cached_response = lambda query, embedding, groups: None
    hra.set_cached_response = lambda query, embedding, response, groups: None
    hra.AzureEmbeddingWrapper = FakeEmbeddingWrapper
    hra.hybrid_search = fake_hybrid_search
    hra.generate_response = fake_generate_response

    state = {
        "user_input": prompt,
        "task_prompt": prompt,
        "current_task": {
            "id": "t1",
            "intent": "knowledge_search",
            "dependencies": [],
        },
        "thread_id": "test-hybrid-1"
    }

    print("Running hybrid_rag_agent with prompt:")
    print(prompt)

    result = await hra.hybrid_rag_agent(state)

    print("\nAgent result:\n", result)


if __name__ == "__main__":
    asyncio.run(main())
