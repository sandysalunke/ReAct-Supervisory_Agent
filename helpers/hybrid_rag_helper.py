import asyncio
import json
from config.azure_config import llm
from utils.tracing import tracer
from opentelemetry import trace
import uuid
from datetime import datetime, UTC
from helpers.cosmos_db_client import container

async def rate_the_answer(thread_id: str, query: str, answer: str, context: str):
    """
    Rate the answer and store the rating in Cosmos DB.
    - Groundedness
    - Relevance
    - Completeness
    """
    with tracer.start_as_current_span("agent.hybrid_rag.rate_answer"):
        # LLM judge to score the answer based on the question and context
        prompt = f"""
                    You are a knowledge assistant that evaluates the quality of answers provided by other knowledge assistants.
                    
                    Context: {context}
                    
                    Question: {query}

                    Answer: {answer}

                    Rules:
                        - Do not answer the question.
                        - Evaluate whether the answer is:
                            - Grounded
                            - Relevant
                            - Complete
                        - Provide scores in JSON format.
                        - Do NOT wrap the JSON in ```json or ``` code fences.
                        - Do not include explanations, markdown, or any text outside the JSON object.
                        - The JSON object should have the following structure:
                        {{
                            "groundedness": <score between 1 and 5>,
                            "relevance": <score between 1 and 5>,
                            "completeness": <score between 1 and 5>
                        }}
                """
        
        response = await llm.ainvoke(prompt)
        score = json.loads(response.content)
        # print("LLM Judge Response:", score)
        
        # Add traces in AI for observability and monitoring
        usage = response.response_metadata["token_usage"]
        trace.get_current_span().set_attribute("llm.model", response.response_metadata["model_name"])
        trace.get_current_span().set_attribute("llm.prompt_tokens", usage["prompt_tokens"])
        trace.get_current_span().set_attribute("llm.completion_tokens", usage["completion_tokens"])
        trace.get_current_span().set_attribute("llm.total_tokens", usage["total_tokens"])
        trace.get_current_span().set_attribute("llm.finish_reason", response.response_metadata["finish_reason"])
        trace.get_current_span().set_attribute("llm.score", score)

        # Add traces in Cosmos DB for cost observability and monitoring
        await asyncio.to_thread(
            container.create_item,
            {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now(UTC).isoformat(),
                "agent": "hybrid_rag_agent",
                "thread_id": thread_id,
                "model": response.response_metadata["model_name"],
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "finish_reason": response.response_metadata["finish_reason"],
                "groundedness": score.get("groundedness"),
                "relevance": score.get("relevance"),
                "completeness": score.get("completeness")
            },
        )

# Build the context from top 5 chunks retrieved from hybrid search
# Join the chunks and return the context
def build_context_hybrid_search(top5chunks):
    context_parts = []
    for index, result in enumerate(top5chunks, start=1):
        title = result.get("title") or "Unknown document"
        source = result.get("metadata_storage_path") or ""
        content = result.get("chunk") or ""
        context_parts.append(
            f"""
            [SOURCE {index}]
            Document: {title}
            Location: {source}
            {content}
            """
        )
    return "\n".join(context_parts)

# Generate the response
# Uses LLM along with context and user prompt  
async def generate_response(
    query: str,
    top5chunks,
    thread_id: str
):
    """
    Generate an answer using retrieved chunks.
    """
    with tracer.start_as_current_span("agent.hybrid_rag.generate_response", attributes={"llm_used": True}):

        context = build_context_hybrid_search(top5chunks)

        prompt = f"""
            You are an enterprise knowledge assistant.

            Answer the user's question using ONLY the supplied
            knowledge base context.

            Citation rules:

            1. Cite factual statements using the source identifier provided
            in the context, for example [1] or [2].
            2. Do not invent source identifiers.
            3. Only cite a source when that source supports the claim.
            4. Multiple sources can be cited as [1][2].
            5. Do not create a Sources section yourself.
            6. If the answer isn't supported by the supplied context, say:
            "I could not find this information in the knowledge base."

            Context:
            {context}

            Question:
            {query}
        """

        response = await llm.ainvoke(prompt)
        content = response.content
        await rate_the_answer(thread_id, query, content, context)

        # Add traces in AI for observability and monitoring
        usage = response.response_metadata["token_usage"]
        trace.get_current_span().set_attribute("llm.model", response.response_metadata["model_name"])
        trace.get_current_span().set_attribute("llm.prompt_tokens", usage["prompt_tokens"])
        trace.get_current_span().set_attribute("llm.completion_tokens", usage["completion_tokens"])
        trace.get_current_span().set_attribute("llm.total_tokens", usage["total_tokens"])
        trace.get_current_span().set_attribute("llm.finish_reason", response.response_metadata["finish_reason"])

        # Add traces in Cosmos DB for cost observability and monitoring
        await asyncio.to_thread(
            container.create_item,
            {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now(UTC).isoformat(),
                "agent": "hybrid_rag_agent",
                "thread_id": thread_id,
                "model": response.response_metadata["model_name"],
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "finish_reason": response.response_metadata["finish_reason"],
            },
        )

    return content