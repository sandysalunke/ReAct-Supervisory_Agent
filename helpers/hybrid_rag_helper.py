from config.azure_config import llm
from utils.tracing import tracer
from opentelemetry import trace

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
def generate_response(
    query: str,
    top5chunks,
    user_groups
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

        response = llm.invoke(prompt)
        content = response.content

        usage = response.response_metadata["token_usage"]
        trace.get_current_span().set_attribute("llm.model", response.response_metadata["model_name"])
        trace.get_current_span().set_attribute("llm.prompt_tokens", usage["prompt_tokens"])
        trace.get_current_span().set_attribute("llm.completion_tokens", usage["completion_tokens"])
        trace.get_current_span().set_attribute("llm.total_tokens", usage["total_tokens"])
        trace.get_current_span().set_attribute("llm.finish_reason", response.response_metadata["finish_reason"])

    return content