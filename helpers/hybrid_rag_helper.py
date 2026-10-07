from config.azure_config import llm
from .llm_cache import get_cached_response, set_cached_response

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

    # Check cache first
    cached = get_cached_response(query, context, user_groups)
    if cached is not None:
        return cached

    response = llm.invoke(prompt)
    content = response.content

    # Cache the LLM output for future identical requests
    try:
        set_cached_response(query, context, content, user_groups)
    except Exception:
        pass

    return content