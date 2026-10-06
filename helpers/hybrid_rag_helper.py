from config.azure_config import llm

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
    top5chunks
):
    """
    Generate an answer using retrieved chunks.
    """

    context = build_context_hybrid_search(top5chunks)

    prompt = f"""
        Use the provided context to answer the question.

        Context:
        {context}

        Question:
        {query}
    """

    response = llm.invoke(prompt)

    return response.content