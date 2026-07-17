from config.azure_config import llm

# Build the context
# Join the similar chunks and return the context
def build_context(
    retrieved_docs
):
    """
    Convert retrieved documents into a prompt context.
    """

    return "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

# Generate the response
# Uses LLM along with context and user prompt  
def generate_response(
    query: str,
    retrieved_docs
):
    """
    Generate an answer using retrieved chunks.
    """

    context = build_context(retrieved_docs)

    prompt = f"""
        Use the provided context to answer the question.

        Context:
        {context}

        Question:
        {query}
    """

    response = llm.invoke(prompt)

    return response.content