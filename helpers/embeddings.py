from langchain_core.embeddings import Embeddings
from config.azure_config import client, EMBED_MODEL
from utils.tracing import tracer
from opentelemetry import trace

class AzureEmbeddingWrapper(Embeddings):

    def embed_documents(self, texts):
        with tracer.start_as_current_span("embed.documents", attributes={"count": len(texts)}):
            response = client.embeddings.create(
                model=EMBED_MODEL,
                input=texts
            )

            embeddings = [item.embedding for item in response.data]
            trace.get_current_span().set_attribute("embeddings.count", len(embeddings))
            return embeddings

    def embed_query(self, text):
        with tracer.start_as_current_span("embed.query", attributes={"text_len": len(str(text) or "")}):
            response = client.embeddings.create(
                model=EMBED_MODEL,
                input=text
            )

            embedding = response.data[0].embedding or ""
            trace.get_current_span().set_attribute("embedding.len", len(embedding) if hasattr(embedding, '__len__') else 0)
            return embedding