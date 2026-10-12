from urllib import response

from langchain_core.embeddings import Embeddings
from config.azure_config import client, EMBED_MODEL
from utils.tracing import tracer
from opentelemetry import trace
import uuid
from datetime import datetime, UTC
from helpers.cosmos_db_client import container

class AzureEmbeddingWrapper(Embeddings):

    def add_usage_tracing(self, response):
        # Add traces in Cosmos DB for cost observability and monitoring
        container.create_item({
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "agent": "query_embedding",
            "thread_id": "",
            "model": response.model,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": "",
            "total_tokens": response.usage.total_tokens,
            "finish_reason": "",
        })

    def embed_documents(self, texts):
        with tracer.start_as_current_span("embed.documents", attributes={"count": len(texts)}):
            response = client.embeddings.create(
                model=EMBED_MODEL,
                input=texts
            )

            self.add_usage_tracing(response)
            
            embeddings = [item.embedding for item in response.data]
            trace.get_current_span().set_attribute("embeddings.count", len(embeddings))
            return embeddings

    def embed_query(self, text):
        with tracer.start_as_current_span("embed.query", attributes={"text_len": len(str(text) or "")}):
            response = client.embeddings.create(
                model=EMBED_MODEL,
                input=text
            )

            self.add_usage_tracing(response)

            embedding = response.data[0].embedding or ""
            trace.get_current_span().set_attribute("embedding.len", len(embedding) if hasattr(embedding, '__len__') else 0)
            return embedding
