from langchain_core.embeddings import Embeddings
from config.azure_config import client, EMBED_MODEL

class AzureEmbeddingWrapper(Embeddings):

    def embed_documents(self, texts):
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=texts
        )

        return [item.embedding for item in response.data]

    def embed_query(self, text):
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=text
        )

        return response.data[0].embedding or ""