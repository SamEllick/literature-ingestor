"""Retrieval: embed query → vector search → return ranked chunks."""
from openai import OpenAI

from literature_ingestor.config import settings
from literature_ingestor.store.vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.client = OpenAI(base_url=settings.lms_base_url, api_key=settings.lms_api_key)

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        filters: dict | None = None,
    ) -> list[dict]:
        """Embed query and return top-k chunks with metadata."""
        response = self.client.embeddings.create(
            model=settings.lms_embed_model,
            input=[query],
        )
        query_vector = response.data[0].embedding
        return self.vector_store.search(query_vector, top_k=top_k, filters=filters)
