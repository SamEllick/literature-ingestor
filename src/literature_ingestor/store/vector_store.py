"""ChromaDB vector store operations."""
import chromadb
from chromadb.config import Settings as ChromaSettings

from literature_ingestor.config import settings


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, points: list[dict]):
        """points: list of {id, vector, payload}"""
        self.collection.upsert(
            ids=[p["id"] for p in points],
            embeddings=[p["vector"] for p in points],
            metadatas=[p["payload"] for p in points],
            documents=[p["payload"].get("text", "") for p in points],
        )

    def search(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[dict]:
        where = None
        if filters:
            conditions = {k: {"$eq": v} for k, v in filters.items() if v is not None}
            if len(conditions) == 1:
                where = conditions
            elif len(conditions) > 1:
                where = {"$and": [{k: v} for k, v in conditions.items()]}

        results = self.collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, self.collection.count() or 1),
            where=where,
            include=["metadatas", "distances"],
        )

        chunks = []
        for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
            # ChromaDB cosine distance is 1 - similarity; invert for a similarity score
            chunks.append({"score": 1.0 - distance, "payload": metadata})
        return chunks

    def delete_by_paper_id(self, paper_id: int):
        self.collection.delete(where={"paper_id": {"$eq": paper_id}})
