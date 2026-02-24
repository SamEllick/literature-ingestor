"""RAG generation: retrieved chunks + query → answer with citations."""
from openai import OpenAI

from literature_ingestor.config import settings
from literature_ingestor.rag.retriever import Retriever


_SYSTEM_PROMPT = """\
You are a research assistant. Answer the user's question using ONLY the provided \
context excerpts from academic papers. For every claim you make, cite the source \
using [Title, Section] notation. If the context does not contain enough information \
to answer, say so explicitly. Do not invent facts.\
"""


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        p = chunk["payload"]
        title = p.get("title") or p.get("filename", "Unknown")
        section = p.get("section", "")
        year = p.get("year", "")
        header = f"[{i}] {title} ({year}) — {section}"
        parts.append(f"{header}\n{p['text']}")
    return "\n\n---\n\n".join(parts)


class RAGPipeline:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.client = OpenAI(base_url=settings.lms_base_url, api_key=settings.lms_api_key)

    def query(
        self,
        question: str,
        top_k: int = 8,
        filters: dict | None = None,
        stream: bool = False,
    ):
        chunks = self.retriever.retrieve(question, top_k=top_k, filters=filters)
        context = _build_context(chunks)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n\n{context}\n\nQuestion: {question}",
            },
        ]

        response = self.client.chat.completions.create(
            model=settings.lms_chat_model,
            messages=messages,
            stream=stream,
        )

        if stream:
            return response  # caller iterates

        return {
            "answer": response.choices[0].message.content,
            "sources": [
                {
                    "title": c["payload"].get("title") or c["payload"].get("filename"),
                    "section": c["payload"].get("section"),
                    "year": c["payload"].get("year"),
                    "doi": c["payload"].get("doi"),
                    "score": c["score"],
                }
                for c in chunks
            ],
        }
