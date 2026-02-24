"""FastAPI application exposing /ingest and /query endpoints."""
import shutil
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from literature_ingestor.config import settings
from literature_ingestor.ingest.pipeline import ingest_pdf
from literature_ingestor.rag.generator import RAGPipeline
from literature_ingestor.rag.retriever import Retriever
from literature_ingestor.store.metadata_store import MetadataStore
from literature_ingestor.store.vector_store import VectorStore

app = FastAPI(title="Literature Ingestor", version="0.1.0")

# Shared singletons
_metadata_store = MetadataStore(settings.metadata_db_path)
_vector_store = VectorStore()
_retriever = Retriever(_vector_store)
_rag = RAGPipeline(_retriever)


# ── Ingest ────────────────────────────────────────────────────────────────────

@app.post("/ingest")
async def ingest(file: UploadFile = File(...), force: bool = False):
    """Upload and ingest a PDF."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    upload_dir = Path(settings.papers_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    result = ingest_pdf(dest, _metadata_store, _vector_store, force=force)
    return result


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = 8
    year: int | None = None
    stream: bool = False


@app.post("/query")
def query(req: QueryRequest):
    """Ask a question against the ingested literature."""
    filters = {"year": req.year} if req.year else None

    if req.stream:
        def generate():
            for chunk in _rag.query(req.question, top_k=req.top_k, filters=filters, stream=True):
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield delta

        return StreamingResponse(generate(), media_type="text/plain")

    return _rag.query(req.question, top_k=req.top_k, filters=filters)


# ── Papers list ───────────────────────────────────────────────────────────────

@app.get("/papers")
def list_papers():
    """List all ingested papers."""
    papers = _metadata_store.list_papers()
    return [
        {
            "id": p.id,
            "filename": p.filename,
            "title": p.title,
            "authors": p.authors,
            "year": p.year,
            "doi": p.doi,
            "chunk_count": p.chunk_count,
            "ingested_at": p.ingested_at,
        }
        for p in papers
    ]
