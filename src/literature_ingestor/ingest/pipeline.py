"""End-to-end ingestion pipeline: PDF → chunks → embeddings → stores."""
import uuid
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from literature_ingestor.config import settings
from literature_ingestor.ingest.chunker import chunk_markdown
from literature_ingestor.ingest.parser import parse_pdf
from literature_ingestor.store.metadata_store import MetadataStore
from literature_ingestor.store.vector_store import VectorStore


def _embed(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=settings.lms_embed_model, input=texts)
    return [item.embedding for item in response.data]


def ingest_pdf(
    pdf_path: Path,
    metadata_store: MetadataStore,
    vector_store: VectorStore,
    batch_size: int = 16,
    force: bool = False,
) -> dict:
    """Ingest a single PDF. Returns a status dict."""
    content_hash = metadata_store.hash_file(pdf_path)

    if not force and metadata_store.is_ingested(content_hash):
        return {"status": "skipped", "file": pdf_path.name, "reason": "already ingested"}

    # 1. Parse
    parsed = parse_pdf(pdf_path)

    # 2. Store metadata
    paper_id = metadata_store.add_paper(
        content_hash=content_hash,
        filename=pdf_path.name,
        title=parsed.title,
        authors=parsed.authors,
        year=parsed.year,
        doi=parsed.doi,
        abstract=parsed.abstract,
    )

    # 3. Chunk
    chunks = chunk_markdown(parsed.markdown, paper_id=paper_id)
    if not chunks:
        return {"status": "error", "file": pdf_path.name, "reason": "no chunks produced"}

    # 4. Embed + index in batches
    client = OpenAI(base_url=settings.lms_base_url, api_key=settings.lms_api_key)
    points: list[dict] = []

    for i in tqdm(range(0, len(chunks), batch_size), desc=f"Embedding {pdf_path.name}", leave=False):
        batch = chunks[i : i + batch_size]
        vectors = _embed(client, [c.text for c in batch])
        for chunk, vector in zip(batch, vectors):
            points.append({
                "id": str(uuid.uuid4()),
                "vector": vector,
                "payload": {
                    "paper_id": paper_id,
                    "filename": pdf_path.name,
                    "title": parsed.title,
                    "authors": parsed.authors,
                    "year": parsed.year,
                    "doi": parsed.doi,
                    "section": chunk.section,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                },
            })

    vector_store.upsert(points)
    metadata_store.mark_indexed(paper_id, chunk_count=len(chunks))

    return {
        "status": "ok",
        "file": pdf_path.name,
        "paper_id": paper_id,
        "chunks": len(chunks),
    }


def ingest_directory(
    directory: Path,
    metadata_store: MetadataStore,
    vector_store: VectorStore,
    force: bool = False,
) -> list[dict]:
    """Ingest all PDFs in a directory."""
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        return [{"status": "error", "reason": f"no PDFs found in {directory}"}]

    results = []
    for pdf in tqdm(pdfs, desc="Ingesting papers"):
        result = ingest_pdf(pdf, metadata_store, vector_store, force=force)
        results.append(result)

    return results
