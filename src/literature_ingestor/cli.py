"""CLI for batch ingestion and interactive querying."""
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from literature_ingestor.config import settings
from literature_ingestor.ingest.pipeline import ingest_directory, ingest_pdf
from literature_ingestor.rag.generator import RAGPipeline
from literature_ingestor.rag.retriever import Retriever
from literature_ingestor.store.metadata_store import MetadataStore
from literature_ingestor.store.vector_store import VectorStore

app = typer.Typer(help="Literature Ingestor CLI")
console = Console()


def _stores():
    return MetadataStore(settings.metadata_db_path), VectorStore()


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="PDF file or directory of PDFs"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest even if already processed"),
):
    """Ingest one PDF or an entire directory."""
    meta, vec = _stores()

    if path.is_dir():
        results = ingest_directory(path, meta, vec, force=force)
    elif path.suffix.lower() == ".pdf":
        results = [ingest_pdf(path, meta, vec, force=force)]
    else:
        console.print("[red]Path must be a .pdf file or a directory.[/red]")
        raise typer.Exit(1)

    table = Table("File", "Status", "Chunks")
    for r in results:
        status = r.get("status", "?")
        color = "green" if status == "ok" else "yellow" if status == "skipped" else "red"
        table.add_row(r.get("file", "?"), f"[{color}]{status}[/{color}]", str(r.get("chunks", "-")))
    console.print(table)


@app.command()
def query(
    question: str = typer.Argument(..., help="Your research question"),
    top_k: int = typer.Option(8, "--top-k", "-k"),
    year: int = typer.Option(None, "--year", "-y", help="Filter by publication year"),
):
    """Ask a question against the ingested literature."""
    _, vec = _stores()
    retriever = Retriever(vec)
    rag = RAGPipeline(retriever)

    filters = {"year": year} if year else None
    result = rag.query(question, top_k=top_k, filters=filters)

    console.rule("[bold]Answer[/bold]")
    console.print(result["answer"])

    console.rule("[bold]Sources[/bold]")
    for s in result["sources"]:
        console.print(f"  • {s['title']} ({s['year']}) — {s['section']}  score={s['score']:.3f}")


@app.command()
def papers():
    """List all ingested papers."""
    meta, _ = _stores()
    rows = meta.list_papers()
    table = Table("ID", "Title", "Authors", "Year", "Chunks", "Filename")
    for p in rows:
        table.add_row(
            str(p.id),
            p.title or "-",
            (p.authors or "-")[:40],
            str(p.year or "-"),
            str(p.chunk_count),
            p.filename,
        )
    console.print(table)


if __name__ == "__main__":
    app()
