# literature-ingestor

A RAG pipeline for ingesting academic PDFs and interrogating them with a locally-hosted LLM via [LM Studio](https://lmstudio.ai/).

"Co"-authored by Cluade

## Stack

| Layer | Technology |
|---|---|
| PDF parsing | [Marker](https://github.com/VikParuchuri/marker) |
| Embeddings & chat | LM Studio (OpenAI-compatible API) |
| Vector store | ChromaDB (embedded) |
| Metadata store | SQLite |
| API | FastAPI |
| CLI | Typer |

---

## Setup


### Start LM Studio server

Enable the local server in LM Studio (default: `http://localhost:1234`), and load your chat and embedding models.

---

## CLI

The CLI entrypoint is `lit`.

### `lit ingest`

Ingest a single PDF or a directory of PDFs into the vector store.

```bash
lit ingest <path>
```

| Flag | Short | Description |
|---|---|---|
| `--force` | `-f` | Re-ingest even if already processed |

**Examples:**

```bash
# Ingest a single PDF
lit ingest data/papers/attention.pdf

# Ingest all PDFs in a directory
lit ingest data/papers/

# Force re-ingest (bypass duplicate check)
lit ingest data/papers/attention.pdf --force
```

---

### `lit query`

Ask a question against all ingested literature. Returns an answer with citations.

```bash
lit query "<question>"
```

| Flag | Short | Default | Description |
|---|---|---|---|
| `--top-k` | `-k` | `8` | Number of chunks to retrieve |
| `--year` | `-y` | — | Filter results to a specific publication year |

**Examples:**

```bash
# Basic question
lit query "What loss functions are used for semantic segmentation?"

# Retrieve more context
lit query "Explain the proposed architecture" --top-k 15

# Filter by year
lit query "What transformer variants are proposed?" --year 2023
```

---

### `lit papers`

List all ingested papers with metadata.

```bash
lit papers
```

Displays a table of ID, title, authors, year, chunk count, and filename.

