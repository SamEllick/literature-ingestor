# literature-ingestor

A RAG pipeline for ingesting academic PDFs and interrogating them with a locally-hosted LLM via [LM Studio](https://lmstudio.ai/).

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

### 1. Install

```bash
conda activate lcf
pip install -e .
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env`:

```env
LMS_CHAT_MODEL=      # model name as shown in LM Studio
LMS_EMBED_MODEL=     # embedding model name
LMS_EMBED_DIM=768    # must match embedding model output dimension
MARKER_DEVICE=cpu    # use "cuda" if you have spare VRAM (≥10 GB recommended)
```

Make sure both models are loaded and the server is running in LM Studio before use.

### 3. Start LM Studio server

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

---

## API

Start the REST API:

```bash
uvicorn literature_ingestor.api.app:app --reload
```

Interactive docs available at **http://localhost:8000/docs**.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest` | Upload and ingest a PDF (`multipart/form-data`) |
| `POST` | `/query` | Ask a question (JSON body) |
| `GET` | `/papers` | List all ingested papers |

**Examples:**

```bash
# Ingest
curl -X POST http://localhost:8000/ingest -F "file=@paper.pdf"

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What datasets are used?", "top_k": 8}'

# Query with year filter and streaming
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What methods are compared?", "year": 2022, "stream": true}'

# List papers
curl http://localhost:8000/papers
```

---

## Data layout

```
data/
├── chroma/        # ChromaDB vector index (auto-created)
├── metadata.db    # SQLite paper registry (auto-created)
└── papers/        # PDFs land here on API upload
```

All contents of `data/` are gitignored. Delete the directory to reset all state.
