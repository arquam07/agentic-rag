# Agentic RAG Pipeline

Document ingestion and retrieval API built with FastAPI, Qdrant, and LangGraph.

Ingest documents (PDF, DOCX, etc.), parse and chunk using Docling, and embed them into Qdrant, then ask questions using hybrid retrieval (dense + BM25) with an LLM of your choice.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

## Running

```bash
uvicorn app.main:app --reload --port 8000
```

Docs at `http://localhost:8000/docs`

## Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

## API

**Ingest a document**

```
POST /ingest/
Content-Type: multipart/form-data

file: <your-file>
```

**Ask a question**

```
POST /query/
Content-Type: application/json

{ "question": "What does the document say about X?" }
```

Returns:

```json
{
  "answer": "...",
  "sources": ["filename.pdf"],
  "num_chunks_used": 5
}
```

## LLM Providers

Set `LLM_PROVIDER` in `.env`:

- `ollama` — local inference, needs [Ollama](https://ollama.com) running with your model pulled (`ollama pull llama3.2`)
- `deepseek` — cloud API, set `DEEPSEEK_API_KEY`

## How retrieval works

1. Query gets embedded and searched against Qdrant (dense/semantic search)
2. Top candidates are re-scored with BM25 (keyword match)
3. Both rankings are merged using Reciprocal Rank Fusion
4. Top chunks are passed to the LLM for answer generation

## Project structure

```
app/
├── main.py                 # FastAPI app
├── router.py               # Query endpoint
├── ingestion/
│   ├── chunker.py          # Text chunking
│   ├── embedder.py         # Sentence-transformers embedding
│   ├── parser.py           # Document parsing (Docling)
│   ├── router.py           # Ingest endpoint
│   └── vector_store.py     # Qdrant operations
├── retrieval/
│   └── retriever.py        # Hybrid search (dense + BM25 + RRF)
├── generation/
│   └── llm.py              # LLM factory (Ollama / DeepSeek)
└── agent/
    └── graph.py            # LangGraph RAG pipeline
```

## Stack

- **FastAPI** — API server
- **Qdrant** — vector database
- **Sentence-Transformers** — embeddings (all-MiniLM-L6-v2)
- **LangGraph** — agent orchestration
- **LangChain** — LLM abstraction
- **rank-bm25** — sparse retrieval
- **Docling** — document parsing