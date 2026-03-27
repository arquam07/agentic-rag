import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.ingestion.parser import parse_document
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import embed_texts
from app.ingestion.vector_store import upsert_chunks, delete_document

logger = logging.getLogger(__name__)
router = APIRouter()

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".md", ".txt"}


class IngestResponse(BaseModel):
    filename: str
    chunks_upserted: int
    message: str


@router.post("/", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload a document → parse → chunk → embed → upsert to Qdrant.
    """
    filename = file.filename or "unknown"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}"
        )

    try:
        file_bytes = await file.read()

        logger.info(f"[ingest] Parsing {filename} ({len(file_bytes)} bytes)")
        markdown = parse_document(file_bytes, filename)

        logger.info(f"[ingest] Chunking {filename}")
        chunks = chunk_text(markdown, source_file=filename)

        if not chunks:
            raise HTTPException(status_code=422, detail="Document produced no text chunks.")

        logger.info(f"[ingest] Embedding {len(chunks)} chunks")
        embeddings = embed_texts([c.text for c in chunks])

        # Remove old version of the same file before upserting
        delete_document(filename)

        logger.info(f"[ingest] Upserting to Qdrant")
        count = upsert_chunks(chunks, embeddings, source_file=filename)

        return IngestResponse(
            filename=filename,
            chunks_upserted=count,
            message="Document ingested successfully."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[ingest] Failed for {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))