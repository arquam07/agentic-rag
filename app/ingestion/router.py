"""
Ingestion Router — upload documents for parsing, chunking, and embedding.
"""

import logging
from fastapi import APIRouter, UploadFile, File, Query, HTTPException

from app.ingestion.parser import parse_document
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import embed_chunks
from app.ingestion.vector_store import upsert_chunks, delete_document

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/")
async def ingest_document(
    file: UploadFile = File(...),
    tier: str = Query(
        default="basic",
        regex="^(basic|premium)$",
        description="Parsing tier: 'basic' (text+tables) or 'premium' (+ image descriptions)",
    ),
):
    """
    Ingest a document into the vector store.

    1. Parse the file to markdown (basic or premium tier)
    2. Chunk with table-aware splitting
    3. Embed chunks
    4. Upsert to Qdrant (deletes old chunks for same filename first)
    """
    filename = file.filename
    logger.info("Ingesting '%s' with tier='%s'", filename, tier)

    try:
        file_bytes = await file.read()

        # Parse
        markdown = parse_document(file_bytes, filename, tier=tier)
        if not markdown.strip():
            raise HTTPException(status_code=400, detail="Document parsed to empty content.")

        # Chunk (table-aware)
        chunks = chunk_text(markdown, source_file=filename)
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks produced from document.")

        logger.info("Produced %d chunks from '%s'", len(chunks), filename)

        # Embed
        embeddings = embed_chunks([c.text for c in chunks])

        # Delete old version if re-ingesting same file
        delete_document(filename)

        # Upsert
        num_upserted = upsert_chunks(chunks, embeddings, source_file=filename)

        return {
            "filename": filename,
            "tier": tier,
            "num_chunks": num_upserted,
            "status": "ok",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Ingestion failed for '%s'", filename)
        raise HTTPException(status_code=500, detail=str(e))