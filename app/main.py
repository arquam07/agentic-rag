import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.ingestion.router import router as ingest_router
from app.ingestion.vector_store import ensure_collection
from app.router import query_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once on startup — ensures Qdrant collection exists
    ensure_collection()
    yield


app = FastAPI(
    title="Agentic RAG Pipeline",
    description="Document ingestion + agentic retrieval API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(ingest_router, prefix="/ingest", tags=["Ingestion"])
app.include_router(query_router, prefix="/query", tags=["Query"])