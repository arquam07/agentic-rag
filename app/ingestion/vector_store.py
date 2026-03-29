import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, Filter, FieldCondition, MatchValue,
    PayloadSchemaType,
)
from dotenv import load_dotenv

load_dotenv()

API_KEY           = os.environ.get("QDRANT_API_KEY", "").strip() or None
URL               = os.environ.get("QDRANT_URL")
COLLECTION_NAME   = os.environ.get("QDRANT_COLLECTION", "documents")
VECTOR_SIZE       = 384        # all-MiniLM-L6-v2 output dimension


def get_client() -> QdrantClient:
    return QdrantClient(url=URL, api_key=API_KEY)


def ensure_collection() -> None:
    """Create the Qdrant collection if it doesn't exist yet."""
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    # Ensure payload index exists for filtering
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="source_file",
        field_schema=PayloadSchemaType.KEYWORD,
    )


def upsert_chunks(chunks, embeddings: list[list[float]], source_file: str) -> int:
    """
    Upsert chunk text + embedding into Qdrant.
    Returns number of points upserted.
    """
    client = get_client()
    ensure_collection()

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk.text,
                "source_file": source_file,
                "chunk_index": chunk.chunk_index,
            },
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def delete_document(source_file: str) -> int:
    """
    Delete all chunks belonging to a specific source file.
    Useful for re-ingestion without duplicates.
    """
    client = get_client()
    result = client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(
                key="source_file",
                match=MatchValue(value=source_file)
            )]
        ),
    )
    return result.operation_id