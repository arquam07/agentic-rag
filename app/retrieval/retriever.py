"""
Hybrid Retriever — Dense (Qdrant) + Sparse (BM25) with Reciprocal Rank Fusion.
"""

import logging
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.ingestion.vector_store import get_client, COLLECTION_NAME

logger = logging.getLogger(__name__)

# same embedding model used at ingestion
_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    """Lazy-load the embedding model (loaded once, reused)."""
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


# data class for a retrieved chunk
@dataclass
class RetrievedChunk:
    text: str
    source_file: str
    chunk_index: int
    score: float  # final fused score


# core retrieval functions

def dense_search(query: str, top_k: int = 20) -> list[dict]:
    """
    Embed the query and fetch top_k nearest neighbours from Qdrant.
    Returns list of dicts with keys: id, text, source_file, chunk_index, score.
    """
    model = _get_embed_model()
    query_vector = model.encode(query).tolist()

    client = get_client()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "id": str(hit.id),
            "text": hit.payload["text"],
            "source_file": hit.payload.get("source_file", ""),
            "chunk_index": hit.payload.get("chunk_index", 0),
            "score": hit.score,
        }
        for hit in results.points
    ]


def bm25_rerank(query: str, docs: list[dict]) -> list[dict]:
    """
    Score the already-retrieved docs with BM25 and return them sorted.
    This is the 'sparse' half of the hybrid — no extra Qdrant index needed.
    """
    if not docs:
        return []

    corpus = [doc["text"].lower().split() for doc in docs]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.lower().split())

    for doc, score in zip(docs, scores):
        doc["bm25_score"] = float(score)

    return sorted(docs, key=lambda d: d["bm25_score"], reverse=True)


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF) — merges multiple ranked lists.
    Formula per doc:  RRF_score = Σ  1 / (k + rank_i)
    """
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = doc["id"]
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            doc_map[doc_id] = doc

    sorted_ids = sorted(fused_scores, key=lambda d: fused_scores[d], reverse=True)

    results = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id]
        doc["rrf_score"] = fused_scores[doc_id]
        results.append(doc)

    return results


# ── public API ──────────────────────────────────────────────────────

def hybrid_retrieve(query: str, top_k: int = 5, dense_fetch: int = 20) -> list[RetrievedChunk]:
    """
    Main entry point.

    1. Dense search  → pull `dense_fetch` candidates from Qdrant
    2. BM25 re-rank  → score same candidates with BM25
    3. RRF fusion    → merge both ranked lists
    4. Return top_k  → final results
    """
    logger.info("Hybrid retrieval for: %s", query)

    # 1 — dense
    dense_results = dense_search(query, top_k=dense_fetch)
    dense_ranked = sorted(dense_results, key=lambda d: d["score"], reverse=True)
    print(f"DENSE CHUNKS", dense_ranked)
    # 2 — sparse (BM25 over the same docs)
    bm25_ranked = bm25_rerank(query, [d.copy() for d in dense_results])
    print(f"SPARCE CHUNKS", bm25_ranked)

    # 3 — fuse
    fused = reciprocal_rank_fusion([dense_ranked, bm25_ranked])

    # 4 — top K
    top = fused[:top_k]

    return [
        RetrievedChunk(
            text=doc["text"],
            source_file=doc["source_file"],
            chunk_index=doc["chunk_index"],
            score=doc["rrf_score"],
        )
        for doc in top
    ]