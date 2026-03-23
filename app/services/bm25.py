import asyncio
import logging

from rank_bm25 import BM25Okapi

from app.services.qdrant import scroll_collection_texts

log = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


async def bm25_search(
    collection_name: str,
    query: str,
    k: int,
) -> list[tuple[str, float]]:
    """
    BM25 search over all documents in a collection.
    Returns list of (text, score) sorted by score descending.
    """
    docs = await asyncio.to_thread(scroll_collection_texts, collection_name)
    if not docs:
        return []

    _ids, texts = zip(*docs, strict=True)
    tokenized = [_tokenize(t) for t in texts]

    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(_tokenize(query))

    scored = list(zip(texts, scores, strict=True))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def reciprocal_rank_fusion(
    vector_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    bm25_weight: float,
    k_rrf: int = 60,
) -> list[tuple[str, float]]:
    """
    Fuse vector and BM25 ranked lists using Reciprocal Rank Fusion.
    Returns merged list sorted by fused score descending.
    """
    vector_weight = 1.0 - bm25_weight
    scores: dict[str, float] = {}

    for rank, (text, _) in enumerate(vector_results):
        scores[text] = scores.get(text, 0.0) + vector_weight / (k_rrf + rank + 1)

    for rank, (text, _) in enumerate(bm25_results):
        scores[text] = scores.get(text, 0.0) + bm25_weight / (k_rrf + rank + 1)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused
