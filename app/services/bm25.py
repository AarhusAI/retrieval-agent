import asyncio
import logging
import time
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.config import settings
from app.services.qdrant import scroll_collection_texts

log = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


@dataclass
class _CacheEntry:
    bm25: BM25Okapi
    texts: tuple[str, ...]
    expires_at: float


_cache: dict[str, _CacheEntry] = {}
_cache_locks: dict[str, asyncio.Lock] = {}


def _get_lock(collection_name: str) -> asyncio.Lock:
    if collection_name not in _cache_locks:
        _cache_locks[collection_name] = asyncio.Lock()
    return _cache_locks[collection_name]


def clear_cache() -> None:
    """Clear the BM25 cache (for testing)."""
    _cache.clear()
    _cache_locks.clear()


async def _get_or_build_index(
    collection_name: str,
) -> tuple[BM25Okapi, tuple[str, ...]] | None:
    """Get cached BM25 index or build a new one. Returns None for empty collections."""
    now = time.monotonic()
    entry = _cache.get(collection_name)
    if entry is not None and entry.expires_at > now:
        return entry.bm25, entry.texts

    lock = _get_lock(collection_name)
    async with lock:
        # Double-check after acquiring lock
        entry = _cache.get(collection_name)
        if entry is not None and entry.expires_at > now:
            return entry.bm25, entry.texts

        docs = await asyncio.to_thread(scroll_collection_texts, collection_name)
        if not docs:
            return None

        _ids, texts = zip(*docs, strict=True)
        tokenized = [_tokenize(t) for t in texts]
        bm25 = BM25Okapi(tokenized)

        _cache[collection_name] = _CacheEntry(
            bm25=bm25,
            texts=texts,
            expires_at=now + settings.bm25_cache_ttl_seconds,
        )
        log.info(
            "BM25 index built for %s: %d documents (TTL=%ds)",
            collection_name,
            len(texts),
            settings.bm25_cache_ttl_seconds,
        )
        return bm25, texts


async def bm25_search(
    collection_name: str,
    query: str,
    k: int,
) -> list[tuple[str, float]]:
    """
    BM25 search over all documents in a collection.
    Returns list of (text, score) sorted by score descending.
    Uses a TTL-based in-memory cache to avoid rebuilding the index on every query.
    """
    result = await _get_or_build_index(collection_name)
    if result is None:
        return []

    bm25, texts = result
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
