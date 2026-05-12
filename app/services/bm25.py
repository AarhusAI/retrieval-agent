"""BM25 fallback retrieval.

Used as the hybrid fusion path when ``ENABLE_HYBRID_SEARCH=true`` but the
target Qdrant collection lacks sparse vectors. When sparse vectors are
present, retrieval uses Qdrant's native hybrid query API and never enters
this module.

Builds the BM25 index by scrolling the configured Qdrant collection with the
new schema (``meta.collection_name IN (collection_names)`` filter, reads
``payload.content``). Cached in-memory keyed on the sorted tuple of logical
collection names so multi-collection queries hit the same cache entry.
"""

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


# Cache key: sorted tuple of collection names — different orderings of the
# same logical collections share an entry.
CacheKey = tuple[str, ...]

_cache: dict[CacheKey, _CacheEntry] = {}
_cache_locks: dict[CacheKey, asyncio.Lock] = {}


def _make_key(collection_names: list[str]) -> CacheKey:
    return tuple(sorted(collection_names))


def _get_lock(key: CacheKey) -> asyncio.Lock:
    if key not in _cache_locks:
        _cache_locks[key] = asyncio.Lock()
    return _cache_locks[key]


def clear_cache() -> None:
    """Clear the BM25 cache (for testing)."""
    _cache.clear()
    _cache_locks.clear()


async def _get_or_build_index(
    collection_names: list[str],
) -> tuple[BM25Okapi, tuple[str, ...]] | None:
    """Get cached BM25 index or build a new one. Returns ``None`` for empty result sets."""
    key = _make_key(collection_names)
    now = time.monotonic()
    entry = _cache.get(key)
    if entry is not None and entry.expires_at > now:
        return entry.bm25, entry.texts

    lock = _get_lock(key)
    async with lock:
        # Double-check after acquiring lock
        entry = _cache.get(key)
        if entry is not None and entry.expires_at > now:
            return entry.bm25, entry.texts

        docs = await asyncio.to_thread(scroll_collection_texts, list(key))
        if not docs:
            return None

        _ids, texts = zip(*docs, strict=True)
        tokenized = [_tokenize(t) for t in texts]
        bm25 = BM25Okapi(tokenized)

        _cache[key] = _CacheEntry(
            bm25=bm25,
            texts=texts,
            expires_at=now + settings.bm25_cache_ttl_seconds,
        )
        log.info(
            "BM25 index built for %s: %d documents (TTL=%ds)",
            list(key),
            len(texts),
            settings.bm25_cache_ttl_seconds,
        )
        return bm25, texts


async def bm25_search(
    collection_names: list[str],
    query: str,
    k: int,
) -> list[tuple[str, float]]:
    """BM25 search across a set of logical collections.

    Returns ``(text, score)`` pairs sorted by score descending. Empty list when
    the collection set has no documents.
    """
    result = await _get_or_build_index(collection_names)
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
    """Fuse vector and BM25 ranked lists using Reciprocal Rank Fusion.

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
