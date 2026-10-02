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
from collections import OrderedDict
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app import metrics
from app.config import settings
from app.log_utils import sanitize_for_log
from app.services.qdrant import scroll_collection_texts

log = logging.getLogger(__name__)

# Hard cap on cached BM25 indexes. An attacker (or a buggy caller) can
# otherwise grow ``_cache`` and ``_cache_locks`` indefinitely by varying
# ``collection_names`` sets per request, since each unique sorted tuple is a
# new entry. FIFO eviction is enough — entries already have a TTL.
_CACHE_MAX_ENTRIES = 32


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


@dataclass
class _CacheEntry:
    bm25: BM25Okapi
    texts: tuple[str, ...]
    metas: tuple[dict, ...]
    expires_at: float


# Cache key: sorted tuple of collection names — different orderings of the
# same logical collections share an entry.
CacheKey = tuple[str, ...]

_cache: OrderedDict[CacheKey, _CacheEntry] = OrderedDict()
_cache_locks: OrderedDict[CacheKey, asyncio.Lock] = OrderedDict()


def clear_cache() -> None:
    """Clear the BM25 cache (for testing)."""
    _cache.clear()
    _cache_locks.clear()


async def _get_or_build_index(
    collection_names: list[str],
) -> tuple[BM25Okapi, tuple[str, ...], tuple[dict, ...]] | None:
    """Get cached BM25 index or build a new one. Returns ``None`` for empty result sets."""
    key = tuple(sorted(collection_names))
    now = time.monotonic()
    # Uncontended asyncio locks are cheap; holding one for the lookup too means
    # concurrent misses for the same key build the index only once.
    async with _cache_locks.setdefault(key, asyncio.Lock()):
        entry = _cache.get(key)
        if entry is not None and entry.expires_at > now:
            metrics.bm25_cache_total.labels(result="hit").inc()
            return entry.bm25, entry.texts, entry.metas

        metrics.bm25_cache_total.labels(result="miss").inc()
        docs = await asyncio.to_thread(scroll_collection_texts, list(key))
        if not docs:
            # No cache entry is inserted for empty scopes, so the eviction in
            # the insert path below would never reclaim this key's lock —
            # drop it here or repeated lookups of empty scopes leak locks.
            _cache_locks.pop(key, None)
            return None

        texts, metas = zip(*docs, strict=True)
        tokenized = [_tokenize(t) for t in texts]
        bm25 = BM25Okapi(tokenized)

        # Evict oldest entries (FIFO) before inserting if at capacity. The
        # lock dict is evicted in lockstep so it can't grow on its own.
        while len(_cache) >= _CACHE_MAX_ENTRIES:
            evicted_key, _ = _cache.popitem(last=False)
            _cache_locks.pop(evicted_key, None)
            log.debug("Evicted BM25 cache entry: %s", sanitize_for_log(list(evicted_key)))

        _cache[key] = _CacheEntry(
            bm25=bm25,
            texts=texts,
            metas=metas,
            expires_at=now + settings.bm25_cache_ttl_seconds,
        )
        log.info(
            "BM25 index built for %s: %d documents (TTL=%ds)",
            sanitize_for_log(list(key)),
            len(texts),
            settings.bm25_cache_ttl_seconds,
        )
        return bm25, texts, metas


async def bm25_search(
    collection_names: list[str],
    query: str,
    k: int,
) -> list[tuple[str, float, dict]]:
    """BM25 search across a set of logical collections.

    Returns ``(text, score, meta)`` triples sorted by score descending — meta
    rides along so BM25-only hits keep their provenance through RRF fusion.
    Empty list when the collection set has no documents.

    Zero-score documents are deliberately *not* filtered out: rank_bm25's IDF
    is ``ln((N - df + 0.5) / (df + 0.5))``, which is exactly 0 for a term
    appearing in half the corpus (and epsilon-floored when more common), so a
    genuinely matching document can score 0 in a small collection. Filtering
    on ``score > 0`` drops those real keyword hits.
    """
    result = await _get_or_build_index(collection_names)
    if result is None:
        return []

    bm25, texts, metas = result
    scores = bm25.get_scores(_tokenize(query))

    scored = list(zip(texts, scores, metas, strict=True))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def reciprocal_rank_fusion(
    vector_texts: list[str],
    bm25_texts: list[str],
    bm25_weight: float,
    k_rrf: int = 60,
) -> list[tuple[str, float]]:
    """Fuse two rank-ordered text lists with Reciprocal Rank Fusion.

    Returns ``(text, fused_score)`` sorted by fused score descending.
    """
    scores: dict[str, float] = {}
    for texts, weight in ((vector_texts, 1.0 - bm25_weight), (bm25_texts, bm25_weight)):
        for rank, text in enumerate(texts):
            scores[text] = scores.get(text, 0.0) + weight / (k_rrf + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
