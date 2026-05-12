"""Qdrant client + vector search.

Phase 3: reads the Haystack-native schema written by the ingestion service —
``payload.content`` for chunk text and ``payload.meta`` for metadata. Filters
by ``meta.collection_name`` (a ``MatchAny`` over the request's logical
collection names). All retrievable data lives in a single physical collection
configured via ``QDRANT_INDEX``; the legacy multitenancy mapping that routed
``file-*``, ``user-memory-*`` etc. to per-class physical collections is gone.

When the configured collection has a sparse named vector (``text-sparse``),
``vector_search`` issues a hybrid Qdrant Query API call with prefetch + RRF
fusion. Otherwise it falls back to a dense-only query. The pipeline / agent
layer adds client-side BM25 RRF on top when hybrid is enabled but the
collection lacks sparse vectors.
"""

import logging
from dataclasses import dataclass

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings

log = logging.getLogger(__name__)

# Named-vector identifiers used by the ingestion-service-written collection.
# qdrant-haystack's QdrantDocumentStore stores the dense vector under
# "text-dense" and (when use_sparse_embeddings=True) the sparse vector under
# "text-sparse".
DENSE_VECTOR_NAME = "text-dense"
SPARSE_VECTOR_NAME = "text-sparse"


@dataclass
class QdrantResult:
    texts: list[str]
    metadatas: list[dict]
    distances: list[float]


# ---------------------------------------------------------------------------
# Client lifecycle
# ---------------------------------------------------------------------------

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=settings.qdrant_uri,
            api_key=settings.qdrant_api_key,
        )
    return _client


def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------

_sparse_capable: bool | None = None


def collection_exists(collection_name: str | None = None) -> bool:
    """True iff the configured Qdrant collection currently exists."""
    name = collection_name or settings.qdrant_index
    try:
        get_client().get_collection(name)
        return True
    except (UnexpectedResponse, ValueError):
        return False


def has_sparse_vectors() -> bool:
    """True iff the configured collection carries a sparse named vector.

    Cached for the process lifetime once a definite answer is obtained; call
    :func:`reset_sparse_capability` after a collection recreation. If the
    collection doesn't yet exist (cold start before any ingest), returns
    ``False`` *without caching* — the next call after the collection is
    created will re-detect.
    """
    global _sparse_capable
    if _sparse_capable is not None:
        return _sparse_capable

    try:
        info = get_client().get_collection(settings.qdrant_index)
    except (UnexpectedResponse, ValueError):
        return False

    sparse_cfg = getattr(info.config.params, "sparse_vectors", None)
    _sparse_capable = bool(sparse_cfg)
    return _sparse_capable


def reset_sparse_capability() -> None:
    """Force the next :func:`has_sparse_vectors` call to re-query Qdrant."""
    global _sparse_capable
    _sparse_capable = None


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def _collection_name_filter(collection_names: list[str]) -> models.Filter:
    """Filter that scopes results to the given logical collection names.

    Matches the ``meta.collection_name`` payload field written by the
    ingestion service. With Qdrant's multitenancy HNSW config
    (``m=0, payload_m=16``) and the ``is_tenant=True`` payload index on
    that field, this hits per-tenant subgraphs cleanly.
    """
    return models.Filter(
        must=[
            models.FieldCondition(
                key="meta.collection_name",
                match=models.MatchAny(any=collection_names),
            )
        ]
    )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


async def vector_search(
    collection_names: list[str],
    query_vector: list[float],
    sparse_vector: models.SparseVector | None,
    k: int,
) -> QdrantResult:
    """Query the configured Qdrant collection.

    * Single physical Qdrant collection (``settings.qdrant_index``).
    * Filter scoped to ``meta.collection_name IN (collection_names)``.
    * Hybrid (dense + sparse, RRF-fused server-side) when ``sparse_vector`` is
      provided AND the collection has sparse vectors. Dense-only fallback
      otherwise.
    * Returns ``QdrantResult``. Distances are normalized cosine for the dense
      path (``(score + 1) / 2``) and raw RRF scores for the hybrid path.
    """
    client = get_client()
    qdrant_collection = settings.qdrant_index
    qfilter = _collection_name_filter(collection_names)

    use_hybrid = sparse_vector is not None and has_sparse_vectors()

    log.info(
        "vector_search: collections=%s (qdrant=%s, hybrid=%s) k=%d",
        collection_names,
        qdrant_collection,
        use_hybrid,
        k,
    )

    try:
        if use_hybrid:
            response = client.query_points(
                collection_name=qdrant_collection,
                prefetch=[
                    models.Prefetch(
                        query=query_vector,
                        using=DENSE_VECTOR_NAME,
                        limit=k * 2,
                        filter=qfilter,
                    ),
                    models.Prefetch(
                        query=sparse_vector,
                        using=SPARSE_VECTOR_NAME,
                        limit=k * 2,
                        filter=qfilter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=k,
            )
        else:
            response = client.query_points(
                collection_name=qdrant_collection,
                query=query_vector,
                using=DENSE_VECTOR_NAME,
                query_filter=qfilter,
                limit=k,
            )
    except UnexpectedResponse:
        log.exception("Qdrant query failed for collection %s", qdrant_collection)
        return QdrantResult(texts=[], metadatas=[], distances=[])

    texts: list[str] = []
    metadatas: list[dict] = []
    distances: list[float] = []
    for point in response.points:
        payload = point.payload or {}
        texts.append(payload.get("content", ""))
        metadatas.append(payload.get("meta", {}))
        score = point.score
        # Dense path: cosine in [-1, 1] → normalize to [0, 1].
        # Hybrid path: RRF score is already a positive rank-fusion score.
        distances.append(score if use_hybrid else (score + 1.0) / 2.0)

    return QdrantResult(texts=texts, metadatas=metadatas, distances=distances)


# ---------------------------------------------------------------------------
# Scroll (used by BM25 fallback)
# ---------------------------------------------------------------------------


def scroll_collection_texts(
    collection_names: list[str],
) -> list[tuple[str, str]]:
    """Scroll all documents matching ``meta.collection_name IN (collection_names)``.

    Returns a list of ``(point_id, text)``. Used by the BM25 fallback path to
    build an in-memory inverted index over the relevant subset of the physical
    collection — never the whole index.
    """
    client = get_client()
    qdrant_collection = settings.qdrant_index
    sfilter = _collection_name_filter(collection_names)

    results: list[tuple[str, str]] = []
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=qdrant_collection,
            scroll_filter=sfilter,
            limit=100,
            offset=offset,
            with_payload=True,
        )
        for point in points:
            payload = point.payload or {}
            text = payload.get("content", "")
            if text:
                results.append((str(point.id), text))
        if next_offset is None:
            break
        offset = next_offset

    return results
