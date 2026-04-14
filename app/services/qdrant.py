import logging
from dataclasses import dataclass

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings

log = logging.getLogger(__name__)

# Shared collection suffixes (mirrors Open WebUI qdrant_multitenancy.py)
_MEMORY_COLLECTION = "memories"
_FILE_COLLECTION = "files"
_WEB_SEARCH_COLLECTION = "web-search"
_HASH_BASED_COLLECTION = "hash-based"
_KNOWLEDGE_COLLECTION = "knowledge"


@dataclass
class QdrantResult:
    texts: list[str]
    metadatas: list[dict]
    distances: list[float]


def _shared_collection_name(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}"


def _get_collection_and_tenant_id(collection_name: str, prefix: str) -> tuple[str, str | None]:
    """
    Maps collection name to (qdrant_collection, tenant_id).
    Replicates Open WebUI's qdrant_multitenancy.py:104-139 exactly.
    """
    if not settings.qdrant_multitenancy:
        return f"{prefix}_{collection_name}", None

    tenant_id = collection_name

    if collection_name.startswith("user-memory-"):
        return _shared_collection_name(prefix, _MEMORY_COLLECTION), tenant_id
    elif collection_name.startswith("file-"):
        return _shared_collection_name(prefix, _FILE_COLLECTION), tenant_id
    elif collection_name.startswith("web-search-"):
        return _shared_collection_name(prefix, _WEB_SEARCH_COLLECTION), tenant_id
    elif len(collection_name) == 63 and all(c in "0123456789abcdef" for c in collection_name):
        return _shared_collection_name(prefix, _HASH_BASED_COLLECTION), tenant_id
    else:
        return _shared_collection_name(prefix, _KNOWLEDGE_COLLECTION), tenant_id


def _tenant_filter(tenant_id: str) -> models.FieldCondition:
    return models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))


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


async def vector_search(
    collection_name: str,
    query_vector: list[float],
    k: int,
) -> QdrantResult:
    """Search a collection by vector similarity. Returns normalized cosine distances."""
    client = get_client()
    prefix = settings.qdrant_collection_prefix
    qdrant_collection, tenant_id = _get_collection_and_tenant_id(collection_name, prefix)

    query_filter = None
    if tenant_id is not None:
        query_filter = models.Filter(must=[_tenant_filter(tenant_id)])

    log.info(
        "vector_search: collection=%s (qdrant=%s, tenant=%s) k=%d",
        collection_name,
        qdrant_collection,
        tenant_id,
        k,
    )

    try:
        response = client.query_points(
            collection_name=qdrant_collection,
            query=query_vector,
            query_filter=query_filter,
            limit=k,
        )
    except UnexpectedResponse:
        log.exception("Qdrant query failed for collection %s", qdrant_collection)
        return QdrantResult(texts=[], metadatas=[], distances=[])

    texts = []
    metadatas = []
    distances = []
    for point in response.points:
        payload = point.payload or {}
        texts.append(payload.get("text", ""))
        metadatas.append(payload.get("metadata", {}))
        # Cosine distance normalization: [-1, 1] → [0, 1]
        distances.append((point.score + 1.0) / 2.0)

    return QdrantResult(texts=texts, metadatas=metadatas, distances=distances)


def scroll_collection_texts(
    collection_name: str,
) -> list[tuple[str, str]]:
    """Scroll all documents from a collection. Returns list of (point_id, text)."""
    client = get_client()
    prefix = settings.qdrant_collection_prefix
    qdrant_collection, tenant_id = _get_collection_and_tenant_id(collection_name, prefix)

    scroll_filter = None
    if tenant_id is not None:
        scroll_filter = models.Filter(must=[_tenant_filter(tenant_id)])

    results: list[tuple[str, str]] = []
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=qdrant_collection,
            scroll_filter=scroll_filter,
            limit=100,
            offset=offset,
            with_payload=True,
        )
        for point in points:
            payload = point.payload or {}
            text = payload.get("text", "")
            if text:
                results.append((str(point.id), text))
        if next_offset is None:
            break
        offset = next_offset

    return results
