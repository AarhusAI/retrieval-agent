import logging

from fastapi import APIRouter, Depends

from app.auth import verify_api_key
from app.models import SearchRequest, SearchResponse
from app.services.pipeline import search

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_endpoint(
    request: SearchRequest,
    _api_key: str = Depends(verify_api_key),
) -> SearchResponse:
    log.info(
        "Search request: queries=%s, messages=%s, collections=%s, k=%d",
        request.queries,
        request.messages,
        request.collection_names,
        request.k,
    )
    return await search(request)
