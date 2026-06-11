import logging

from fastapi import APIRouter, Depends

from app.auth import verify_api_key
from app.log_utils import sanitize_for_log
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
        "Search request: queries=%d, messages=%d, collections=%s, k=%d",
        len(request.queries) if request.queries else 0,
        len(request.messages) if request.messages else 0,
        sanitize_for_log(request.collection_names),
        request.k,
    )
    log.debug(
        "Search request payload: queries=%s, messages=%s",
        sanitize_for_log(request.queries),
        sanitize_for_log(request.messages),
    )
    return await search(request)
