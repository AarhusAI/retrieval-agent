from typing import Annotated

from pydantic import BaseModel, Field, model_validator

# Open WebUI collection names look like ``file-<uuid>``, ``user-memory-<uuid>``,
# or bare knowledge-collection UUIDs — all ASCII alphanumerics + ``_``/``-``.
# Anything else is either an injection attempt or a misconfigured upstream.
CollectionName = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")]
Query = Annotated[str, Field(max_length=2000)]


class ChatMessage(BaseModel):
    role: str = Field(max_length=32)
    content: str = Field(max_length=10_000)


class SearchRequest(BaseModel):
    queries: list[Query] | None = Field(default=None, max_length=10)
    messages: list[ChatMessage] | None = Field(default=None, max_length=50)
    collection_names: list[CollectionName] = Field(min_length=1, max_length=20)
    k: int = Field(default=5, ge=1, le=100)
    retrieval_query_generation_prompt_template: str | None = Field(default=None, max_length=8000)

    @model_validator(mode="after")
    def require_queries_or_messages(self):
        if not self.queries and not self.messages:
            raise ValueError("At least one of 'queries' or 'messages' must be provided")
        return self


class RetrievalResult(BaseModel):
    """One query's ranked hits as parallel lists."""

    texts: list[str]
    metadatas: list[dict]
    distances: list[float]

    @model_validator(mode="after")
    def require_parallel_lists(self):
        # interleave_dedup indexes all three lists by position — a mismatch must
        # fail here, not as an IndexError or silent truncation mid-merge.
        if not (len(self.texts) == len(self.metadatas) == len(self.distances)):
            raise ValueError(
                f"texts ({len(self.texts)}), metadatas ({len(self.metadatas)}) and "
                f"distances ({len(self.distances)}) must have the same length"
            )
        return self


class SearchResponse(BaseModel):
    documents: list[list[str]]
    metadatas: list[list[dict]]
    distances: list[list[float]]
