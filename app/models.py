import re

from pydantic import BaseModel, Field, field_validator, model_validator

# Open WebUI collection names look like ``file-<uuid>``, ``user-memory-<uuid>``,
# or bare knowledge-collection UUIDs — all ASCII alphanumerics + ``_``/``-``.
# Anything else is either an injection attempt or a misconfigured upstream.
_COLLECTION_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class ChatMessage(BaseModel):
    role: str = Field(max_length=32)
    content: str = Field(max_length=10_000)


class SearchRequest(BaseModel):
    queries: list[str] | None = Field(default=None, max_length=10)
    messages: list[ChatMessage] | None = Field(default=None, max_length=50)
    collection_names: list[str] = Field(min_length=1, max_length=20)
    k: int = Field(default=5, ge=1, le=100)
    retrieval_query_generation_prompt_template: str | None = Field(default=None, max_length=8000)

    @field_validator("queries")
    @classmethod
    def validate_query_length(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            for q in v:
                if len(q) > 2000:
                    raise ValueError("Individual query must not exceed 2000 characters")
        return v

    @field_validator("collection_names")
    @classmethod
    def validate_collection_names(cls, v: list[str]) -> list[str]:
        for name in v:
            if not _COLLECTION_NAME_RE.fullmatch(name):
                raise ValueError("collection_names entries must match [A-Za-z0-9_-]{1,128}")
        return v

    @model_validator(mode="after")
    def require_queries_or_messages(self):
        if not self.queries and not self.messages:
            raise ValueError("At least one of 'queries' or 'messages' must be provided")
        return self


class SearchResponse(BaseModel):
    documents: list[list[str]]
    metadatas: list[list[dict]]
    distances: list[list[float]]
