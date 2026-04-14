from pydantic import BaseModel, Field, field_validator, model_validator


class ChatMessage(BaseModel):
    role: str
    content: str


class SearchRequest(BaseModel):
    queries: list[str] | None = Field(default=None, max_length=10)
    messages: list[ChatMessage] | None = None
    collection_names: list[str] = Field(max_length=20)
    k: int = Field(default=5, ge=1)
    retrieval_query_generation_prompt_template: str | None = None

    @field_validator("queries")
    @classmethod
    def validate_query_length(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            for q in v:
                if len(q) > 2000:
                    raise ValueError("Individual query must not exceed 2000 characters")
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
