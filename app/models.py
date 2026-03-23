from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    role: str
    content: str


class SearchRequest(BaseModel):
    queries: list[str] | None = None
    messages: list[ChatMessage] | None = None
    collection_names: list[str]
    k: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def require_queries_or_messages(self):
        if not self.queries and not self.messages:
            raise ValueError("At least one of 'queries' or 'messages' must be provided")
        return self


class SearchResponse(BaseModel):
    documents: list[list[str]]
    metadatas: list[list[dict]]
    distances: list[list[float]]
