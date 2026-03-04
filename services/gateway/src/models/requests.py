from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str
    message: str
    team: str = "platform"


class ChatConfirmRequest(BaseModel):
    action_id: str
    approved: bool


class DocumentListParams(BaseModel):
    team: str | None = None
    tags: list[str] | None = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
