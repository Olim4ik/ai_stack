from pydantic import BaseModel


class DocumentResponse(BaseModel):
    doc_id: str
    title: str
    source_file: str
    team: str
    tags: list[str]
    chunk_count: int
    ingested_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int
    page: int
    page_size: int


class HealthResponse(BaseModel):
    status: str
    services: dict[str, str] = {}


class IngestResponse(BaseModel):
    doc_id: str
    title: str
    chunks_created: int
    message: str
