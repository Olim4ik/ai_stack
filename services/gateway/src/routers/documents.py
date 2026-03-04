from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request
from typing import Annotated

from ..models.responses import IngestResponse

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    request: Request,
    file: Annotated[UploadFile, File()],
    team: Annotated[str, Form()] = "platform",
    tags: Annotated[str, Form()] = "",
):
    """Upload and ingest a document into the knowledge base."""
    pipeline = request.app.state.ingestion_pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Ingestion pipeline not available")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    try:
        result = await pipeline.ingest(
            filename=file.filename or "unknown",
            content=content,
            team=team,
            tags=tag_list,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return IngestResponse(
        doc_id=result["doc_id"],
        title=result["title"],
        chunks_created=result["chunks_created"],
        message=f"Document ingested successfully into team_{team}",
    )


@router.get("")
async def list_documents(
    request: Request,
    team: str | None = None,
    page: int = 1,
    page_size: int = 20,
):
    """List ingested documents. Queries Qdrant for unique doc_ids."""
    # Stub — will be enhanced with proper document registry
    return {
        "documents": [],
        "total": 0,
        "page": page,
        "page_size": page_size,
    }


@router.delete("/{doc_id}")
async def delete_document(request: Request, doc_id: str, team: str = "platform"):
    """Delete a document and all its chunks from Qdrant."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    qdrant = request.app.state.qdrant
    if not qdrant:
        raise HTTPException(status_code=503, detail="Qdrant not available")

    collection = f"team_{team}"
    await qdrant.delete(
        collection_name=collection,
        points_selector=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
    )

    return {"message": f"Document {doc_id} deleted from {collection}"}
