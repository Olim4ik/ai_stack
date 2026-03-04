"""Global error handler middleware."""

import structlog
from fastapi import Request
from fastapi.responses import JSONResponse

logger = structlog.get_logger()


async def global_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
