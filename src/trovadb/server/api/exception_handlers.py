from typing import Callable, Optional
from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger

def log_exception(request: Request, status_code: int, exc: Exception) -> None:
    if status_code >= 500:
        logger.exception(f"Internal server error at {request.method} {request.url}")
    else: 
        logger.warning(f"Client error {status_code} at {request.url}: {exc}")

def create_exception_handler(status_code: int, detail: Optional[str] = None) -> Callable:
    # A factory for creating exception handlers
    async def handler(request: Request, exc: Exception) -> JSONResponse:
        log_exception(request, status_code, exc)
        return JSONResponse(
            status_code=status_code,
            content={"detail": detail if detail else str(exc)},
        )
    return handler