from typing import Callable, Optional
from fastapi import Request
from fastapi.responses import JSONResponse

def create_exception_handler(status_code: int, detail: Optional[str] = None) -> Callable:
    # A factory for creating exception handlers
    async def handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content={"detail": detail if detail else str(exc)},
        )
    return handler