from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from src.api import collection_router, search_router, vector_router
from src.api.dependencies import get_indexer_manager
from src.api.exception_handlers import create_exception_handler
from src.common import (
    DuplicateEntityError,
    EntityNotFoundError,
    InvalidOperationError,
    setup_logger,
)
from src.common.config import DB_PORT
from src.db import session_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting VectorDB...")
    setup_logger()

    logger.info("VectorDB started successfully")

    yield

    logger.info("Shutting down VectorDB...")

    indexer_manager = get_indexer_manager()
    indexer_manager.stop()

    logger.info("Closing database connections...")
    session_manager.close()


app = FastAPI(title="VectorDB", lifespan=lifespan)

app.include_router(vector_router)
app.include_router(search_router)
app.include_router(collection_router)

app.add_exception_handler(
    EntityNotFoundError, create_exception_handler(status_code=404)
)

app.add_exception_handler(
    DuplicateEntityError, create_exception_handler(status_code=409)
)

app.add_exception_handler(
    InvalidOperationError, create_exception_handler(status_code=400)
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "VectorDB is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=DB_PORT, reload=True)
