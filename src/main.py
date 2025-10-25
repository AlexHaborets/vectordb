from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.api.routers import vector_router, search_router
from src.core import VamanaIndexer
from src.db import session_manager
from src.db.crud import VectorDBRepository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s (%(name)s): %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.logger = logger

    indexer = VamanaIndexer()
    
    logger.info("Vector DB starting: loading index")
    with session_manager.get_session() as session:
        repo = VectorDBRepository(session=session)
        indexer.load_index(repo)
    app.state.indexer = indexer
    yield

    logger.info("Vector DB stopping: saving index")
    with session_manager.get_session() as db:
        repo = VectorDBRepository(db)
        app.state.indexer.save_index(repo=repo)

    session_manager.close()


app = FastAPI(title="Simple VectorDB", lifespan=lifespan)

app.include_router(vector_router)
app.include_router(search_router)

@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Vector DB is running"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app="src.main:app", host="0.0.0.0", reload=True, port=8000)
