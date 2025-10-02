from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.api.routers import vector_router, search_router
from src.db import session_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield   
    session_manager.close()


app = FastAPI(title="Simple VectorDB", lifespan=lifespan)

app.include_router(vector_router)
app.include_router(search_router)

@app.get("/")
async def root() -> JSONResponse:
    return {"message": "Vector DB is running"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app="src.main:app", host="0.0.0.0", reload=True, port="8000")
