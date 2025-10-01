from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from models.vector import Vector, VectorCreate 
from models.search_result import SearchResult

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return JSONResponse(
        status_code=200, content={"message": "Vector DB is running"}
    )

@app.post("/add", response_model=Vector)
async def add(vector: VectorCreate):
    pass

@app.post("/search", response_model=SearchResult)
async def search():
    pass

