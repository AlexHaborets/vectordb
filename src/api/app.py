from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield   

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return JSONResponse(
        status_code=200, content={"message": "Vector DB is running"}
    )

@app.post("/add")
async def add():
    pass

@app.post("/search")
async def search():
    pass