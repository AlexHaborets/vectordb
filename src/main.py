import uvicorn
import config

if __name__ == "__main__":
    uvicorn.run(app="api.app:app", host="0.0.0.0", reload=True, port=config.PORT)