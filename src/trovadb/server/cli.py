import uvicorn

from trovadb.server.common.config import DB_PORT


def main():
    uvicorn.run("trovadb.server.main:app", host="0.0.0.0", port=DB_PORT, reload=False)


if __name__ == "__main__":
    main()
