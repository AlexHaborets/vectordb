from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_FILE = ROOT_DIR / "data" / "vector_db.sqlite"
DATABASE_URL = f"sqlite:///{DB_FILE}"

VECTOR_DIMENSIONS = 64