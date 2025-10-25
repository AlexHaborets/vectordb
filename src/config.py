from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_FILE = ROOT_DIR / "data" / "vector_db.sqlite"
DATABASE_URL = f"sqlite:///{DB_FILE}"

VECTOR_DIMENSIONS = 4

INDEX_RND_SAMPLE_SIZE = 1000

VAMANA_L = 3
VAMANA_R = 2
VAMANA_ALPHA = 1.2 

NUMPY_DTYPE = np.float32