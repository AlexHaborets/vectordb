from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DB_FILE = ROOT_DIR / "data" / "vector_db.sqlite"
DATABASE_URL = f"sqlite:///{DB_FILE}"

INDEX_RND_SAMPLE_SIZE = 512

# Search list size when building the index
VAMANA_L_BUILD = 64
# Search list size during quering
VAMANA_L_SEARCH = 64

VAMANA_R = 32

# Decimal places to round the similarity score
# between the query and vectors-results from search
SIMILARITY_SCORE_PRECISION = 4

NUMPY_DTYPE = np.float32

BATCH_SIZE_LIMIT = 256
