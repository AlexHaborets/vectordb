import os
from pathlib import Path
import numba
import numpy as np
from dotenv import load_dotenv

load_dotenv()

NUMBA_THREADING_LAYER = os.getenv("NUMBA_THREADING_LAYER", "tbb")

try:
    numba.config.THREADING_LAYER = NUMBA_THREADING_LAYER # type: ignore
except ImportError:
    pass

"""
DB settings
"""

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DB_FILE = ROOT_DIR / "data" / "vector_db.sqlite"
DATABASE_URL = f"sqlite:///{DB_FILE}"

INDEX_RND_SAMPLE_SIZE = 512

"""
#################
Indexing settings
#################
"""

# Search list size when building the index
VAMANA_L_BUILD = 80
# Search list size during quering
VAMANA_L_SEARCH = 60

VAMANA_R = 32

VAMANA_ALPHA_FIRST_PASS = 1.0

VAMANA_ALPHA_SECOND_PASS = 1.2

# Decimal places to round the similarity score
# between the query and vectors-results from search
SIMILARITY_SCORE_PRECISION = 4

NUMPY_DTYPE = np.float32

"""
API settings
"""

BATCH_SIZE_LIMIT = 2048

MAX_ID_LENGTH = 64

MAX_META_SIZE = 50 # keys

MAX_DIMENSIONS = 4096 

MIN_DIMENSIONS = 2 

MAX_COLLECTION_NAME_LENGTH = 63

DB_PORT = 8000

