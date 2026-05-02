import os
from pathlib import Path

import numba
import numpy as np
from dotenv import load_dotenv

load_dotenv()

NUMBA_THREADING_LAYER = os.getenv("NUMBA_THREADING_LAYER", "tbb")

try:
    numba.config.THREADING_LAYER = NUMBA_THREADING_LAYER  # type: ignore
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
Indexing settings
THESE CAN BE MODIFIED
"""

VAMANA_L = 120

VAMANA_R = 70

VAMANA_TARGET_UTILIZATION: float = 0.85

# NOTE: Optimal alpha is within 1.0 to 2.0
# The database dynamically adjusts alpha
# to keep the average degree at VAMANA_TARGET_UTILIZATION * VAMANA_R
VAMANA_ALPHA_FIRST_PASS = 1.0

VAMANA_ALPHA_SECOND_PASS = 1.2

MAX_L_SEARCH = 1000

MIN_L_SEARCH = 60

# Decimal places to round the similarity score
# between the query and vectors-results from search
SIMILARITY_SCORE_PRECISION = 4

NUMPY_DTYPE = np.float32

# Saves updated indexes to disk every 5 seconds
PERSIST_PERIOD = 5  # seconds

"""
API settings
"""

BATCH_SIZE_LIMIT = 2048

MAX_ID_LENGTH = 64

MAX_META_SIZE = 50  # keys

MAX_DIMENSIONS = 4096

MIN_DIMENSIONS = 2

MAX_COLLECTION_NAME_LENGTH = 63

DB_PORT = 8000
