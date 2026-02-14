import os
from pathlib import Path

# Project Root
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DB_PATH = DATA_DIR / "research_cache.db"
EMBEDDINGS_CACHE_PATH = DATA_DIR / "embeddings.npy"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)

# Model Config
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemini-1.5-flash"

# Search Config
MAX_SEARCH_RESULTS = 10
SIMILARITY_THRESHOLD = 0.5  # Cosine similarity threshold for relevance
MAX_DEPTH = 2  # How many hops from the original paper

# API Config
SEMANTIC_SCHOLAR_RATE_LIMIT = 1.0  # Seconds between requests (S2 public API allows ~100 req/5min)
