from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = PROCESSED_DIR / "faiss_index"

# File paths
PDF_PATH = RAW_DIR / "student_handbook.pdf"
CHUNKS_JSON_PATH = PROCESSED_DIR / "handbook_chunks.json"
FAISS_INDEX_PATH = INDEX_DIR / "handbook.index"

# Model config
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "google/flan-t5-base"

# Chunk settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 5
