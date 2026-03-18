import json
from pathlib import Path

import faiss


def build_faiss_index(embeddings):
    """
    Build a FAISS L2 index from embeddings.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def save_faiss_index(index, index_path: Path) -> None:
    """
    Save FAISS index to disk.
    """
    faiss.write_index(index, str(index_path))


def load_faiss_index(index_path: Path):
    """
    Load FAISS index from disk.
    """
    return faiss.read_index(str(index_path))


def save_metadata(metadata, metadata_path: Path) -> None:
    """
    Save metadata to JSON file.
    """
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


def load_metadata(metadata_path: Path):
    """
    Load metadata from JSON file.
    """
    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)
