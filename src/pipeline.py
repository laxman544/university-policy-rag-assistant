import json
from pathlib import Path

from src.config import (
    PDF_PATH,
    PROCESSED_DIR,
    INDEX_DIR,
    CHUNKS_JSON_PATH,
    CHUNKS_JSONL_PATH,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    EMBEDDING_MODEL_NAME,
    GENERATION_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
)
from src.loader import extract_text_from_pdf
from src.chunker import chunk_text
from src.embedder import Embedder
from src.indexer import (
    build_faiss_index,
    save_faiss_index,
    load_faiss_index,
    save_metadata,
    load_metadata,
)
from src.retriever import Retriever
from src.generator import AnswerGenerator


def ensure_directories():
    """
    Create required directories if they do not exist.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def save_json(data, file_path: Path):
    """
    Save list/dict to JSON.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_jsonl(records, file_path: Path):
    """
    Save list of records to JSONL.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_assets():
    """
    Full preprocessing pipeline:
    PDF -> chunks -> embeddings -> FAISS index -> metadata
    """
    ensure_directories()

    print("Step 1: Extracting text from PDF...")
    text = extract_text_from_pdf(str(PDF_PATH))

    print("Step 2: Chunking text...")
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    print("Step 3: Saving chunk files...")
    save_json(chunks, CHUNKS_JSON_PATH)
    save_jsonl(chunks, CHUNKS_JSONL_PATH)

    print("Step 4: Generating embeddings...")
    embedder = Embedder(EMBEDDING_MODEL_NAME)
    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.encode_texts(chunk_texts)

    print("Step 5: Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("Step 6: Saving FAISS index and metadata...")
    save_faiss_index(index, FAISS_INDEX_PATH)
    save_metadata(chunks, METADATA_PATH)

    print("Pipeline assets created successfully.")


def run_query(query: str):
    """
    Load saved assets and answer a query.
    """
    print("Loading embedder, index, and metadata...")
    embedder = Embedder(EMBEDDING_MODEL_NAME)
    index = load_faiss_index(FAISS_INDEX_PATH)
    metadata = load_metadata(METADATA_PATH)

    print("Retrieving top chunks...")
    retriever = Retriever(index, metadata, embedder)
    retrieved_chunks = retriever.retrieve(query, top_k=TOP_K)

    print("\nTop Retrieved Chunks:")
    for item in retrieved_chunks:
        print("-" * 80)
        print(f"Rank: {item['rank']}")
        print(f"Chunk ID: {item['chunk_id']}")
        print(f"Distance: {item['distance']}")
        print(item["text"][:300])

    print("\nGenerating final answer...")
    generator = AnswerGenerator(GENERATION_MODEL_NAME)
    answer = generator.generate_answer(query, retrieved_chunks)

    print("\nFinal Answer:")
    print(answer)


if __name__ == "__main__":
    # Build assets first
    build_assets()

    # Example query
    sample_query = "What is the attendance policy?"
    run_query(sample_query)
