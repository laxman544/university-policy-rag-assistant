from typing import List, Dict


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    """
    Split text into overlapping word-based chunks.
    Returns a list of dictionaries.
    """
    words = text.split()
    chunks = []

    start = 0
    chunk_id = 1

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_string = " ".join(chunk_words)

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_string,
            "start_word": start,
            "end_word": min(end, len(words))
        })

        chunk_id += 1
        start += max(1, chunk_size - overlap)

    return chunks
