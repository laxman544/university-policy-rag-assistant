import numpy as np


class Retriever:
    """
    Retrieves top-k relevant chunks from FAISS index.
    """

    def __init__(self, index, metadata, embedder):
        self.index = index
        self.metadata = metadata
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5):
        """
        Return top-k retrieved chunks for a user query.
        """
        query_embedding = self.embedder.encode_texts([query])
        distances, indices = self.index.search(
            np.asarray(query_embedding, dtype="float32"),
            top_k
        )

        results = []
        for rank, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.metadata):
                record = self.metadata[idx]
                results.append({
                    "rank": rank + 1,
                    "chunk_id": record["chunk_id"],
                    "text": record["text"],
                    "distance": float(distances[0][rank])
                })

        return results
