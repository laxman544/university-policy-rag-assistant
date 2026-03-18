import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Creates embeddings using a SentenceTransformer model.
    """

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts):
        """
        Convert list of texts into float32 numpy embeddings.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return np.asarray(embeddings, dtype="float32")
