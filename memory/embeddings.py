"""Real local embeddings using sentence-transformers.

Runs entirely on-device — no API calls, no fake vectors.
Default model: all-MiniLM-L6-v2 (384-dim, ~80 MB).
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim: int = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string. Returns a normalized float32 vector."""
        return self.model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of strings. Returns (N, dim) normalized matrix."""
        return self.model.encode(texts, normalize_embeddings=True)
