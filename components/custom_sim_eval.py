from typing import Tuple
import numpy as np

class CustomSimilarityEvaluation:
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def __call__(self, query_embedding: Tuple[float, ...], candidate_embedding: Tuple[float, ...]) -> float:
        print("\033[91m"+"Custom sim eval called!"+"\033[0m")
        # Convert embeddings to numpy arrays
        query = np.array(query_embedding, dtype=float)
        candidate = np.array(candidate_embedding, dtype=float)
        # Compute cosine similarity
        dot = np.dot(query, candidate)
        norm_q = np.linalg.norm(query)
        norm_c = np.linalg.norm(candidate)
        if norm_q == 0 or norm_c == 0:
            return 0.0
        similarity = dot / (norm_q * norm_c)
        return similarity

    def is_similar(self, similarity: float) -> bool:
        return similarity >= self.threshold