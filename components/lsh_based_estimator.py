import numpy as np
from collections import defaultdict, deque
from typing import Tuple, List

class LSHEstimator:
    def __init__(self, embedding_dim=384, num_hyperplanes=8, window_size=1000):
        # LSH components
        self.num_hyperplanes = num_hyperplanes
        self.hyperplanes = np.random.randn(num_hyperplanes, embedding_dim)
        self.hyperplanes /= np.linalg.norm(self.hyperplanes, axis=1, keepdims=True)
        self.bucket_counts = defaultdict(int)
        self.bucket_history = deque(maxlen=window_size)
        self.total_count = 0
        
    def get_lsh_bucket(self, embedding: np.ndarray) -> str:
        """Hash embedding to bucket using hyperplanes"""
        projections = np.dot(self.hyperplanes, embedding)
        binary = ''.join('1' if p > 0 else '0' for p in projections)
        return binary
    
    def estimate_density(self, embedding: np.ndarray) -> Tuple[float, dict]:
        """
        Estimate topic density using LSH
        Returns: (temperature, debug_info)
        """
        bucket = self.get_lsh_bucket(embedding)
        bucket_density = self._calculate_bucket_density(bucket)
        density = 0.7 * bucket_density
        self._update_stats(bucket)
        base_temp = 0.8
        cache_factor = 1.0 / np.log2(self.total_count + 2)
        temperature = base_temp * (1 - density * 0.7) * cache_factor
        temperature = max(0.1, min(2.0, temperature))
        
        debug_info = {
            "lsh_bucket": bucket,
            "bucket_count": self.bucket_counts[bucket],
            "bucket_density": round(bucket_density, 3),
            "density": round(density, 3),
            "temperature": round(temperature, 3)
        }
        
        return temperature, debug_info
    
    def _calculate_bucket_density(self, bucket: str) -> float:
        """Calculate density for a bucket and its neighbors"""
        if self.total_count == 0:
            return 0.0
            
        main_count = self.bucket_counts[bucket]
        
        neighbor_count = 0
        for i in range(len(bucket)):
            neighbor = list(bucket)
            neighbor[i] = '1' if bucket[i] == '0' else '0'
            neighbor_key = ''.join(neighbor)
            neighbor_count += self.bucket_counts.get(neighbor_key, 0)
        
        relevant_count = main_count + neighbor_count * 0.5  # Neighbors weighted less
        
        density = min(relevant_count / (self.total_count * 0.1), 1.0)
        return density
    
    def _update_stats(self, bucket: str):
        """Update all statistics"""
        self.bucket_history.append(bucket)
        self.bucket_counts[bucket] += 1
        self.total_count += 1


class LSHCache:
    def __init__(self, embedding_dim=384, num_hyperplanes=8, window_size=1000):
        self.estimator = LSHEstimator(embedding_dim, num_hyperplanes, window_size)
    
    def estimate_temperature(self, embedding: np.ndarray) -> Tuple[float, dict]:
        return self.estimator.estimate_density(embedding)
    
    def get_temperature(self, embedding: np.ndarray) -> float:
        temperature, _ = self.estimate_temperature(embedding)
        return temperature
    