from sklearn.cluster import MiniBatchKMeans
import numpy as np

class MiniBatchKMeansClustering:
    def __init__(self, num_clusters=8, batch_size=100, init_size=300):
        self.kmeans = MiniBatchKMeans(
            n_clusters=num_clusters,
            batch_size=batch_size,
            init_size=init_size,
            random_state=42
        )
        self.is_fitted = False
        self.vectors_buffer = []
        self.vector_to_cluster = {}

    def initial_fit(self, initial_vectors):
        """
        Initial fitting with a batch of vectors.
        """
        if len(initial_vectors) > 0:
            self.kmeans.fit(initial_vectors)
            self.is_fitted = True
            return self.kmeans.labels_
        return []
    
    def partial_fit(self, new_vectors, vector_ids=None):
        """
        Update the model with new vectors.
        """
        if len(new_vectors) == 0:
            return []

        if not self.is_fitted:
            return self.initial_fit(new_vectors)
    
        self.kmeans.partial_fit(new_vectors)
    
        labels = self.kmeans.predict(new_vectors)
    
        if vector_ids is not None:
            for i, vector_id in enumerate(vector_ids):
                self.vector_to_cluster[vector_id] = labels[i]

        return labels
    
    def add_to_buffer(self, vector, vector_id=None):
        """Add a vector to the buffer for later batch processing."""
        self.vectors_buffer.append((vector, vector_id))
    
    def process_buffer(self, min_batch_size=10):
        """Process the buffer when it reaches sufficient size."""
        if len(self.vectors_buffer) >= min_batch_size:
            vectors = [v[0] for v in self.vectors_buffer]
            ids = [v[1] for v in self.vectors_buffer]

            vectors_array = np.array(vectors)

            self.partial_fit(vectors_array, ids)

            self.vectors_buffer = []
            return True
        return False
    
    def get_temperature_adjustment(self, query_embedding):
        """
        Calculate temperature adjustment based on cluster proximity.
        
        Returns a value between 0.0 and 1.0 that can be used to 
        adjust the base temperature.
        """
        if not self.is_fitted:
            return 1.0  # High temperature for empty/unfitted clusterer
        
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        cluster_idx = self.kmeans.predict(query_embedding)[0]
        
        centroid = self.kmeans.cluster_centers_[cluster_idx]
        distance = np.linalg.norm(query_embedding - centroid)
        
        cluster_vectors = [v for v_id, v in self.vector_to_cluster.items() 
                          if v == cluster_idx]
        
        if cluster_vectors:
            avg_distance = np.mean([np.linalg.norm(v - centroid) 
                                   for v in cluster_vectors])
            relative_distance = distance / max(avg_distance, 1e-6)
        else:
            relative_distance = distance
        
        temp_adjustment = min(1.0, relative_distance)
        
        return temp_adjustment
    
    def get_cluster_stats(self):
        """
        Get statistics about clusters to inform caching decisions.
        """
        if not self.is_fitted:
            return None
            
        stats = {
            "num_clusters": self.kmeans.n_clusters,
            "cluster_sizes": {},
            "cluster_densities": {},
        }
    
        # Count vectors per cluster
        for cluster_idx in range(self.kmeans.n_clusters):
            cluster_vectors = [v_id for v_id, c in self.vector_to_cluster.items() 
                              if c == cluster_idx]
            stats["cluster_sizes"][cluster_idx] = len(cluster_vectors)
    
        # Calculate inertia per cluster (average distance to centroid)
        # This requires additional logic to compute
    
        return stats