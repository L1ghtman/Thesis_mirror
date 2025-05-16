from sklearn.cluster import MiniBatchKMeans
import numpy as np

class MiniBatchKMeansClustering:
    def __init__(self, max_clusters=8, batch_size=100, init_size=300):
        """
        Initialize with a maximum number of clusters.
        
        Args:
            max_clusters: Maximum number of clusters to use
            batch_size: Batch size for MiniBatchKMeans
            init_size: Initial size for MiniBatchKMeans
        """
        self.max_clusters = max_clusters
        self.batch_size = batch_size
        self.init_size = init_size
        self.kmeans = None
        self.is_fitted = False
        self.vectors_buffer = []
        self.vector_to_cluster = {}
        self.all_vectors = []  # Store all vectors for potential refit

    def initial_fit(self, initial_vectors):
        """
        Initial fitting with a batch of vectors.
        Adapts the number of clusters to the data size.
        """
        if len(initial_vectors) == 0:
            print("Warning: No vectors provided for initial fit")
            return []
            
        # Store these vectors for later use
        self.all_vectors.extend(initial_vectors)
        
        # Determine appropriate number of clusters based on data size
        n_samples = len(initial_vectors)
        n_clusters = min(self.max_clusters, n_samples)
        
        print(f"Initial fit with {n_samples} samples, using {n_clusters} clusters")
        
        # Create a new KMeans with appropriate number of clusters
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=min(self.batch_size, n_samples),
            init_size=min(self.init_size, n_samples),
            random_state=42
        )
        
        # Fit the model
        self.kmeans.fit(initial_vectors)
        self.is_fitted = True
        
        # Get and return the labels
        labels = self.kmeans.labels_
        
        # Store cluster assignments
        for i in range(len(labels)):
            vector_id = i  # Use position as ID if not explicitly provided
            self.vector_to_cluster[vector_id] = labels[i]
            
        return labels
    
    def partial_fit(self, new_vectors, vector_ids=None):
        """
        Update the model with new vectors.
        """
        if len(new_vectors) == 0:
            print("Warning: No vectors provided for partial fit")
            return []

        # Store these vectors
        self.all_vectors.extend(new_vectors)

        # If not fitted yet, we need to do initial fit
        if not self.is_fitted:
            return self.initial_fit(new_vectors)
        
        # Check if we need to increase the number of clusters
        current_n_clusters = self.kmeans.n_clusters
        total_vectors = len(self.all_vectors)
        ideal_n_clusters = min(self.max_clusters, total_vectors // 10 + 1)  # 1 cluster per 10 vectors, at least 1
        
        # If we have enough data for more clusters and less than max, refit with more clusters
        if ideal_n_clusters > current_n_clusters and ideal_n_clusters <= self.max_clusters:
            print(f"Refitting model with {ideal_n_clusters} clusters (up from {current_n_clusters})")
            
            # Create a new model with more clusters
            self.kmeans = MiniBatchKMeans(
                n_clusters=ideal_n_clusters,
                batch_size=min(self.batch_size, total_vectors),
                init_size=min(self.init_size, total_vectors),
                random_state=42
            )
            
            # Fit on all data we've seen
            self.kmeans.fit(np.array(self.all_vectors))
            
            # Update all cluster assignments
            self.vector_to_cluster = {}
            labels = self.kmeans.predict(np.array(self.all_vectors))
            for i, label in enumerate(labels):
                self.vector_to_cluster[i] = label
                
            if vector_ids is not None:
                # Also update the specific assignments for the current batch
                new_labels = self.kmeans.predict(new_vectors)
                for i, vector_id in enumerate(vector_ids):
                    self.vector_to_cluster[vector_id] = new_labels[i]
                    
            return labels
        
        # Regular partial fit with existing model
        self.kmeans.partial_fit(new_vectors)
    
        # Get predictions for the new vectors
        labels = self.kmeans.predict(new_vectors)
    
        # Update cluster assignments if IDs are provided
        if vector_ids is not None:
            for i, vector_id in enumerate(vector_ids):
                self.vector_to_cluster[vector_id] = labels[i]

        return labels
    
    def add_to_buffer(self, vector, vector_id=None):
        """Add a vector to the buffer for later batch processing."""
        if vector_id is None:
            vector_id = len(self.all_vectors) + len(self.vectors_buffer)
        self.vectors_buffer.append((vector, vector_id))
    
    def process_buffer(self, min_batch_size=10):
        """Process the buffer when it reaches sufficient size."""
        if len(self.vectors_buffer) >= min_batch_size:
            vectors = [v[0] for v in self.vectors_buffer]
            ids = [v[1] for v in self.vectors_buffer]

            vectors_array = np.array(vectors)
            
            try:
                self.partial_fit(vectors_array, ids)
                self.vectors_buffer = []
                return True
            except Exception as e:
                print(f"Error processing buffer: {e}")
                return False
        return False
    
    def force_process_buffer(self):
        """Force processing of the buffer regardless of its size."""
        if len(self.vectors_buffer) > 0:
            vectors = [v[0] for v in self.vectors_buffer]
            ids = [v[1] for v in self.vectors_buffer]
            
            vectors_array = np.array(vectors)
            
            try:
                self.partial_fit(vectors_array, ids)
                processed_count = len(self.vectors_buffer)
                self.vectors_buffer = []
                return processed_count
            except Exception as e:
                print(f"Error force processing buffer: {e}")
                return 0
        return 0
    
    def get_temperature_adjustment(self, query_embedding):
        """
        Calculate temperature adjustment based on cluster proximity.
        
        Returns a value between 0.0 and 1.0 that can be used to 
        adjust the base temperature.
        """
        if not self.is_fitted or self.kmeans is None:
            print("Warning: Clusterer not fitted, returning default temp adjustment")
            return 1.0  # High temperature for empty/unfitted clusterer
        
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        # Get cluster assignment and distance to centroid
        try:
            cluster_idx = self.kmeans.predict(query_embedding)[0]
            centroid = self.kmeans.cluster_centers_[cluster_idx]
            distance = np.linalg.norm(query_embedding - centroid)
            
            # Get all vectors in this cluster
            cluster_vectors = [v_id for v_id, v in self.vector_to_cluster.items() 
                             if v == cluster_idx]
            
            # Calculate average distance or use distance directly
            if cluster_vectors:
                # Here we'd ideally compute average distance of all points in cluster to centroid
                # But since we don't store all vectors by ID, we'll use a proxy
                
                # Use inertia (average squared distance) as a proxy
                # For better accuracy, we could store all vectors along with IDs
                relative_distance = distance / (np.sqrt(self.kmeans.inertia_ / len(self.all_vectors)) + 1e-6)
            else:
                relative_distance = distance
            
            # Scale to appropriate range (0.0 to 1.0)
            temp_adjustment = min(1.0, relative_distance / 2.0)
            
            return temp_adjustment
            
        except Exception as e:
            print(f"Error calculating temperature adjustment: {e}")
            return 1.0  # Fall back to high temperature
    
    def get_cluster_stats(self):
        """
        Get statistics about clusters to inform caching decisions.
        """
        if not self.is_fitted or self.kmeans is None:
            print("Clusterer is not fitted yet - no stats available")
            return None
            
        stats = {
            "num_clusters": self.kmeans.n_clusters,
            "total_points": len(self.vector_to_cluster),
            "cluster_sizes": {},
            "cluster_ids": list(range(self.kmeans.n_clusters)),
            "clusters_with_data": set(),
            "total_vectors_seen": len(self.all_vectors),
        }
    
        # Count vectors per cluster
        for cluster_idx in range(self.kmeans.n_clusters):
            cluster_vectors = [v_id for v_id, c in self.vector_to_cluster.items() 
                              if c == cluster_idx]
            stats["cluster_sizes"][cluster_idx] = len(cluster_vectors)
            if len(cluster_vectors) > 0:
                stats["clusters_with_data"].add(cluster_idx)
        
        stats["clusters_with_data"] = list(stats["clusters_with_data"])
        stats["empty_clusters"] = self.kmeans.n_clusters - len(stats["clusters_with_data"])
        stats["average_points_per_cluster"] = (
            stats["total_points"] / len(stats["clusters_with_data"]) 
            if stats["clusters_with_data"] else 0
        )
        
        if hasattr(self.kmeans, 'inertia_'):
            stats["inertia"] = self.kmeans.inertia_
            stats["avg_distance"] = np.sqrt(self.kmeans.inertia_ / len(self.all_vectors)) if self.all_vectors else 0
    
        return stats