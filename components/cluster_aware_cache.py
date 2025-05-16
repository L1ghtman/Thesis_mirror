from gptcache.core import Cache

# Extend the Cache class to add cluster tracking functionality
class ClusterAwareCache(Cache):
    """
    An extension of the GPTCache Cache class that tracks cluster information
    for better analytics.
    """
    
    def __init__(self):
        super().__init__()
        self.last_context = {}
        self.cluster_stats = {}
        
    def track_cluster(self, cluster_id, query=None, is_hit=False):
        """
        Track a cluster assignment for analytics.
        
        Args:
            cluster_id: The cluster ID assigned to this query
            query: The query text (optional)
            is_hit: Whether this query resulted in a cache hit
        """
        if cluster_id is None:
            return
            
        # Initialize cluster tracking if needed
        if cluster_id not in self.cluster_stats:
            self.cluster_stats[cluster_id] = {
                "total_queries": 0,
                "hits": 0,
                "queries": []
            }
            
        # Update stats
        self.cluster_stats[cluster_id]["total_queries"] += 1
        if is_hit:
            self.cluster_stats[cluster_id]["hits"] += 1
            
        # Store query if provided
        if query:
            if len(self.cluster_stats[cluster_id]["queries"]) < 10:  # Keep only last 10 for memory efficiency
                self.cluster_stats[cluster_id]["queries"].append(query)
            else:
                self.cluster_stats[cluster_id]["queries"].pop(0)
                self.cluster_stats[cluster_id]["queries"].append(query)
                
        # Update last context
        self.last_context["cluster_id"] = cluster_id
        
    def get_cluster_stats(self):
        """
        Get statistics about all tracked clusters.
        
        Returns:
            Dict containing cluster statistics
        """
        stats = {
            "clusters": list(self.cluster_stats.keys()),
            "total_clusters": len(self.cluster_stats),
            "by_cluster": self.cluster_stats
        }
        
        # Calculate overall hit rate per cluster
        for cluster_id, cluster_data in self.cluster_stats.items():
            if cluster_data["total_queries"] > 0:
                cluster_data["hit_rate"] = cluster_data["hits"] / cluster_data["total_queries"]
            else:
                cluster_data["hit_rate"] = 0
                
        return stats