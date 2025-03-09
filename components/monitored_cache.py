import time
from typing import Any, Callable, Dict, Optional, List, Tuple
from gptcache.manager.scalar_data.base import CacheData
from gptcache.core import Cache, Config

# Import the logger
from components.cache_logger import CachePerformanceLogger

class MonitoredCache:
    """
    A wrapper around GPTCache that adds performance monitoring capabilities.
    Uses composition pattern to avoid inheritance issues.
    """
    
    def __init__(self, log_dir: str = "cache_logs"):
        """
        Initialize the monitored cache with performance logging.
        
        Args:
            log_dir: Directory where log files will be stored
        """
        # Create the actual Cache instance that we'll delegate to
        self.cache = Cache()
        self.logger = CachePerformanceLogger(log_dir=log_dir)
    
    @property
    def has_init(self):
        """Delegate has_init property to the wrapped cache."""
        return self.cache.has_init
    
    @property
    def data_manager(self):
        """Delegate data_manager property to the wrapped cache."""
        return self.cache.data_manager
    
    @property
    def similarity_evaluation(self):
        """Delegate similarity_evaluation property to the wrapped cache."""
        return self.cache.similarity_evaluation
    
    @property
    def config(self):
        """Delegate config property to the wrapped cache."""
        return self.cache.config
    
    @property
    def cache_policy_source(self):
        """Delegate cache_policy_source property to the wrapped cache."""
        return self.cache.cache_policy_source if hasattr(self.cache, 'cache_policy_source') else None
    
    def init(self, *args, **kwargs):
        """Initialize the underlying cache with the provided arguments."""
        result = self.cache.init(*args, **kwargs)
        
        # Log configuration for reference
        config_info = {
            "pre_embedding_func": kwargs.get("pre_embedding_func", "default"),
            "embedding_func": kwargs.get("embedding_func", "default"),
            "data_manager": str(kwargs.get("data_manager", "default")),
            "similarity_evaluation": str(kwargs.get("similarity_evaluation", "default")),
            "config": str(kwargs.get("config", {}))
        }
        
        self.logger.logger.info(f"Cache initialized with config: {config_info}")
        return result
    
    def get(self, data: Any, **kwargs) -> Tuple[bool, Any]:
        """
        Monitored version of the get method that logs performance metrics.
        
        Args:
            data: The data to search for in the cache
            kwargs: Additional arguments
        
        Returns:
            A tuple containing a boolean indicating cache hit and the cache data
        """
        # Extract metadata from kwargs
        temperature = kwargs.get("temperature")
        extra_metadata = kwargs.get("metadata", {})
        
        start_time = time.time()
        query = str(data)[:200]  # Truncate long queries for logging
        
        # Check if we should bypass cache due to temperature setting
        bypass_cache = kwargs.get("bypass_cache", False)
        if temperature is not None and temperature > 0.0:
            bypass_cache = True
        
        if bypass_cache:
            # Log direct LLM call
            self.logger.log_request(
                query=query,
                used_cache=False,
                response_time=0.0,  # Will be updated after LLM response
                temperature=temperature,
                metadata=extra_metadata
            )
            return False, None
            
        # Try to get from cache
        hit, cache_data = self.cache.get(data, **kwargs)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # For cache hit, we need to determine if it's a positive or negative hit
        # This would typically be determined by your similarity evaluator
        positive_hit = None
        similarity_score = None
        
        if hit:
            # Extract similarity score if available
            if hasattr(cache_data, "similarity_score"):
                similarity_score = cache_data.similarity_score
            
            # Determine if it's a positive hit (you might want to customize this logic)
            # Typically based on a threshold for the similarity score
            threshold = kwargs.get("similarity_threshold", 0.7)
            positive_hit = similarity_score is None or similarity_score >= threshold
        
        # Log the request
        self.logger.log_request(
            query=query,
            used_cache=True,
            cache_hit=hit,
            positive_hit=positive_hit,
            response_time=response_time,
            similarity_score=similarity_score,
            temperature=temperature,
            metadata=extra_metadata
        )
        
        return hit, cache_data
    
    def put(self, data: Any, cache_data: Any, **kwargs) -> Any:
        """
        Monitored version of the put method.
        
        Args:
            data: The query data
            cache_data: The data to cache
            kwargs: Additional arguments
        
        Returns:
            Result of the put operation
        """
        start_time = time.time()
        result = self.cache.put(data, cache_data, **kwargs)
        end_time = time.time()
        
        # Log cache update
        self.logger.logger.info(
            f"Cache updated - query: '{str(data)[:50]}...' time: {end_time - start_time:.4f}s"
        )
        
        return result
    
    def close(self):
        """Close the logger (GPTCache doesn't have a close method)."""
        self.logger.close()
        # Note: The underlying Cache object doesn't have a close method
        # We only need to close our logger
    
    # Add delegation methods for any other methods that Cache exposes
    def set_chat_cache(self, *args, **kwargs):
        return self.cache.set_chat_cache(*args, **kwargs)
    
    def set_access_policy(self, *args, **kwargs):
        return self.cache.set_access_policy(*args, **kwargs)
    
    def set_map_dict(self, *args, **kwargs):
        return self.cache.set_map_dict(*args, **kwargs)
    
    def flush(self):
        return self.cache.flush()
        
    # Additional method delegations required by the adapter
    def __getattr__(self, name):
        """Delegate any other attribute accesses to the wrapped cache object."""
        if hasattr(self.cache, name):
            return getattr(self.cache, name)
        raise AttributeError(f"Neither MonitoredCache nor its wrapped Cache has attribute '{name}'")
        
    def set_cache_policy_source(self, *args, **kwargs):
        if hasattr(self.cache, 'set_cache_policy_source'):
            return self.cache.set_cache_policy_source(*args, **kwargs)
            
    def set_pre_embedding_func(self, *args, **kwargs):
        if hasattr(self.cache, 'set_pre_embedding_func'):
            return self.cache.set_pre_embedding_func(*args, **kwargs)
            
    def set_embedding_func(self, *args, **kwargs):
        if hasattr(self.cache, 'set_embedding_func'):
            return self.cache.set_embedding_func(*args, **kwargs)


# Helper function for easy setup
def create_monitored_cache(log_dir: str = "cache_logs", **init_kwargs) -> MonitoredCache:
    """
    Create and initialize a monitored cache in a single call.
    
    Args:
        log_dir: Directory where log files will be stored
        init_kwargs: Arguments to pass to the cache initialization
    
    Returns:
        An initialized MonitoredCache instance
    """
    cache = MonitoredCache(log_dir=log_dir)
    cache.init(**init_kwargs)
    return cache