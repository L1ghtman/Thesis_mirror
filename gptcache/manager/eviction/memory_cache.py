from typing import Any, Callable, List, Tuple

import cachetools

from gptcache.manager.eviction.base import EvictionBase
from adaptive_pipeline import AdaptivePipelineCache


# Sentinel value returned by AdaptivePipelineCache.popitem() when queue is empty
# The C++ code returns: std::make_pair(0, std::make_tuple(0, std::numeric_limits<uint64_t>::max()))
AP_POPITEM_SENTINEL_TOKENS = 18446744073709551615  # uint64_t max


def popitem_wrapper(func, wrapper_func, clean_size):
    def wrapper(*args, **kwargs):
        keys = []
        try:
            keys = [func(*args, **kwargs)[0] for _ in range(clean_size)]
        except KeyError:
            pass
        if wrapper_func is not None:
            wrapper_func(keys)

    return wrapper


class MemoryCacheEviction(EvictionBase):
    """eviction: Memory Cache

    :param policy: eviction strategy
    :type policy: str
    :param maxsize: the maxsize of cache data
    :type maxsize: int
    :param clean_size: will clean the size of data when the size of cache data reaches the max size
    :type clean_size: int
    :param on_evict: the function for cleaning the data in the store
    :type  on_evict: Callable[[List[Any]], None]


    """

    def __init__(
            self,
            policy: str = "LRU",
            maxsize: int = 1000,
            clean_size: int = 0,
            on_evict: Callable[[List[Any]], None] = None,
            **kwargs,
    ):
        self._policy = policy.upper()
        self._on_evict = on_evict  # Store callback for AP policy

        print(f"--- Policy set to '{self._policy}' ---")

        if self._policy == "LRU":
            self._cache = cachetools.LRUCache(maxsize=maxsize, **kwargs)
        elif self._policy == "LFU":
            self._cache = cachetools.LFUCache(maxsize=maxsize, **kwargs)
        elif self._policy == "FIFO":
            self._cache = cachetools.FIFOCache(maxsize=maxsize, **kwargs)
        elif self._policy == "RR":
            self._cache = cachetools.RRCache(maxsize=maxsize, **kwargs)
        elif self._policy == "AP":
            config_path = 'config.json'
            self._cache = AdaptivePipelineCache(config_path=config_path)
        else:
            raise ValueError(f"Unknown policy {policy}")

#        if not on_evict:
#            self.eviction_manager = EvictionManager()

        # For non-AP policies, wrap popitem to trigger eviction callback
        # For AP policy, we handle eviction differently in put()
        if self._policy != "AP":
            self._cache.popitem = popitem_wrapper(self._cache.popitem, on_evict, clean_size)

    def _drain_ap_eviction_queue(self) -> List[int]:
        """
        Drain the AdaptivePipelineCache eviction queue after insert operations.
        Returns list of evicted item keys.
        
        The C++ insert_item() pushes evicted items to an internal queue.
        popitem() returns them one by one until the queue is empty,
        at which point it returns a sentinel value with tokens = uint64_max.
        """
        evicted_keys = []
        while True:
            key, (latency, tokens) = self._cache.popitem()
            # Check for sentinel value indicating empty queue
            if tokens == AP_POPITEM_SENTINEL_TOKENS:
                break
            evicted_keys.append(key)
        return evicted_keys

    def put(self, objs: List[Tuple[int, Tuple[float, int]]]):
        for obj in objs:
            if isinstance(obj, tuple):
                self._cache[obj[0]] = obj[1]
        
        # For AP policy, drain eviction queue and call callback
        if self._policy == "AP":
            evicted_keys = self._drain_ap_eviction_queue()
            if evicted_keys and self._on_evict is not None:
                self._on_evict(evicted_keys)

    def get(self, obj: Any):
        return self._cache.get(obj)

    @property
    def policy(self) -> str:
        return self._policy
