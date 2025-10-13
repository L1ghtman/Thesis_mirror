from typing import Any, Callable, List, Dict, Optional, Tuple
import time
import random
from collections import OrderedDict
from gptcache.manager.eviction.base import EvictionBase

from components.customcache import DynamicCache

class OldDynamicCacheO:
    #def __init__(self, maxsize: int = 1000, **kwargs):
    #    print(f"--- Initializing dynamic cache with maxsize {maxsize} ---")
    #    self.maxsize = maxsize
    #    self._data: Dict[Any, Any] = {}
    #    self._access_times: Dict[Any, float] = {}
    #    self._access_counts: Dict[Any, int] = {}
    #    self._insertion_order = OrderedDict()

    #    self.temperature = kwargs.get('temperature', 1.0)
    #    self.decay_factor = kwargs.get('decay_factor', 0.95)

    #    self._on_evict = kwargs.get('on_evict', None)

    def __init__(
            self, 
            policy: str = "DYN",
            maxsize: int = 1000, 
            clean_size: int = 1,
            on_evict: Callable[[List[Any]], None] = None,
            **kwargs
    ):
        self._policy = policy.upper()

        print(f"--- Initializing dynamic cache with maxsize {maxsize} ---")
        self.maxsize = maxsize
        self._data: Dict[Any, Any] = {}
        self._access_times: Dict[Any, float] = {}
        self._access_counts: Dict[Any, int] = {}
        self._insertion_order = OrderedDict()
        
        self.temperature = kwargs.get('temperature', 1.0)
        self.decay_factor = kwargs.get('decay_factor', 0.95)
        
        # CRITICAL: Store and verify the callback
        self._on_evict = kwargs.get('on_evict', None)
        print(f"[DEBUG] (dynamic_eviction.py > DynamicCache > __init__) on_evict callback received: {self._on_evict}")
        print(f"[DEBUG] (dynamic_eviction.py > DynamicCache > __init__) Type of on_evict: {type(self._on_evict)}")

    def __getitem__(self, key):
        if key not in self._data:
            raise KeyError(key)
        self._update_access_stats(key)
        return self._data[key]
    
    #def __setitem__(self, key, value):
    #    if key in self._data:
    #        self._data[key] = value
    #        self._update_access_stats(key)
    #        return
    #    if len(self._data) >= self.maxsize:
    #        self._evict_items(1)
    #    self._data[key] = value
    #    self._access_times[key] = time.time()
    #    self._access_counts[key] = 1
    #    self._insertion_order[key] = True

    def __setitem__(self, key, value):
        print(f"\n[DEBUG] (dynamic_eviction.py > DynamicCache > __setitem__) __setitem__ called with key={key}, value={value}")
        print(f"[DEBUG] (dynamic_eviction.py > DynamicCache > __setitem__) Current size: {len(self._data)}, maxsize: {self.maxsize}")
        print(f"[DEBUG] (dynamic_eviction.py > DynamicCache > __setitem__) Current keys: {list(self._data.keys())}")
        
        if key in self._data:
            self._data[key] = value
            self._update_access_stats(key)
            return
    
        # Evict BEFORE adding if at capacity
        if len(self._data) >= self.maxsize:
            print(f"[DEBUG] (dynamic_eviction.py > DynamicCache > __setitem__) Cache full, need to evict")
            print(f"[DEBUG] (dynamic_eviction.py > DynamicCache > __setitem__) self._on_evict is: {self._on_evict}")
            victim_key = self._select_eviction_victim()
            if victim_key is not None:
                # Remove from cache
                del self._data[victim_key]
                del self._access_times[victim_key]
                del self._access_counts[victim_key]
                if victim_key in self._insertion_order:
                    del self._insertion_order[victim_key]

                # CRITICAL: Call on_evict NOW!
                if self._on_evict:
                    print(f"[DEBUG] (dynamic_eviction.py > DynamicCache > __setitem__) Calling on_evict with victim key: {victim_key}")
                    self._on_evict([victim_key])
                else:
                    print(f"[DEBUG] (dynamic_eviction.py > DynamicCache > __setitem__) WARNING: on_evict is None!")
    
        # Now add new item
        self._data[key] = value
        self._access_times[key] = time.time()
        self._access_counts[key] = 1
        self._insertion_order[key] = True

    def __delitem__(self, key):
        if key not in self._data:
            raise KeyError(key)
        del self._data[key]
        del self._access_times[key]
        del self._access_counts[key]
        if key in self._insertion_order:
            del self._insertion_order[key]

    def __contains__(self, key):
        return key in self._data

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def get(self, key, default=None):
        if key in self._data:
            self._update_access_stats(key)
            return self._data[key]
        return default

    def keys(self):
        self._data.keys()

    def values(self):
        self._data.values()

    def items(self):
        self._data.items()

    def popitem(self):
        if not self._data:
            raise KeyError('popitem(): dictonary is empty')
        victim_key = self._select_eviction_victim()
        if victim_key:
            value = self._data[victim_key]
            del self[victim_key]
            return (victim_key, value)
        oldest_key = min(self._access_times, key=self._access_times.get)
        value = self._data[oldest_key]
        del self[oldest_key]
        return (oldest_key, value)
 
    def _update_access_stats(self, key):
        self._access_times[key] = time.time()
        self._access_counts[key] = self._access_counts.get(key, 0) + 1

    def _evict_items(self, num_items: int = 1):
        if len(self._data) == 0:
            return
        evicted_keys = []
        for _ in range(min(num_items, len(self._data))):
            victim_key = self._select_eviction_victim()
            if victim_key:
                evicted_keys.append(victim_key)
                del self[victim_key]
        if self._on_evict and evicted_keys:
            print("Calling on_evict with keys: {evicted_keys}")
            self._on_evict(evicted_keys)

    def _select_eviction_victim(self) -> Optional[Any]:
        # TODO: Implement eviction policy here

        print("selecting eviction victim")

        if not self._data:
            return None
        candidates = list(self._data.keys())
        if random.random() < self.temperature:
            return random.choice(candidates)
        scores = {}
        current_time = time.time()
        for key in candidates:
            time_since_access = current_time - self._access_times[key]
            recency_score = 1.0 / (1.0 + time_since_access)
            frequency_score = self._access_counts[key]
            age_penalty = time_since_access * 0.1
            scores[key] = (
                            recency_score * 0.4 +
                            frequency_score * 0.4 +
                            age_penalty * 0.2
                        )
        victim_key = min(scores, key=scores.get)
        return victim_key

    def update_temperature(self, new_temperature: float):
        self.temperature = max(0.0, min(2.0, new_temperature))

    def get_cache_stats(self):
        if not self._data:
            return {"size": 0, "avg_access_count": 0, "temperature": self.temperature}
        return {
            "size": len(self._data),
            "max_size": self.maxsize,
            "avg_access_count": sum(self._access_counts.values()) / len(self._access_counts),
            "temperature": self.temperature,
            "oldest_access": min(self._access_times.values()) if self._access_times else None,
            "newest_access": max(self._access_times.values()) if self._access_times else None,

        }



def popitem_wrapper(func, wrapper_func, clean_size):
    def wrapper(*args, **kwargs):
        keys = []
        try:
            keys = [func(*args, **kwargs)[0] for _ in range(clean_size)]
        except KeyError:
            pass
        wrapper_func(keys)

    return wrapper


class DynamicEviction(EvictionBase):
    """
    Description of DynamicEviction
    """
    def __init__(
            self,
            policy: str = "DYN",
            maxsize: int = 1000,
            clean_size: int = 0,
            on_evict: Callable[[List[Any]], None] = None,
            **kwargs,
    ):
        self._policy = policy.upper()
        if self._policy == "DYN":
            self._cache = DynamicCache(maxsize=maxsize, **kwargs)
            print(f"[INFO] initialized DynamicCache with maxsize {maxsize}")
        else:
            raise ValueError(f"Unknown policy {policy}")

        self._cache.popitem = popitem_wrapper(self._cache.popitem, on_evict, clean_size)

    def put(self, objs: List[Any]):
        for obj in objs:
            self._cache[obj] = True

    def get(self, obj: Any):
        return self._cache.get(obj)

    def update_temperature(self, new_temperature: float):
        self._cache.update_temperature(new_temperature)
        
    def get_cache_status(self):
        return self._cache.get_cache_stats()

    @property
    def policy(self) -> str:
        return self._policy