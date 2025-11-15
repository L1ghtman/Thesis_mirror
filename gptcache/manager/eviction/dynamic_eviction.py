from typing import Any, Callable, List, Dict, Optional, Tuple
import time
import random
from collections import OrderedDict
from gptcache.manager.eviction.base import EvictionBase

from components.customcache import DynamicCache

def popitem_wrapper(func, wrapper_func, clean_size):
    def wrapper(*args, **kwargs):
        keys = []
        try:
            keys = [func(*args, **kwargs)[0] for _ in range(clean_size)]
        except KeyError:
            pass
        # TODO I think I forgot this line originally, or perhaps omitted for good reason
        if wrapper_func is not None:
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
            #print(f"[INFO] initialized DynamicCache with maxsize {maxsize}")
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