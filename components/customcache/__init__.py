__all__ = (
    "DYNCache"
)

import collections
import collections.abc
import functools
import heapq
import random
import time

#from . import keys
from cachetools import Cache

#class _DefaultSize:
#    __slots__ = ()
#
#    def __getitem__(self, _):
#        return 1
#
#    def __setitem__(self, _, value):
#        assert value == 1
#
#    def pop(self, _):
#        return 1

#class Cache(collections.abc.MutableMapping):
#    __marker = object()
#    __size = _DefaultSize()
#
#    def __init__(self, maxsize, getsizeof=None):
#        pass
#
#    def __repr__(self):
#        pass
#
#    def __getitem__(self, key):
#        pass
#
#    def __setitem__(self, key, value):
#        pass
#
#    def __delitem__(self, key):
#        pass
#
#    def __contains__(self, key):
#        pass
#
#    def __missing__(self, key):
#        pass
#
#    def __iter__(self):
#        pass
#
#    def __len__(self):
#        pass

class DynamicCache(Cache):
    def __init__(self, maxsize, getsizeof=None):
        Cache.__init__(self, maxsize, getsizeof)
        self.__order = collections.OrderedDict()
        #print(f"[DEBUG] on_evict callback received: {self._on_evict}")
        #print(f"[DEBUG] Type of on_evict: {type(self._on_evict)}")
        #print(f"[DEBUG] self attributes:\n{'\n'.join(dir(self))}")

    def __getitem__(self, key, cache_getitem=Cache.__getitem__):
        value = cache_getitem(self, key)
        if key in self:  # __missing__ may not store item
            self.__update(key)
        return value

    def __setitem__(self, key, value, cache_setitem=Cache.__setitem__):
        cache_setitem(self, key, value)

        #print(f"\n[DEBUG] __setitem__ called with key={key}, value={value}")
        #print(f"[DEBUG] Current size: {len(self._data)}, maxsize: {self.maxsize}")
        #print(f"[DEBUG] Current keys: {list(self._data.keys())}")        

        self.__update(key)

    def __delitem__(self, key, cache_delitem=Cache.__delitem__):
        cache_delitem(self, key)
        #print(f"[DEBUG] Cache full, need to evict")
        #print(f"[DEBUG] self._on_evict is: {self._on_evict}")
        del self.__order[key]

    def popitem(self):
        """Remove and return the `(key, value)` pair least recently used."""
        try:
            key = next(iter(self.__order))
        except StopIteration:
            raise KeyError("%s is empty" % type(self).__name__) from None
        else:
            return (key, self.pop(key))

    def __update(self, key):
        try:
            self.__order.move_to_end(key)
        except KeyError:
            self.__order[key] = None

    