import hashlib
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from src.config.settings import settings
from src.core.interfaces import ReflexionMemory


class ReflexionMemoryCache:
    """Memory cache for reflexion loops with LRU eviction"""

    def __init__(self, max_size: Optional[int] = None):
        self.max_size = max_size or settings.max_cache_size
        self.cache: OrderedDict[str, ReflexionMemory] = OrderedDict()
        self.access_times: Dict[str, float] = {}

    def get(self, query_hash: str) -> Optional[ReflexionMemory]:
        """Get reflexion memory from cache"""
        if query_hash in self.cache:
            # Move to end (most recently used)
            memory = self.cache.pop(query_hash)
            self.cache[query_hash] = memory
            self.access_times[query_hash] = time.time()
            return memory
        return None

    def put(self, query_hash: str, memory: ReflexionMemory) -> None:
        """Store reflexion memory in cache"""
        if query_hash in self.cache:
            # Update existing
            self.cache.pop(query_hash)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            oldest_key = next(iter(self.cache))
            self.cache.pop(oldest_key)
            self.access_times.pop(oldest_key, None)

        self.cache[query_hash] = memory
        self.access_times[query_hash] = time.time()

    def has(self, query_hash: str) -> bool:
        """Check if query hash exists in cache"""
        return query_hash in self.cache

    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
        self.access_times.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "oldest_entry": self._get_oldest_entry_age(),
        }

    def _get_oldest_entry_age(self) -> Optional[float]:
        """Get age of oldest cache entry in seconds"""
        if not self.access_times:
            return None
        oldest_time = min(self.access_times.values())
        return time.time() - oldest_time


def create_query_hash(query: str) -> str:
    """Create a hash for query caching"""
    return hashlib.md5(query.lower().strip().encode()).hexdigest()
