import hashlib
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from ..config.settings import settings
from ..core.interfaces import ReflexionMemory


class ReflexionMemoryCache:
    """Memory cache for reflexion loops with LRU eviction and TTL"""

    def __init__(self, max_size: Optional[int] = None, cache_ttl: int = 86400):
        self.max_size = max_size or 500  # Increased from 100 to 500
        self.cache: OrderedDict[str, ReflexionMemory] = OrderedDict()
        self.access_times: Dict[str, float] = {}
        self.cache_ttl = cache_ttl  # Default: 24 hours

        # Track cache statistics
        self.hits = 0
        self.misses = 0

    def get(self, query_hash: str) -> Optional[ReflexionMemory]:
        """Get reflexion memory from cache with TTL check"""
        if query_hash in self.cache:
            # Check TTL expiration
            age = time.time() - self.access_times.get(query_hash, 0)
            if age > self.cache_ttl:
                # Cache entry expired, remove it
                self.cache.pop(query_hash)
                self.access_times.pop(query_hash, None)
                self.misses += 1
                return None

            # Move to end (most recently used)
            memory = self.cache.pop(query_hash)
            self.cache[query_hash] = memory
            self.access_times[query_hash] = time.time()
            self.hits += 1
            return memory

        self.misses += 1
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
        self.hits = 0
        self.misses = 0

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "oldest_entry": self._get_oldest_entry_age(),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.get_hit_rate():.2%}",
            "ttl_hours": self.cache_ttl / 3600,
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
