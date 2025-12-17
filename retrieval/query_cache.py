"""
Query rewrite caching to avoid repeated LLM calls for similar queries.
"""
import hashlib
from typing import List, Optional

class QueryRewriteCache:
    """Cache for query rewrites to avoid repeated LLM calls."""
    
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
    
    def _hash_query(self, query: str) -> str:
        """Create hash key for query."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[List[str]]:
        """Get cached rewrite if exists."""
        key = self._hash_query(query)
        return self.cache.get(key)
    
    def set(self, query: str, rewrites: List[str]):
        """Cache query rewrite."""
        key = self._hash_query(query)
        
        # Simple LRU: remove oldest if cache full
        if len(self.cache) >= self.max_size:
            # Remove first item (FIFO)
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        
        self.cache[key] = rewrites
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()

# Global cache instance
_query_cache = QueryRewriteCache()

def get_cached_rewrite(query: str) -> Optional[List[str]]:
    """Get cached rewrite for query."""
    return _query_cache.get(query)

def cache_rewrite(query: str, rewrites: List[str]):
    """Cache query rewrite."""
    _query_cache.set(query, rewrites)
