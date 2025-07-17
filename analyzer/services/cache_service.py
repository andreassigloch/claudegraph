#!/usr/bin/env python3
"""
Cache Service for Code Architecture Analyzer

Provides a unified caching abstraction that decouples the rest of the system
from the specific caching implementation. This service replaces direct usage
of AdvancedCache throughout the codebase.
"""

import logging
from typing import Any, Optional, Dict, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract interface for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Store value in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class AdvancedCacheBackend(CacheBackend):
    """Cache backend using the existing AdvancedCache implementation."""
    
    def __init__(self, 
                 max_size: int = 1000,
                 default_ttl: Optional[float] = None,
                 eviction_policy: str = "lru"):
        """Initialize with AdvancedCache."""
        from ..llm.cache import AdvancedCache
        
        self._cache = AdvancedCache(
            max_size=max_size,
            default_ttl=default_ttl,
            eviction_policy=eviction_policy,
            enable_persistence=False  # Disable persistence for service layer
        )
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self._cache.get(key)
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Store value in cache."""
        return self._cache.put(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        return self._cache.delete(key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_statistics()


class InMemoryCacheBackend(CacheBackend):
    """Simple in-memory cache backend for testing."""
    
    def __init__(self):
        """Initialize in-memory cache."""
        self._data: Dict[str, Any] = {}
        self._stats = {'hits': 0, 'misses': 0, 'puts': 0, 'deletes': 0}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self._data:
            self._stats['hits'] += 1
            return self._data[key]
        else:
            self._stats['misses'] += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Store value in cache."""
        self._data[key] = value
        self._stats['puts'] += 1
        return True
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if key in self._data:
            del self._data[key]
            self._stats['deletes'] += 1
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._data.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            **self._stats,
            'total_entries': len(self._data),
            'cache_type': 'in_memory'
        }


class CacheService:
    """
    Unified cache service that provides caching abstraction.
    
    This service decouples the rest of the system from specific cache
    implementations and provides a clean interface for caching operations.
    """
    
    def __init__(self, backend: Optional[CacheBackend] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize cache service.
        
        Args:
            backend: Cache backend implementation
            config: Configuration for cache service
        """
        self.config = config or {}
        
        if backend:
            self._backend = backend
        else:
            # Create default backend based on configuration
            cache_config = self.config.get('cache', {})
            backend_type = cache_config.get('backend', 'advanced')
            
            if backend_type == 'advanced':
                self._backend = AdvancedCacheBackend(
                    max_size=cache_config.get('max_size', 1000),
                    default_ttl=cache_config.get('default_ttl', 3600),
                    eviction_policy=cache_config.get('eviction_policy', 'lru')
                )
            elif backend_type == 'memory':
                self._backend = InMemoryCacheBackend()
            else:
                raise ValueError(f"Unknown cache backend: {backend_type}")
        
        self._enabled = self.config.get('cache', {}).get('enabled', True)
        self._key_prefix = self.config.get('cache', {}).get('key_prefix', 'analyzer')
        
        logger.info(f"Cache service initialized with {type(self._backend).__name__}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        if not self._enabled:
            return default
        
        try:
            full_key = self._make_key(key)
            value = self._backend.get(full_key)
            return value if value is not None else default
        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
            return default
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """
        Store value in cache.
        
        Args:
            key: Cache key
            value: Value to store
            ttl: Time-to-live in seconds
            
        Returns:
            True if stored successfully
        """
        if not self._enabled:
            return False
        
        try:
            full_key = self._make_key(key)
            return self._backend.put(full_key, value, ttl)
        except Exception as e:
            logger.warning(f"Cache put failed for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted successfully
        """
        if not self._enabled:
            return False
        
        try:
            full_key = self._make_key(key)
            return self._backend.delete(full_key)
        except Exception as e:
            logger.warning(f"Cache delete failed for key {key}: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        if not self._enabled:
            return
        
        try:
            self._backend.clear()
            logger.info("Cache cleared")
        except Exception as e:
            logger.warning(f"Cache clear failed: {e}")
    
    def get_or_compute(self, key: str, compute_func: callable, ttl: Optional[float] = None) -> Any:
        """
        Get value from cache or compute and store it.
        
        Args:
            key: Cache key
            compute_func: Function to compute value if not cached
            ttl: Time-to-live in seconds
            
        Returns:
            Cached or computed value
        """
        # Try to get from cache first
        value = self.get(key)
        if value is not None:
            return value
        
        # Compute value
        try:
            value = compute_func()
            self.put(key, value, ttl)
            return value
        except Exception as e:
            logger.error(f"Failed to compute value for key {key}: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            stats = self._backend.get_statistics()
            stats['enabled'] = self._enabled
            stats['key_prefix'] = self._key_prefix
            return stats
        except Exception as e:
            logger.warning(f"Failed to get cache statistics: {e}")
            return {'enabled': self._enabled, 'error': str(e)}
    
    def _make_key(self, key: str) -> str:
        """Create full cache key with prefix."""
        return f"{self._key_prefix}:{key}"