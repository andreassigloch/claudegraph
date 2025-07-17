#!/usr/bin/env python3
"""
Advanced Caching System for Code Architecture Analyzer

Provides pattern-based caching for LLM responses and other expensive operations
with TTL support, statistics tracking, and configurable eviction policies.
"""

import logging
import time
import json
import hashlib
import threading
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import OrderedDict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """Update last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary for serialization."""
        return {
            'key': self.key,
            'value': self.value,
            'created_at': self.created_at,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count,
            'ttl': self.ttl
        }


class EvictionPolicy(ABC):
    """Abstract base class for cache eviction policies."""
    
    @abstractmethod
    def select_victim(self, entries: Dict[str, CacheEntry]) -> Optional[str]:
        """Select a cache entry key for eviction."""
        pass


class LRUEvictionPolicy(EvictionPolicy):
    """Least Recently Used eviction policy."""
    
    def select_victim(self, entries: Dict[str, CacheEntry]) -> Optional[str]:
        """Select least recently used entry for eviction."""
        if not entries:
            return None
        
        oldest_key = min(entries.keys(), 
                        key=lambda k: entries[k].last_accessed)
        return oldest_key


class LFUEvictionPolicy(EvictionPolicy):
    """Least Frequently Used eviction policy."""
    
    def select_victim(self, entries: Dict[str, CacheEntry]) -> Optional[str]:
        """Select least frequently used entry for eviction."""
        if not entries:
            return None
        
        least_used_key = min(entries.keys(), 
                           key=lambda k: entries[k].access_count)
        return least_used_key


class FIFOEvictionPolicy(EvictionPolicy):
    """First In, First Out eviction policy."""
    
    def select_victim(self, entries: Dict[str, CacheEntry]) -> Optional[str]:
        """Select oldest entry for eviction."""
        if not entries:
            return None
        
        oldest_key = min(entries.keys(), 
                        key=lambda k: entries[k].created_at)
        return oldest_key


@dataclass
class CacheStatistics:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expired_removals: int = 0
    manual_removals: int = 0
    total_entries: int = 0
    total_size_bytes: int = 0
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            **asdict(self),
            'hit_rate': self.hit_rate(),
            'miss_rate': self.miss_rate()
        }


class AdvancedCache:
    """
    Advanced caching system with TTL, eviction policies, and statistics.
    
    Features:
    - Configurable eviction policies (LRU, LFU, FIFO)
    - Time-to-live (TTL) support for entries
    - Thread-safe operations
    - Detailed statistics tracking
    - Optional persistence to disk
    - Pattern-based key generation
    """
    
    def __init__(self, 
                 max_size: int = 1000,
                 default_ttl: Optional[float] = None,
                 eviction_policy: str = "lru",
                 enable_persistence: bool = False,
                 persistence_file: Optional[str] = None):
        """
        Initialize the advanced cache.
        
        Args:
            max_size: Maximum number of entries to store
            default_ttl: Default time-to-live in seconds
            eviction_policy: Eviction policy ("lru", "lfu", "fifo")
            enable_persistence: Whether to persist cache to disk
            persistence_file: File path for cache persistence
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.enable_persistence = enable_persistence
        self.persistence_file = persistence_file or "cache.json"
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Storage and metadata
        self._entries: Dict[str, CacheEntry] = {}
        self._statistics = CacheStatistics()
        
        # Eviction policy
        self._eviction_policy = self._create_eviction_policy(eviction_policy)
        
        # Load persistent cache if enabled
        if self.enable_persistence:
            self._load_from_disk()
        
        logger.info(f"Cache initialized: max_size={max_size}, ttl={default_ttl}, "
                   f"policy={eviction_policy}, persistence={enable_persistence}")
    
    def _create_eviction_policy(self, policy_name: str) -> EvictionPolicy:
        """Create eviction policy instance."""
        policies = {
            "lru": LRUEvictionPolicy,
            "lfu": LFUEvictionPolicy, 
            "fifo": FIFOEvictionPolicy
        }
        
        policy_class = policies.get(policy_name.lower())
        if not policy_class:
            logger.warning(f"Unknown eviction policy '{policy_name}', using LRU")
            policy_class = LRUEvictionPolicy
        
        return policy_class()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            entry = self._entries.get(key)
            
            if entry is None:
                self._statistics.misses += 1
                return None
            
            # Check if expired
            if entry.is_expired():
                self._remove_entry(key, reason="expired")
                self._statistics.misses += 1
                self._statistics.expired_removals += 1
                return None
            
            # Update access information
            entry.touch()
            self._statistics.hits += 1
            
            logger.debug(f"Cache hit: {key}")
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """
        Store value in cache.
        
        Args:
            key: Cache key
            value: Value to store
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            True if stored successfully
        """
        with self._lock:
            # Use provided TTL or default
            effective_ttl = ttl if ttl is not None else self.default_ttl
            
            # Check if we need to evict entries
            if key not in self._entries and len(self._entries) >= self.max_size:
                self._evict_entry()
            
            # Create new entry
            current_time = time.time()
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                ttl=effective_ttl
            )
            
            # Store entry
            self._entries[key] = entry
            self._statistics.total_entries = len(self._entries)
            
            # Persist if enabled
            if self.enable_persistence:
                self._save_to_disk()
            
            logger.debug(f"Cache put: {key} (ttl={effective_ttl})")
            return True
    
    def delete(self, key: str) -> bool:
        """
        Remove entry from cache.
        
        Args:
            key: Cache key to remove
            
        Returns:
            True if entry was removed
        """
        with self._lock:
            if key in self._entries:
                self._remove_entry(key, reason="manual")
                self._statistics.manual_removals += 1
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            removed_count = len(self._entries)
            self._entries.clear()
            self._statistics.total_entries = 0
            self._statistics.manual_removals += removed_count
            
            if self.enable_persistence:
                self._save_to_disk()
            
            logger.info(f"Cache cleared: removed {removed_count} entries")
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._entries.items() 
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                self._remove_entry(key, reason="expired")
                self._statistics.expired_removals += 1
            
            logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
            return len(expired_keys)
    
    def _evict_entry(self):
        """Evict one entry based on eviction policy."""
        victim_key = self._eviction_policy.select_victim(self._entries)
        if victim_key:
            self._remove_entry(victim_key, reason="evicted")
            self._statistics.evictions += 1
            logger.debug(f"Evicted entry: {victim_key}")
    
    def _remove_entry(self, key: str, reason: str = "unknown"):
        """Remove entry and update statistics."""
        if key in self._entries:
            del self._entries[key]
            self._statistics.total_entries = len(self._entries)
            logger.debug(f"Removed entry {key} (reason: {reason})")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            # Calculate current cache size in bytes (approximate)
            total_size = 0
            for entry in self._entries.values():
                try:
                    # Rough approximation of entry size
                    total_size += len(str(entry.value)) + len(entry.key) + 100
                except:
                    total_size += 100  # Fallback estimate
            
            self._statistics.total_size_bytes = total_size
            
            return {
                **self._statistics.to_dict(),
                'current_entries': len(self._entries),
                'max_size': self.max_size,
                'utilization': len(self._entries) / self.max_size,
                'eviction_policy': self._eviction_policy.__class__.__name__,
                'default_ttl': self.default_ttl,
                'persistence_enabled': self.enable_persistence
            }
    
    def get_keys(self) -> List[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self._entries.keys())
    
    def contains(self, key: str) -> bool:
        """Check if key exists in cache (and is not expired)."""
        with self._lock:
            entry = self._entries.get(key)
            return entry is not None and not entry.is_expired()
    
    def size(self) -> int:
        """Get current number of entries in cache."""
        with self._lock:
            return len(self._entries)
    
    def _save_to_disk(self):
        """Save cache to disk for persistence."""
        try:
            cache_data = {
                'entries': {
                    key: entry.to_dict() 
                    for key, entry in self._entries.items()
                },
                'statistics': self._statistics.to_dict(),
                'saved_at': time.time()
            }
            
            with open(self.persistence_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save cache to disk: {str(e)}")
    
    def _load_from_disk(self):
        """Load cache from disk if file exists."""
        try:
            if not Path(self.persistence_file).exists():
                return
            
            with open(self.persistence_file, 'r') as f:
                cache_data = json.load(f)
            
            # Restore entries
            current_time = time.time()
            loaded_count = 0
            
            for key, entry_data in cache_data.get('entries', {}).items():
                # Skip expired entries
                if entry_data.get('ttl') and (current_time - entry_data['created_at'] > entry_data['ttl']):
                    continue
                
                entry = CacheEntry(
                    key=entry_data['key'],
                    value=entry_data['value'],
                    created_at=entry_data['created_at'],
                    last_accessed=entry_data['last_accessed'],
                    access_count=entry_data.get('access_count', 0),
                    ttl=entry_data.get('ttl')
                )
                
                self._entries[key] = entry
                loaded_count += 1
            
            self._statistics.total_entries = len(self._entries)
            logger.info(f"Loaded {loaded_count} entries from cache file")
            
        except Exception as e:
            logger.warning(f"Failed to load cache from disk: {str(e)}")


class PatternBasedCache(AdvancedCache):
    """
    Specialized cache for pattern-based caching of actor enhancements.
    
    Generates cache keys based on code patterns and provides
    specialized methods for actor enhancement caching.
    """
    
    def __init__(self, **kwargs):
        """Initialize pattern-based cache with sensible defaults for actor enhancement."""
        # Override defaults for actor enhancement use case
        kwargs.setdefault('max_size', 1000)
        kwargs.setdefault('default_ttl', 24 * 3600)  # 24 hours
        kwargs.setdefault('eviction_policy', 'lru')
        
        super().__init__(**kwargs)
    
    def generate_pattern_key(self, 
                           actor_type: str,
                           library: str, 
                           url_or_target: str = "",
                           function_context: str = "") -> str:
        """
        Generate cache key based on actor pattern.
        
        Args:
            actor_type: Type of actor (HttpClient, Database, etc.)
            library: Library used (requests, sqlite3, etc.)
            url_or_target: URL or target (optional)
            function_context: Function context (optional)
            
        Returns:
            Generated cache key
        """
        # Create pattern identifier
        pattern_parts = [actor_type, library]
        
        # Add URL domain or target if available
        if url_or_target:
            if url_or_target.startswith('http'):
                # Extract domain from URL
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(url_or_target).netloc
                    pattern_parts.append(domain)
                except:
                    pattern_parts.append(url_or_target[:50])  # Truncate long targets
            else:
                pattern_parts.append(url_or_target[:50])
        
        # Add function context hints
        if function_context:
            # Extract key terms from function context
            context_terms = []
            key_terms = ['payment', 'user', 'config', 'auth', 'data', 'file', 'api']
            context_lower = function_context.lower()
            
            for term in key_terms:
                if term in context_lower:
                    context_terms.append(term)
            
            if context_terms:
                pattern_parts.extend(context_terms[:2])  # Max 2 context terms
        
        # Create hash of the pattern
        pattern_string = ":".join(pattern_parts)
        pattern_hash = hashlib.md5(pattern_string.encode()).hexdigest()[:16]
        
        return f"actor_pattern_{pattern_hash}"
    
    def get_actor_enhancement(self, 
                            actor_type: str,
                            library: str,
                            url_or_target: str = "",
                            function_context: str = "") -> Optional[Dict[str, Any]]:
        """Get cached actor enhancement by pattern."""
        key = self.generate_pattern_key(actor_type, library, url_or_target, function_context)
        return self.get(key)
    
    def put_actor_enhancement(self,
                            actor_type: str,
                            library: str, 
                            enhancement: Dict[str, Any],
                            url_or_target: str = "",
                            function_context: str = "",
                            ttl: Optional[float] = None) -> bool:
        """Store actor enhancement by pattern."""
        key = self.generate_pattern_key(actor_type, library, url_or_target, function_context)
        return self.put(key, enhancement, ttl)


def create_actor_cache(config: Dict[str, Any]) -> PatternBasedCache:
    """
    Create and configure actor enhancement cache from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured PatternBasedCache instance
    """
    cache_config = config.get('llm', {}).get('actor_enhancement', {})
    
    return PatternBasedCache(
        max_size=cache_config.get('cache_size_limit', 1000),
        default_ttl=cache_config.get('cache_ttl_hours', 24) * 3600,
        eviction_policy=cache_config.get('cache_eviction_policy', 'lru'),
        enable_persistence=cache_config.get('cache_persistence', False),
        persistence_file=cache_config.get('cache_file', 'actor_cache.json')
    )