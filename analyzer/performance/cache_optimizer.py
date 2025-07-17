#!/usr/bin/env python3
"""
Cache Optimization for Code Architecture Analyzer

Provides advanced caching strategies, precomputation engines, and
cache performance optimization for analysis workflows.
"""

import logging
import time
import hashlib
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import threading

logger = logging.getLogger(__name__)


@dataclass
class CacheKey:
    """Structured cache key with metadata."""
    
    namespace: str
    identifier: str
    version: str = "1.0"
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    
    def to_string(self) -> str:
        """Convert to string representation."""
        # Create deterministic string from parameters
        param_str = ""
        if self.parameters:
            sorted_params = sorted(self.parameters.items())
            param_str = str(sorted_params)
        
        # Include dependencies
        dep_str = ""
        if self.dependencies:
            dep_str = "|".join(sorted(self.dependencies))
        
        # Combine all parts
        full_key = f"{self.namespace}:{self.identifier}:{self.version}:{param_str}:{dep_str}"
        
        # Hash for consistent length
        return hashlib.md5(full_key.encode()).hexdigest()


@dataclass
class CacheEntry:
    """Enhanced cache entry with optimization metadata."""
    
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    computation_time: float = 0.0
    size_estimate: int = 0
    dependencies: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    
    def update_access(self):
        """Update access information."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def is_dependent_on(self, invalidated_key: str) -> bool:
        """Check if this entry depends on an invalidated key."""
        return invalidated_key in self.dependencies


class CacheStrategy(ABC):
    """Abstract base class for cache strategies."""
    
    @abstractmethod
    def should_cache(self, key: CacheKey, computation_time: float, result_size: int) -> bool:
        """Determine if a result should be cached."""
        pass
    
    @abstractmethod
    def get_ttl(self, key: CacheKey, computation_time: float) -> Optional[float]:
        """Get time-to-live for cache entry."""
        pass
    
    @abstractmethod
    def select_eviction_candidates(self, entries: Dict[str, CacheEntry], target_count: int) -> List[str]:
        """Select entries for eviction."""
        pass


class AdaptiveCacheStrategy(CacheStrategy):
    """Adaptive cache strategy that learns from access patterns."""
    
    def __init__(self,
                 min_computation_time: float = 0.1,
                 size_threshold_mb: float = 10.0,
                 base_ttl: float = 3600.0):
        """
        Initialize adaptive cache strategy.
        
        Args:
            min_computation_time: Minimum computation time to consider caching
            size_threshold_mb: Maximum size threshold for caching
            base_ttl: Base time-to-live in seconds
        """
        self.min_computation_time = min_computation_time
        self.size_threshold_bytes = size_threshold_mb * 1024 * 1024
        self.base_ttl = base_ttl
        
        # Learning data
        self.access_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.computation_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
    
    def should_cache(self, key: CacheKey, computation_time: float, result_size: int) -> bool:
        """Determine if result should be cached based on adaptive criteria."""
        key_str = key.to_string()
        
        # Record computation time
        self.computation_times[key_str].append(computation_time)
        
        # Don't cache if computation is too fast
        if computation_time < self.min_computation_time:
            return False
        
        # Don't cache if result is too large
        if result_size > self.size_threshold_bytes:
            return False
        
        # Cache if computation is expensive relative to access frequency
        if key_str in self.access_patterns:
            recent_accesses = len([t for t in self.access_patterns[key_str] 
                                 if time.time() - t < 3600])  # Last hour
            
            # Cache if accessed multiple times and computation is expensive
            if recent_accesses > 1 and computation_time > 0.5:
                return True
        
        # Default caching for expensive computations
        return computation_time > 1.0
    
    def get_ttl(self, key: CacheKey, computation_time: float) -> Optional[float]:
        """Get adaptive TTL based on computation cost and access patterns."""
        key_str = key.to_string()
        
        # Base TTL adjusted by computation time
        ttl = self.base_ttl * min(computation_time / 10.0, 5.0)
        
        # Adjust based on access frequency
        if key_str in self.access_patterns:
            access_frequency = len(self.access_patterns[key_str]) / max(1, 
                len(self.access_patterns[key_str]))
            ttl *= (1 + access_frequency)
        
        return min(ttl, 86400.0)  # Max 24 hours
    
    def select_eviction_candidates(self, entries: Dict[str, CacheEntry], target_count: int) -> List[str]:
        """Select entries for eviction using adaptive scoring."""
        if len(entries) <= target_count:
            return []
        
        # Score entries for eviction (higher score = more likely to evict)
        scores = {}
        current_time = time.time()
        
        for key, entry in entries.items():
            age = current_time - entry.created_at
            last_access_age = current_time - entry.last_accessed
            
            # Factors that increase eviction likelihood
            age_score = age / 86400.0  # Age in days
            access_score = last_access_age / 3600.0  # Hours since last access
            frequency_score = 1.0 / max(1, entry.access_count)
            size_score = entry.size_estimate / (1024 * 1024)  # Size in MB
            
            # Factors that decrease eviction likelihood
            computation_score = entry.computation_time / 10.0
            
            # Combined score
            eviction_score = (age_score + access_score + frequency_score + size_score) - computation_score
            scores[key] = max(0, eviction_score)
        
        # Sort by score and return candidates
        sorted_entries = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        candidates_to_evict = len(entries) - target_count
        
        return [key for key, score in sorted_entries[:candidates_to_evict]]


class PrecomputationEngine:
    """
    Engine for precomputing and caching expensive operations.
    """
    
    def __init__(self, cache_service, strategy: CacheStrategy = None):
        """
        Initialize precomputation engine.
        
        Args:
            cache_service: Cache service instance
            strategy: Cache strategy (uses adaptive if None)
        """
        self.cache_service = cache_service
        self.strategy = strategy or AdaptiveCacheStrategy()
        
        # Precomputation tracking
        self.precomputation_queue: deque = deque()
        self.precomputation_results: Dict[str, Any] = {}
        self.background_thread = None
        self.running = False
        
        logger.info("Precomputation engine initialized")
    
    def register_precomputation(self, 
                              key: CacheKey,
                              computation_func: Callable[[], Any],
                              priority: int = 0):
        """
        Register a computation for background precomputation.
        
        Args:
            key: Cache key for the computation
            computation_func: Function to compute the result
            priority: Priority (higher = computed first)
        """
        self.precomputation_queue.append({
            'key': key,
            'func': computation_func,
            'priority': priority,
            'registered_at': time.time()
        })
        
        # Sort by priority
        self.precomputation_queue = deque(
            sorted(self.precomputation_queue, key=lambda x: x['priority'], reverse=True)
        )
        
        logger.debug(f"Registered precomputation: {key.to_string()}")
    
    def start_background_precomputation(self):
        """Start background precomputation thread."""
        if self.running:
            return
        
        self.running = True
        self.background_thread = threading.Thread(target=self._precomputation_loop, daemon=True)
        self.background_thread.start()
        
        logger.info("Background precomputation started")
    
    def stop_background_precomputation(self):
        """Stop background precomputation thread."""
        self.running = False
        if self.background_thread:
            self.background_thread.join(timeout=5.0)
        
        logger.info("Background precomputation stopped")
    
    def _precomputation_loop(self):
        """Background precomputation loop."""
        while self.running:
            try:
                if not self.precomputation_queue:
                    time.sleep(1.0)
                    continue
                
                # Get next computation
                computation = self.precomputation_queue.popleft()
                key = computation['key']
                func = computation['func']
                
                # Check if already cached
                cached_result = self.cache_service.get(key.to_string())
                if cached_result is not None:
                    continue
                
                # Perform computation
                start_time = time.time()
                try:
                    result = func()
                    computation_time = time.time() - start_time
                    
                    # Estimate result size
                    result_size = self._estimate_size(result)
                    
                    # Decide whether to cache
                    if self.strategy.should_cache(key, computation_time, result_size):
                        ttl = self.strategy.get_ttl(key, computation_time)
                        self.cache_service.put(key.to_string(), result, ttl=ttl)
                        
                        logger.debug(f"Precomputed and cached: {key.to_string()} "
                                   f"({computation_time:.3f}s, {result_size} bytes)")
                    
                    # Store in results
                    self.precomputation_results[key.to_string()] = {
                        'result': result,
                        'computation_time': computation_time,
                        'size': result_size,
                        'computed_at': time.time()
                    }
                    
                except Exception as e:
                    logger.warning(f"Precomputation failed for {key.to_string()}: {e}")
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Precomputation loop error: {e}")
                time.sleep(1.0)
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            import sys
            return sys.getsizeof(obj)
        except:
            # Fallback estimation
            if isinstance(obj, str):
                return len(obj) * 2
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj[:10])  # Sample first 10
            elif isinstance(obj, dict):
                sample_items = list(obj.items())[:10]  # Sample first 10
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in sample_items)
            else:
                return 100  # Default estimate
    
    def get_precomputation_stats(self) -> Dict[str, Any]:
        """Get precomputation statistics."""
        return {
            'queue_size': len(self.precomputation_queue),
            'results_cached': len(self.precomputation_results),
            'background_running': self.running,
            'total_computation_time': sum(
                r['computation_time'] for r in self.precomputation_results.values()
            ),
            'total_results_size': sum(
                r['size'] for r in self.precomputation_results.values()
            )
        }


class CacheOptimizer:
    """
    High-level cache optimizer that coordinates caching strategies.
    """
    
    def __init__(self, 
                 cache_service,
                 strategy: CacheStrategy = None,
                 enable_precomputation: bool = True):
        """
        Initialize cache optimizer.
        
        Args:
            cache_service: Cache service instance
            strategy: Caching strategy
            enable_precomputation: Whether to enable precomputation
        """
        self.cache_service = cache_service
        self.strategy = strategy or AdaptiveCacheStrategy()
        
        # Cache tracking
        self.cache_entries: Dict[str, CacheEntry] = {}
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.invalidation_count = 0
        
        # Precomputation engine
        if enable_precomputation:
            self.precomputation_engine = PrecomputationEngine(cache_service, strategy)
            self.precomputation_engine.start_background_precomputation()
        else:
            self.precomputation_engine = None
        
        logger.info("Cache optimizer initialized")
    
    def cache_computation(self, 
                         key: CacheKey, 
                         computation_func: Callable[[], Any],
                         force_cache: bool = False) -> Any:
        """
        Cache a computation result with optimization.
        
        Args:
            key: Cache key
            computation_func: Function to compute result
            force_cache: Force caching regardless of strategy
            
        Returns:
            Computation result
        """
        key_str = key.to_string()
        
        # Check if already cached
        cached_result = self.cache_service.get(key_str)
        if cached_result is not None:
            # Update access tracking
            if key_str in self.cache_entries:
                self.cache_entries[key_str].update_access()
            
            logger.debug(f"Cache hit: {key_str}")
            return cached_result
        
        # Perform computation
        start_time = time.time()
        try:
            result = computation_func()
            computation_time = time.time() - start_time
            
            # Estimate result size
            result_size = self._estimate_size(result)
            
            # Decide whether to cache
            should_cache = force_cache or self.strategy.should_cache(key, computation_time, result_size)
            
            if should_cache:
                ttl = self.strategy.get_ttl(key, computation_time)
                success = self.cache_service.put(key_str, result, ttl=ttl)
                
                if success:
                    # Track cache entry
                    entry = CacheEntry(
                        key=key_str,
                        value=result,
                        created_at=time.time(),
                        last_accessed=time.time(),
                        computation_time=computation_time,
                        size_estimate=result_size,
                        dependencies=key.dependencies.copy()
                    )
                    self.cache_entries[key_str] = entry
                    
                    # Update dependency graph
                    for dep in key.dependencies:
                        self.dependency_graph[dep].add(key_str)
                    
                    logger.debug(f"Cached computation: {key_str} "
                               f"({computation_time:.3f}s, {result_size} bytes)")
            
            return result
            
        except Exception as e:
            logger.error(f"Computation failed for {key_str}: {e}")
            raise
    
    def invalidate_cache(self, key_pattern: str = None, tags: Set[str] = None):
        """
        Invalidate cache entries based on pattern or tags.
        
        Args:
            key_pattern: Key pattern to match (None = all)
            tags: Tags to match for invalidation
        """
        keys_to_invalidate = set()
        
        # Find keys to invalidate
        if key_pattern:
            keys_to_invalidate.update([
                key for key in self.cache_entries.keys()
                if key_pattern in key
            ])
        
        if tags:
            keys_to_invalidate.update([
                key for key, entry in self.cache_entries.items()
                if tags.intersection(entry.tags)
            ])
        
        # Also invalidate dependent entries
        for key in list(keys_to_invalidate):
            keys_to_invalidate.update(self.dependency_graph.get(key, set()))
        
        # Perform invalidation
        for key in keys_to_invalidate:
            self.cache_service.delete(key)
            if key in self.cache_entries:
                del self.cache_entries[key]
            self.invalidation_count += 1
        
        logger.info(f"Invalidated {len(keys_to_invalidate)} cache entries")
    
    def optimize_cache_performance(self) -> Dict[str, Any]:
        """
        Optimize cache performance and return statistics.
        
        Returns:
            Optimization results
        """
        optimization_results = {
            'entries_before': len(self.cache_entries),
            'entries_evicted': 0,
            'memory_freed_estimate': 0,
            'optimization_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            # Get current cache statistics
            cache_stats = self.cache_service.get_statistics()
            utilization = cache_stats.get('utilization', 0.0)
            
            # Optimize if cache is getting full
            if utilization > 0.8:
                target_entries = int(len(self.cache_entries) * 0.7)  # Target 70% full
                
                # Use strategy to select eviction candidates
                candidates = self.strategy.select_eviction_candidates(
                    self.cache_entries, target_entries
                )
                
                # Evict candidates
                memory_freed = 0
                for key in candidates:
                    if key in self.cache_entries:
                        memory_freed += self.cache_entries[key].size_estimate
                        self.cache_service.delete(key)
                        del self.cache_entries[key]
                
                optimization_results['entries_evicted'] = len(candidates)
                optimization_results['memory_freed_estimate'] = memory_freed
            
            # Clean up dependency graph
            self._cleanup_dependency_graph()
            
            optimization_results['entries_after'] = len(self.cache_entries)
            optimization_results['optimization_time'] = time.time() - start_time
            
            logger.info(f"Cache optimization completed: "
                       f"evicted {optimization_results['entries_evicted']} entries, "
                       f"freed ~{optimization_results['memory_freed_estimate'] / 1024 / 1024:.1f}MB")
            
        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
            optimization_results['error'] = str(e)
        
        return optimization_results
    
    def _cleanup_dependency_graph(self):
        """Clean up dependency graph of invalid references."""
        valid_keys = set(self.cache_entries.keys())
        
        for key in list(self.dependency_graph.keys()):
            self.dependency_graph[key] = {
                dep for dep in self.dependency_graph[key]
                if dep in valid_keys
            }
            
            # Remove empty dependency sets
            if not self.dependency_graph[key]:
                del self.dependency_graph[key]
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size."""
        if self.precomputation_engine:
            return self.precomputation_engine._estimate_size(obj)
        
        try:
            import sys
            return sys.getsizeof(obj)
        except:
            return 100
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = {
            'cache_entries': len(self.cache_entries),
            'dependency_relationships': sum(len(deps) for deps in self.dependency_graph.values()),
            'invalidation_count': self.invalidation_count,
            'cache_service_stats': self.cache_service.get_statistics()
        }
        
        if self.precomputation_engine:
            stats['precomputation'] = self.precomputation_engine.get_precomputation_stats()
        
        # Cache entry statistics
        if self.cache_entries:
            total_size = sum(entry.size_estimate for entry in self.cache_entries.values())
            total_computation_time = sum(entry.computation_time for entry in self.cache_entries.values())
            
            stats['cache_analysis'] = {
                'total_size_estimate': total_size,
                'average_size': total_size / len(self.cache_entries),
                'total_computation_time_saved': total_computation_time,
                'average_computation_time': total_computation_time / len(self.cache_entries)
            }
        
        return stats
    
    def shutdown(self):
        """Shutdown cache optimizer."""
        if self.precomputation_engine:
            self.precomputation_engine.stop_background_precomputation()
        
        logger.info("Cache optimizer shutdown")


# Convenience functions
def create_cache_key(namespace: str, identifier: str, **kwargs) -> CacheKey:
    """Create a cache key with automatic parameter handling."""
    return CacheKey(
        namespace=namespace,
        identifier=identifier,
        parameters=kwargs
    )


def cached_computation(cache_optimizer: CacheOptimizer, 
                      namespace: str, 
                      identifier: str,
                      **cache_params):
    """Decorator for automatic computation caching."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Create cache key including function parameters
            key = CacheKey(
                namespace=namespace,
                identifier=identifier,
                parameters={**cache_params, 'args': args, 'kwargs': kwargs}
            )
            
            return cache_optimizer.cache_computation(key, lambda: func(*args, **kwargs))
        
        return wrapper
    return decorator