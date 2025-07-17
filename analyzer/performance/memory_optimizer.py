#!/usr/bin/env python3
"""
Memory Optimization for Code Architecture Analyzer

Provides memory usage monitoring, optimization strategies, and efficient
data structures for large-scale code analysis.
"""

import logging
import gc
import sys
import time
import weakref

# Optional dependency
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False
from typing import Dict, List, Optional, Any, Callable, Generator, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import threading

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    
    process_memory_mb: float = 0.0
    process_memory_percent: float = 0.0
    system_memory_mb: float = 0.0
    system_memory_percent: float = 0.0
    gc_objects: int = 0
    gc_collections: Dict[int, int] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'process_memory_mb': self.process_memory_mb,
            'process_memory_percent': self.process_memory_percent,
            'system_memory_mb': self.system_memory_mb,
            'system_memory_percent': self.system_memory_percent,
            'gc_objects': self.gc_objects,
            'gc_collections': dict(self.gc_collections),
            'timestamp': self.timestamp
        }


class MemoryMonitor:
    """
    Monitors memory usage and provides optimization recommendations.
    """
    
    def __init__(self, 
                 warning_threshold_mb: float = 1000.0,
                 critical_threshold_mb: float = 2000.0,
                 monitoring_interval: float = 10.0):
        """
        Initialize memory monitor.
        
        Args:
            warning_threshold_mb: Memory usage warning threshold in MB
            critical_threshold_mb: Memory usage critical threshold in MB
            monitoring_interval: Monitoring interval in seconds
        """
        self.warning_threshold = warning_threshold_mb
        self.critical_threshold = critical_threshold_mb
        self.monitoring_interval = monitoring_interval
        
        # Statistics tracking
        self.stats_history: deque = deque(maxlen=100)
        self.peak_memory = 0.0
        self.warning_count = 0
        self.critical_count = 0
        
        # Process reference
        if HAS_PSUTIL:
            try:
                self.process = psutil.Process()
            except:
                self.process = None
                logger.warning("Failed to initialize psutil process")
        else:
            self.process = None
            logger.info("psutil not available - limited memory monitoring")
        
        # Monitoring thread
        self._monitoring_active = False
        self._monitor_thread = None
        
    def get_current_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        stats = MemoryStats()
        
        try:
            if self.process:
                memory_info = self.process.memory_info()
                stats.process_memory_mb = memory_info.rss / 1024 / 1024
                stats.process_memory_percent = self.process.memory_percent()
            
            # System memory (if psutil available)
            if HAS_PSUTIL:
                system_memory = psutil.virtual_memory()
                stats.system_memory_mb = system_memory.used / 1024 / 1024
                stats.system_memory_percent = system_memory.percent
            
            # Garbage collection stats
            stats.gc_objects = len(gc.get_objects())
            for i in range(3):
                stats.gc_collections[i] = gc.get_count()[i]
                
        except Exception as e:
            logger.warning(f"Failed to get memory stats: {e}")
        
        return stats
    
    def start_monitoring(self):
        """Start continuous memory monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous memory monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Internal monitoring loop."""
        while self._monitoring_active:
            try:
                stats = self.get_current_stats()
                self.stats_history.append(stats)
                
                # Update peak memory
                self.peak_memory = max(self.peak_memory, stats.process_memory_mb)
                
                # Check thresholds
                if stats.process_memory_mb >= self.critical_threshold:
                    self.critical_count += 1
                    logger.critical(f"CRITICAL: Memory usage {stats.process_memory_mb:.1f}MB "
                                  f"exceeds threshold {self.critical_threshold}MB")
                    
                elif stats.process_memory_mb >= self.warning_threshold:
                    self.warning_count += 1
                    logger.warning(f"WARNING: Memory usage {stats.process_memory_mb:.1f}MB "
                                 f"exceeds threshold {self.warning_threshold}MB")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of monitoring results."""
        if not self.stats_history:
            return {}
        
        current_stats = self.stats_history[-1]
        
        # Calculate averages
        avg_memory = sum(s.process_memory_mb for s in self.stats_history) / len(self.stats_history)
        max_memory = max(s.process_memory_mb for s in self.stats_history)
        min_memory = min(s.process_memory_mb for s in self.stats_history)
        
        return {
            'current_memory_mb': current_stats.process_memory_mb,
            'peak_memory_mb': self.peak_memory,
            'average_memory_mb': avg_memory,
            'max_memory_mb': max_memory,
            'min_memory_mb': min_memory,
            'warning_count': self.warning_count,
            'critical_count': self.critical_count,
            'samples_collected': len(self.stats_history),
            'monitoring_active': self._monitoring_active
        }


class MemoryOptimizer:
    """
    Provides memory optimization strategies and utilities.
    """
    
    def __init__(self, 
                 auto_gc_threshold: float = 500.0,
                 enable_weak_references: bool = True):
        """
        Initialize memory optimizer.
        
        Args:
            auto_gc_threshold: Automatic garbage collection threshold in MB
            enable_weak_references: Whether to use weak references for caching
        """
        self.auto_gc_threshold = auto_gc_threshold
        self.enable_weak_references = enable_weak_references
        
        # Optimization tracking
        self.gc_runs = 0
        self.objects_freed = 0
        self.bytes_freed = 0
        
        # Weak reference cache for large objects
        if enable_weak_references:
            self.weak_cache: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        else:
            self.weak_cache = None
        
        logger.info("Memory optimizer initialized")
    
    def optimize_memory(self, force: bool = False) -> Dict[str, Any]:
        """
        Run memory optimization strategies.
        
        Args:
            force: Force optimization even if thresholds not met
            
        Returns:
            Dictionary with optimization results
        """
        results = {
            'gc_collected': 0,
            'memory_before_mb': 0.0,
            'memory_after_mb': 0.0,
            'memory_freed_mb': 0.0,
            'optimization_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            # Get memory before optimization
            monitor = MemoryMonitor()
            before_stats = monitor.get_current_stats()
            results['memory_before_mb'] = before_stats.process_memory_mb
            
            # Check if optimization is needed
            if not force and before_stats.process_memory_mb < self.auto_gc_threshold:
                logger.debug("Memory optimization skipped - below threshold")
                return results
            
            # Run garbage collection
            collected = self._run_garbage_collection()
            results['gc_collected'] = collected
            
            # Clear weak references if enabled
            if self.weak_cache:
                weak_cleared = len(self.weak_cache)
                self.weak_cache.clear()
                logger.debug(f"Cleared {weak_cleared} weak references")
            
            # Get memory after optimization
            after_stats = monitor.get_current_stats()
            results['memory_after_mb'] = after_stats.process_memory_mb
            results['memory_freed_mb'] = before_stats.process_memory_mb - after_stats.process_memory_mb
            
            # Update tracking
            self.gc_runs += 1
            self.objects_freed += collected
            self.bytes_freed += results['memory_freed_mb'] * 1024 * 1024
            
            results['optimization_time'] = time.time() - start_time
            
            logger.info(f"Memory optimization completed: "
                       f"freed {results['memory_freed_mb']:.1f}MB in {results['optimization_time']:.3f}s")
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _run_garbage_collection(self) -> int:
        """Run garbage collection and return number of objects collected."""
        # Ensure all generations are collected
        collected = 0
        for generation in range(3):
            collected += gc.collect(generation)
        
        # Force full collection
        collected += gc.collect()
        
        return collected
    
    def create_memory_efficient_list(self, items: List[Any], chunk_size: int = 1000) -> Generator[List[Any], None, None]:
        """
        Create memory-efficient chunked iterator for large lists.
        
        Args:
            items: List to chunk
            chunk_size: Size of each chunk
            
        Yields:
            Chunks of the original list
        """
        for i in range(0, len(items), chunk_size):
            yield items[i:i + chunk_size]
    
    def cache_with_weak_reference(self, key: str, obj: Any) -> bool:
        """
        Cache an object using weak references.
        
        Args:
            key: Cache key
            obj: Object to cache
            
        Returns:
            True if cached successfully
        """
        if not self.weak_cache:
            return False
        
        try:
            self.weak_cache[key] = obj
            return True
        except TypeError:
            # Object doesn't support weak references
            return False
    
    def get_from_weak_cache(self, key: str) -> Optional[Any]:
        """
        Get object from weak reference cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached object or None
        """
        if not self.weak_cache:
            return None
        
        return self.weak_cache.get(key)
    
    def optimize_data_structures(self, data: Any) -> Any:
        """
        Optimize data structures for memory efficiency.
        
        Args:
            data: Data structure to optimize
            
        Returns:
            Optimized data structure
        """
        if isinstance(data, dict):
            return self._optimize_dict(data)
        elif isinstance(data, list):
            return self._optimize_list(data)
        elif isinstance(data, set):
            return self._optimize_set(data)
        else:
            return data
    
    def _optimize_dict(self, data: dict) -> dict:
        """Optimize dictionary for memory efficiency."""
        # Use __slots__ for small dictionaries with known keys
        if len(data) <= 10:
            # Convert to a more memory-efficient structure if beneficial
            pass
        
        # Remove None values to reduce memory
        optimized = {k: v for k, v in data.items() if v is not None}
        
        return optimized
    
    def _optimize_list(self, data: list) -> list:
        """Optimize list for memory efficiency."""
        # Remove None values
        optimized = [item for item in data if item is not None]
        
        # Use tuple if list is not modified
        # This would need context to determine if it's safe
        
        return optimized
    
    def _optimize_set(self, data: set) -> set:
        """Optimize set for memory efficiency."""
        # Remove None values
        optimized = {item for item in data if item is not None}
        
        # Use frozenset if set is not modified
        # This would need context to determine if it's safe
        
        return optimized
    
    def get_object_size_estimate(self, obj: Any) -> int:
        """
        Estimate the memory size of an object.
        
        Args:
            obj: Object to estimate
            
        Returns:
            Estimated size in bytes
        """
        try:
            return sys.getsizeof(obj)
        except:
            # Fallback estimation
            if isinstance(obj, str):
                return len(obj) * 2  # Rough estimate for Unicode
            elif isinstance(obj, (list, tuple)):
                return sum(self.get_object_size_estimate(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(self.get_object_size_estimate(k) + self.get_object_size_estimate(v) 
                          for k, v in obj.items())
            else:
                return 64  # Default estimate
    
    def memory_efficient_merge(self, *iterables) -> Generator[Any, None, None]:
        """
        Memory-efficiently merge multiple iterables.
        
        Args:
            *iterables: Iterables to merge
            
        Yields:
            Items from all iterables
        """
        for iterable in iterables:
            for item in iterable:
                yield item
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get memory optimization statistics."""
        return {
            'gc_runs': self.gc_runs,
            'objects_freed': self.objects_freed,
            'bytes_freed': self.bytes_freed,
            'mb_freed': self.bytes_freed / 1024 / 1024,
            'weak_cache_enabled': self.weak_cache is not None,
            'weak_cache_size': len(self.weak_cache) if self.weak_cache else 0,
            'auto_gc_threshold_mb': self.auto_gc_threshold
        }


class MemoryEfficientDataStructures:
    """Collection of memory-efficient data structures for analysis."""
    
    @staticmethod
    def create_compact_ast_result(file_path: str, module_name: str, 
                                 functions: List[Dict], classes: List[Dict],
                                 imports: List[Dict]) -> Dict[str, Any]:
        """Create memory-efficient AST result representation."""
        return {
            'f': file_path,  # Shortened keys
            'm': module_name,
            'fn': [MemoryEfficientDataStructures._compact_function(f) for f in functions],
            'cl': [MemoryEfficientDataStructures._compact_class(c) for c in classes],
            'im': [MemoryEfficientDataStructures._compact_import(i) for i in imports]
        }
    
    @staticmethod
    def _compact_function(func: Dict[str, Any]) -> Dict[str, Any]:
        """Create compact function representation."""
        return {
            'n': func.get('name', ''),
            'l': func.get('line_start', 0),
            'a': len(func.get('args', [])),
            't': func.get('return_type', ''),
            'd': bool(func.get('decorators'))
        }
    
    @staticmethod
    def _compact_class(cls: Dict[str, Any]) -> Dict[str, Any]:
        """Create compact class representation."""
        return {
            'n': cls.get('name', ''),
            'l': cls.get('line_start', 0),
            'b': cls.get('base_classes', []),
            'm': len(cls.get('methods', []))
        }
    
    @staticmethod
    def _compact_import(imp: Dict[str, Any]) -> Dict[str, Any]:
        """Create compact import representation."""
        return {
            'n': imp.get('name', ''),
            'a': imp.get('alias', ''),
            'f': imp.get('from_module', ''),
            'l': imp.get('line_number', 0)
        }
    
    @staticmethod
    def create_string_interning_dict() -> Dict[str, str]:
        """Create dictionary with automatic string interning."""
        class InterningDict(dict):
            def __setitem__(self, key, value):
                if isinstance(value, str):
                    value = sys.intern(value)
                super().__setitem__(key, value)
        
        return InterningDict()


# Convenience functions
def monitor_memory_usage(func: Callable) -> Callable:
    """Decorator to monitor memory usage of a function."""
    def wrapper(*args, **kwargs):
        monitor = MemoryMonitor()
        before_stats = monitor.get_current_stats()
        
        try:
            result = func(*args, **kwargs)
            
            after_stats = monitor.get_current_stats()
            memory_used = after_stats.process_memory_mb - before_stats.process_memory_mb
            
            if memory_used > 10:  # Log if more than 10MB used
                logger.info(f"{func.__name__} used {memory_used:.1f}MB memory")
            
            return result
            
        except Exception as e:
            after_stats = monitor.get_current_stats()
            memory_used = after_stats.process_memory_mb - before_stats.process_memory_mb
            logger.warning(f"{func.__name__} failed using {memory_used:.1f}MB memory: {e}")
            raise
    
    return wrapper


def optimize_for_memory(auto_gc: bool = True, weak_refs: bool = True):
    """Decorator to optimize function for memory usage."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            optimizer = MemoryOptimizer(enable_weak_references=weak_refs)
            
            try:
                result = func(*args, **kwargs)
                
                if auto_gc:
                    optimizer.optimize_memory()
                
                return result
                
            except Exception as e:
                if auto_gc:
                    optimizer.optimize_memory(force=True)
                raise
        
        return wrapper
    return decorator