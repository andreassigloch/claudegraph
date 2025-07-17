#!/usr/bin/env python3
"""
Performance Profiler for Code Architecture Analyzer

Provides comprehensive performance monitoring, profiling, and optimization
recommendations for analysis workflows.
"""

import logging
import time
import functools
from typing import Dict, List, Optional, Any, Callable, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import contextvars

logger = logging.getLogger(__name__)


@dataclass 
class ProfilerStats:
    """Performance profiling statistics."""
    
    function_name: str
    call_count: int = 0
    total_time: float = 0.0
    average_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    total_memory_delta: int = 0
    average_memory_delta: float = 0.0
    last_called: Optional[float] = None
    
    def update(self, execution_time: float, memory_delta: int = 0):
        """Update statistics with new execution data."""
        self.call_count += 1
        self.total_time += execution_time
        self.average_time = self.total_time / self.call_count
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.total_memory_delta += memory_delta
        self.average_memory_delta = self.total_memory_delta / self.call_count
        self.last_called = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'function_name': self.function_name,
            'call_count': self.call_count,
            'total_time': self.total_time,
            'average_time': self.average_time,
            'min_time': self.min_time if self.min_time != float('inf') else 0.0,
            'max_time': self.max_time,
            'total_memory_delta': self.total_memory_delta,
            'average_memory_delta': self.average_memory_delta,
            'last_called': self.last_called
        }


class ExecutionFrame(NamedTuple):
    """Represents a function execution frame."""
    function_name: str
    start_time: float
    start_memory: int


class PerformanceProfiler:
    """
    Comprehensive performance profiler for analysis workflows.
    
    Provides function-level timing, memory tracking, call graphs,
    and optimization recommendations.
    """
    
    def __init__(self, 
                 enable_memory_tracking: bool = True,
                 enable_call_graph: bool = True,
                 max_call_history: int = 1000):
        """
        Initialize performance profiler.
        
        Args:
            enable_memory_tracking: Whether to track memory usage
            enable_call_graph: Whether to track call relationships
            max_call_history: Maximum number of calls to track in history
        """
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_call_graph = enable_call_graph
        self.max_call_history = max_call_history
        
        # Statistics storage
        self.function_stats: Dict[str, ProfilerStats] = {}
        self.call_history: deque = deque(maxlen=max_call_history)
        self.call_graph: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Execution tracking
        self.execution_stack: contextvars.ContextVar = contextvars.ContextVar('execution_stack', default=[])
        self.thread_local = threading.local()
        
        # Profiling state
        self.profiling_enabled = True
        self.start_time = time.time()
        
        logger.info(f"Performance profiler initialized: memory_tracking={enable_memory_tracking}, "
                   f"call_graph={enable_call_graph}")
    
    def profile_function(self, func: Callable = None, *, name: str = None):
        """
        Decorator to profile function execution.
        
        Args:
            func: Function to profile (when used as @profile_function)
            name: Custom name for the function (optional)
            
        Returns:
            Decorated function or decorator
        """
        def decorator(f: Callable) -> Callable:
            function_name = name or f"{f.__module__}.{f.__qualname__}"
            
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                if not self.profiling_enabled:
                    return f(*args, **kwargs)
                
                return self._profile_execution(function_name, f, args, kwargs)
            
            return wrapper
        
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    def profile_context(self, name: str):
        """
        Context manager for profiling code blocks.
        
        Args:
            name: Name for the profiled context
        """
        return self._ProfileContext(self, name)
    
    class _ProfileContext:
        """Context manager for profiling code blocks."""
        
        def __init__(self, profiler: 'PerformanceProfiler', name: str):
            self.profiler = profiler
            self.name = name
            self.start_time = None
            self.start_memory = None
        
        def __enter__(self):
            if not self.profiler.profiling_enabled:
                return self
            
            self.start_time = time.time()
            if self.profiler.enable_memory_tracking:
                self.start_memory = self.profiler._get_memory_usage()
            
            # Add to execution stack
            execution_stack = self.profiler.execution_stack.get()
            frame = ExecutionFrame(self.name, self.start_time, self.start_memory or 0)
            execution_stack.append(frame)
            self.profiler.execution_stack.set(execution_stack)
            
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if not self.profiler.profiling_enabled or self.start_time is None:
                return
            
            end_time = time.time()
            execution_time = end_time - self.start_time
            
            memory_delta = 0
            if self.profiler.enable_memory_tracking and self.start_memory is not None:
                end_memory = self.profiler._get_memory_usage()
                memory_delta = end_memory - self.start_memory
            
            # Remove from execution stack
            execution_stack = self.profiler.execution_stack.get()
            if execution_stack:
                execution_stack.pop()
                self.profiler.execution_stack.set(execution_stack)
            
            # Record statistics
            self.profiler._record_execution(self.name, execution_time, memory_delta)
    
    def _profile_execution(self, function_name: str, func: Callable, args: tuple, kwargs: dict):
        """Profile function execution."""
        start_time = time.time()
        start_memory = self._get_memory_usage() if self.enable_memory_tracking else 0
        
        # Add to execution stack
        execution_stack = self.execution_stack.get()
        frame = ExecutionFrame(function_name, start_time, start_memory)
        execution_stack.append(frame)
        self.execution_stack.set(execution_stack)
        
        # Track call graph
        if self.enable_call_graph and len(execution_stack) > 1:
            caller = execution_stack[-2].function_name
            self.call_graph[caller][function_name] += 1
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            memory_delta = 0
            if self.enable_memory_tracking:
                end_memory = self._get_memory_usage()
                memory_delta = end_memory - start_memory
            
            # Record statistics
            self._record_execution(function_name, execution_time, memory_delta)
            
            return result
            
        finally:
            # Remove from execution stack
            execution_stack = self.execution_stack.get()
            if execution_stack:
                execution_stack.pop()
                self.execution_stack.set(execution_stack)
    
    def _record_execution(self, function_name: str, execution_time: float, memory_delta: int):
        """Record execution statistics."""
        # Update function statistics
        if function_name not in self.function_stats:
            self.function_stats[function_name] = ProfilerStats(function_name)
        
        self.function_stats[function_name].update(execution_time, memory_delta)
        
        # Add to call history
        self.call_history.append({
            'function_name': function_name,
            'execution_time': execution_time,
            'memory_delta': memory_delta,
            'timestamp': time.time()
        })
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except:
            # Fallback to gc object count
            import gc
            return len(gc.get_objects()) * 100  # Rough estimate
    
    def get_function_stats(self, sort_by: str = 'total_time') -> List[Dict[str, Any]]:
        """
        Get function performance statistics.
        
        Args:
            sort_by: Field to sort by ('total_time', 'call_count', 'average_time', etc.)
            
        Returns:
            List of function statistics sorted by specified field
        """
        stats_list = [stats.to_dict() for stats in self.function_stats.values()]
        
        if sort_by in ['total_time', 'call_count', 'average_time', 'max_time']:
            stats_list.sort(key=lambda x: x[sort_by], reverse=True)
        
        return stats_list
    
    def get_call_graph(self) -> Dict[str, Dict[str, int]]:
        """Get call graph showing function relationships."""
        if not self.enable_call_graph:
            return {}
        
        return dict(self.call_graph)
    
    def get_hotspots(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get performance hotspots (slowest functions).
        
        Args:
            top_n: Number of top hotspots to return
            
        Returns:
            List of hotspot functions with performance metrics
        """
        stats = self.get_function_stats(sort_by='total_time')
        hotspots = []
        
        for i, stat in enumerate(stats[:top_n]):
            hotspot = {
                'rank': i + 1,
                'function_name': stat['function_name'],
                'total_time': stat['total_time'],
                'percentage_of_total': 0.0,
                'call_count': stat['call_count'],
                'average_time': stat['average_time'],
                'efficiency_score': 0.0
            }
            
            # Calculate percentage of total profiling time
            total_profiling_time = time.time() - self.start_time
            if total_profiling_time > 0:
                hotspot['percentage_of_total'] = (stat['total_time'] / total_profiling_time) * 100
            
            # Calculate efficiency score (lower is better)
            if stat['call_count'] > 0:
                hotspot['efficiency_score'] = stat['total_time'] / stat['call_count']
            
            hotspots.append(hotspot)
        
        return hotspots
    
    def get_memory_analysis(self) -> Dict[str, Any]:
        """Get memory usage analysis."""
        if not self.enable_memory_tracking:
            return {'error': 'Memory tracking not enabled'}
        
        memory_stats = []
        total_memory_allocated = 0
        total_memory_freed = 0
        
        for stats in self.function_stats.values():
            if stats.total_memory_delta != 0:
                memory_stats.append({
                    'function_name': stats.function_name,
                    'total_memory_delta': stats.total_memory_delta,
                    'average_memory_delta': stats.average_memory_delta,
                    'call_count': stats.call_count
                })
                
                if stats.total_memory_delta > 0:
                    total_memory_allocated += stats.total_memory_delta
                else:
                    total_memory_freed += abs(stats.total_memory_delta)
        
        # Sort by total memory impact
        memory_stats.sort(key=lambda x: abs(x['total_memory_delta']), reverse=True)
        
        return {
            'total_memory_allocated': total_memory_allocated,
            'total_memory_freed': total_memory_freed,
            'net_memory_change': total_memory_allocated - total_memory_freed,
            'memory_intensive_functions': memory_stats[:10],
            'functions_analyzed': len(memory_stats)
        }
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations based on profiling data."""
        recommendations = []
        
        # Identify slow functions
        hotspots = self.get_hotspots(5)
        for hotspot in hotspots:
            if hotspot['total_time'] > 1.0:  # Functions taking more than 1 second
                recommendations.append({
                    'type': 'performance_hotspot',
                    'priority': 'high',
                    'function': hotspot['function_name'],
                    'issue': f"Function consumes {hotspot['total_time']:.2f}s "
                           f"({hotspot['percentage_of_total']:.1f}% of total time)",
                    'recommendation': 'Consider optimizing this function or adding caching'
                })
        
        # Identify frequently called functions
        frequent_functions = [
            stats for stats in self.function_stats.values()
            if stats.call_count > 100 and stats.average_time > 0.01
        ]
        
        for stats in sorted(frequent_functions, key=lambda x: x.call_count, reverse=True)[:3]:
            recommendations.append({
                'type': 'frequent_calls',
                'priority': 'medium',
                'function': stats.function_name,
                'issue': f"Called {stats.call_count} times with {stats.average_time:.4f}s average",
                'recommendation': 'Consider caching results or optimizing algorithm'
            })
        
        # Identify memory-intensive functions
        if self.enable_memory_tracking:
            memory_analysis = self.get_memory_analysis()
            for func_stats in memory_analysis.get('memory_intensive_functions', [])[:3]:
                if func_stats['total_memory_delta'] > 10 * 1024 * 1024:  # > 10MB
                    recommendations.append({
                        'type': 'memory_usage',
                        'priority': 'medium',
                        'function': func_stats['function_name'],
                        'issue': f"Uses {func_stats['total_memory_delta'] / 1024 / 1024:.1f}MB memory",
                        'recommendation': 'Consider memory optimization or streaming processing'
                    })
        
        return recommendations
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        total_profiling_time = time.time() - self.start_time
        
        report = {
            'profiling_summary': {
                'profiling_duration': total_profiling_time,
                'functions_profiled': len(self.function_stats),
                'total_function_calls': sum(stats.call_count for stats in self.function_stats.values()),
                'total_execution_time': sum(stats.total_time for stats in self.function_stats.values()),
                'memory_tracking_enabled': self.enable_memory_tracking,
                'call_graph_enabled': self.enable_call_graph
            },
            'performance_hotspots': self.get_hotspots(10),
            'function_statistics': self.get_function_stats('total_time')[:20],
            'optimization_recommendations': self.get_optimization_recommendations()
        }
        
        if self.enable_memory_tracking:
            report['memory_analysis'] = self.get_memory_analysis()
        
        if self.enable_call_graph:
            report['call_graph_summary'] = {
                'total_relationships': sum(
                    sum(calls.values()) for calls in self.call_graph.values()
                ),
                'unique_callers': len(self.call_graph),
                'most_called_functions': self._get_most_called_functions()
            }
        
        return report
    
    def _get_most_called_functions(self) -> List[Dict[str, Any]]:
        """Get most called functions from call graph."""
        call_counts = defaultdict(int)
        
        for caller, callees in self.call_graph.items():
            for callee, count in callees.items():
                call_counts[callee] += count
        
        most_called = sorted(call_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return [
            {'function_name': func, 'total_calls': count}
            for func, count in most_called
        ]
    
    def reset_statistics(self):
        """Reset all profiling statistics."""
        self.function_stats.clear()
        self.call_history.clear()
        self.call_graph.clear()
        self.start_time = time.time()
        
        logger.info("Performance profiler statistics reset")
    
    def enable_profiling(self):
        """Enable profiling."""
        self.profiling_enabled = True
        logger.info("Performance profiling enabled")
    
    def disable_profiling(self):
        """Disable profiling."""
        self.profiling_enabled = False
        logger.info("Performance profiling disabled")
    
    def export_profile_data(self, file_path: str):
        """
        Export profile data to file.
        
        Args:
            file_path: Path to export file
        """
        import json
        
        report = self.generate_report()
        
        try:
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Profile data exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export profile data: {e}")
            raise


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_global_profiler() -> PerformanceProfiler:
    """Get or create global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def profile(func: Callable = None, *, name: str = None):
    """Convenience decorator using global profiler."""
    return get_global_profiler().profile_function(func, name=name)


def profile_context(name: str):
    """Convenience context manager using global profiler."""
    return get_global_profiler().profile_context(name)


def get_performance_report() -> Dict[str, Any]:
    """Get performance report from global profiler."""
    return get_global_profiler().generate_report()


def reset_profiler():
    """Reset global profiler statistics."""
    get_global_profiler().reset_statistics()


# Example usage decorator for automatic profiling
def auto_profile(enable: bool = True):
    """Decorator to automatically profile all methods of a class."""
    def class_decorator(cls):
        if not enable:
            return cls
        
        profiler = get_global_profiler()
        
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if callable(attr) and not attr_name.startswith('__'):
                setattr(cls, attr_name, profiler.profile_function(attr, name=f"{cls.__name__}.{attr_name}"))
        
        return cls
    
    return class_decorator