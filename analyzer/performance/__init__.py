#!/usr/bin/env python3
"""
Performance Optimization Components for Code Architecture Analyzer

This module provides performance enhancements including concurrent processing,
memory optimization, caching strategies, and streaming analysis capabilities.

Components included:
- ConcurrentProcessor: Parallel execution of analysis tasks
- MemoryOptimizer: Memory usage optimization and monitoring
- StreamingAnalyzer: Large file processing with streaming
- CacheOptimizer: Advanced caching strategies and precomputation
- PerformanceProfiler: Analysis performance monitoring and tuning
"""

from .concurrent_processor import ConcurrentProcessor, TaskPool
from .memory_optimizer import MemoryOptimizer, MemoryMonitor
from .streaming_analyzer import StreamingAnalyzer, ChunkProcessor
from .cache_optimizer import CacheOptimizer, PrecomputationEngine
from .performance_profiler import PerformanceProfiler, ProfilerStats

__all__ = [
    'ConcurrentProcessor',
    'TaskPool',
    'MemoryOptimizer',
    'MemoryMonitor',
    'StreamingAnalyzer',
    'ChunkProcessor',
    'CacheOptimizer',
    'PrecomputationEngine',
    'PerformanceProfiler',
    'ProfilerStats'
]