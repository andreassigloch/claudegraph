#!/usr/bin/env python3
"""
Concurrent Processing for Code Architecture Analyzer

Provides parallel execution capabilities for CPU-intensive analysis tasks
with proper resource management and error handling.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple, Iterator
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from queue import Queue, Empty
from abc import ABC, abstractmethod
import multiprocessing as mp

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result of a concurrent task execution."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    worker_id: Optional[str] = None


@dataclass
class TaskStats:
    """Statistics for task execution."""
    tasks_submitted: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_duration: float = 0.0
    average_duration: float = 0.0
    max_duration: float = 0.0
    min_duration: float = float('inf')
    
    def update(self, result: TaskResult):
        """Update statistics with a task result."""
        self.tasks_completed += 1
        if result.success:
            pass  # Success count is implicit
        else:
            self.tasks_failed += 1
        
        self.total_duration += result.duration_seconds
        self.average_duration = self.total_duration / self.tasks_completed
        self.max_duration = max(self.max_duration, result.duration_seconds)
        self.min_duration = min(self.min_duration, result.duration_seconds)


class TaskPool(ABC):
    """Abstract base class for task execution pools."""
    
    @abstractmethod
    def submit(self, task_id: str, func: Callable, *args, **kwargs) -> Future:
        """Submit a task for execution."""
        pass
    
    @abstractmethod
    def shutdown(self, wait: bool = True):
        """Shutdown the task pool."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        pass


class ThreadTaskPool(TaskPool):
    """Thread-based task pool for I/O-bound operations."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.submitted_tasks = 0
        self.completed_tasks = 0
        
        logger.info(f"Thread pool initialized with {self.max_workers} workers")
    
    def submit(self, task_id: str, func: Callable, *args, **kwargs) -> Future:
        """Submit a task to the thread pool."""
        self.submitted_tasks += 1
        
        def wrapped_task():
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                self.completed_tasks += 1
                return TaskResult(
                    task_id=task_id,
                    success=True,
                    result=result,
                    duration_seconds=duration,
                    worker_id=threading.current_thread().name
                )
            except Exception as e:
                duration = time.time() - start_time
                self.completed_tasks += 1
                return TaskResult(
                    task_id=task_id,
                    success=False,
                    error=str(e),
                    duration_seconds=duration,
                    worker_id=threading.current_thread().name
                )
        
        return self.executor.submit(wrapped_task)
    
    def shutdown(self, wait: bool = True):
        """Shutdown the thread pool."""
        self.executor.shutdown(wait=wait)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get thread pool statistics."""
        return {
            'pool_type': 'thread',
            'max_workers': self.max_workers,
            'submitted_tasks': self.submitted_tasks,
            'completed_tasks': self.completed_tasks,
            'pending_tasks': self.submitted_tasks - self.completed_tasks
        }


class ProcessTaskPool(TaskPool):
    """Process-based task pool for CPU-bound operations."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.submitted_tasks = 0
        self.completed_tasks = 0
        
        logger.info(f"Process pool initialized with {self.max_workers} workers")
    
    def submit(self, task_id: str, func: Callable, *args, **kwargs) -> Future:
        """Submit a task to the process pool."""
        self.submitted_tasks += 1
        
        def wrapped_task():
            import os
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                return TaskResult(
                    task_id=task_id,
                    success=True,
                    result=result,
                    duration_seconds=duration,
                    worker_id=str(os.getpid())
                )
            except Exception as e:
                duration = time.time() - start_time
                return TaskResult(
                    task_id=task_id,
                    success=False,
                    error=str(e),
                    duration_seconds=duration,
                    worker_id=str(os.getpid())
                )
        
        future = self.executor.submit(wrapped_task)
        return future
    
    def shutdown(self, wait: bool = True):
        """Shutdown the process pool."""
        self.executor.shutdown(wait=wait)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get process pool statistics."""
        return {
            'pool_type': 'process',
            'max_workers': self.max_workers,
            'submitted_tasks': self.submitted_tasks,
            'completed_tasks': self.completed_tasks,
            'pending_tasks': self.submitted_tasks - self.completed_tasks
        }


class ConcurrentProcessor:
    """
    High-level concurrent processor for analysis tasks.
    
    Automatically chooses between thread and process pools based on task type
    and provides comprehensive task management and monitoring.
    """
    
    def __init__(self, 
                 thread_workers: int = None,
                 process_workers: int = None,
                 enable_process_pool: bool = True):
        """
        Initialize concurrent processor.
        
        Args:
            thread_workers: Number of thread workers (None = auto)
            process_workers: Number of process workers (None = auto)
            enable_process_pool: Whether to enable process pool for CPU-bound tasks
        """
        self.thread_pool = ThreadTaskPool(thread_workers)
        
        if enable_process_pool:
            try:
                self.process_pool = ProcessTaskPool(process_workers)
            except Exception as e:
                logger.warning(f"Failed to initialize process pool: {e}")
                self.process_pool = None
        else:
            self.process_pool = None
        
        self.task_stats = TaskStats()
        self.active_futures: Dict[str, Future] = {}
        self._lock = threading.Lock()
        
        logger.info("Concurrent processor initialized")
    
    def submit_io_task(self, task_id: str, func: Callable, *args, **kwargs) -> str:
        """
        Submit an I/O-bound task (file reading, network calls, etc.).
        
        Args:
            task_id: Unique identifier for the task
            func: Function to execute
            *args, **kwargs: Function arguments
            
        Returns:
            Task ID for tracking
        """
        future = self.thread_pool.submit(task_id, func, *args, **kwargs)
        
        with self._lock:
            self.active_futures[task_id] = future
            self.task_stats.tasks_submitted += 1
        
        logger.debug(f"Submitted I/O task: {task_id}")
        return task_id
    
    def submit_cpu_task(self, task_id: str, func: Callable, *args, **kwargs) -> str:
        """
        Submit a CPU-bound task (parsing, computation, etc.).
        
        Args:
            task_id: Unique identifier for the task
            func: Function to execute
            *args, **kwargs: Function arguments
            
        Returns:
            Task ID for tracking
        """
        # Use process pool if available, otherwise fall back to thread pool
        pool = self.process_pool if self.process_pool else self.thread_pool
        future = pool.submit(task_id, func, *args, **kwargs)
        
        with self._lock:
            self.active_futures[task_id] = future
            self.task_stats.tasks_submitted += 1
        
        logger.debug(f"Submitted CPU task: {task_id}")
        return task_id
    
    def submit_batch_io_tasks(self, tasks: List[Tuple[str, Callable, Tuple, Dict]]) -> List[str]:
        """
        Submit multiple I/O tasks efficiently.
        
        Args:
            tasks: List of (task_id, func, args, kwargs) tuples
            
        Returns:
            List of task IDs
        """
        task_ids = []
        
        for task_id, func, args, kwargs in tasks:
            self.submit_io_task(task_id, func, *args, **kwargs)
            task_ids.append(task_id)
        
        logger.info(f"Submitted {len(tasks)} I/O tasks")
        return task_ids
    
    def submit_batch_cpu_tasks(self, tasks: List[Tuple[str, Callable, Tuple, Dict]]) -> List[str]:
        """
        Submit multiple CPU tasks efficiently.
        
        Args:
            tasks: List of (task_id, func, args, kwargs) tuples
            
        Returns:
            List of task IDs
        """
        task_ids = []
        
        for task_id, func, args, kwargs in tasks:
            self.submit_cpu_task(task_id, func, *args, **kwargs)
            task_ids.append(task_id)
        
        logger.info(f"Submitted {len(tasks)} CPU tasks")
        return task_ids
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """
        Get the result of a specific task.
        
        Args:
            task_id: Task identifier
            timeout: Maximum time to wait for result
            
        Returns:
            TaskResult with execution details
        """
        with self._lock:
            future = self.active_futures.get(task_id)
        
        if not future:
            return TaskResult(
                task_id=task_id,
                success=False,
                error="Task not found"
            )
        
        try:
            result = future.result(timeout=timeout)
            
            with self._lock:
                if task_id in self.active_futures:
                    del self.active_futures[task_id]
                self.task_stats.update(result)
            
            return result
            
        except Exception as e:
            error_result = TaskResult(
                task_id=task_id,
                success=False,
                error=str(e)
            )
            
            with self._lock:
                if task_id in self.active_futures:
                    del self.active_futures[task_id]
                self.task_stats.update(error_result)
            
            return error_result
    
    def get_results(self, task_ids: List[str], timeout: Optional[float] = None) -> List[TaskResult]:
        """
        Get results for multiple tasks.
        
        Args:
            task_ids: List of task identifiers
            timeout: Maximum time to wait for all results
            
        Returns:
            List of TaskResults in the same order as task_ids
        """
        results = []
        
        for task_id in task_ids:
            result = self.get_result(task_id, timeout)
            results.append(result)
        
        return results
    
    def wait_for_completion(self, task_ids: Optional[List[str]] = None, timeout: Optional[float] = None) -> bool:
        """
        Wait for tasks to complete.
        
        Args:
            task_ids: Specific tasks to wait for (None = all active tasks)
            timeout: Maximum time to wait
            
        Returns:
            True if all tasks completed within timeout
        """
        if task_ids:
            futures = [self.active_futures.get(tid) for tid in task_ids if tid in self.active_futures]
        else:
            with self._lock:
                futures = list(self.active_futures.values())
        
        try:
            # Wait for all futures to complete
            for future in as_completed(futures, timeout=timeout):
                future.result()  # This will raise any exceptions
            return True
            
        except Exception as e:
            logger.warning(f"Wait for completion failed: {e}")
            return False
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a specific task if possible.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if task was cancelled
        """
        with self._lock:
            future = self.active_futures.get(task_id)
        
        if future:
            cancelled = future.cancel()
            if cancelled:
                with self._lock:
                    if task_id in self.active_futures:
                        del self.active_futures[task_id]
                logger.debug(f"Cancelled task: {task_id}")
            return cancelled
        
        return False
    
    def get_active_tasks(self) -> List[str]:
        """Get list of active task IDs."""
        with self._lock:
            return list(self.active_futures.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive processor statistics."""
        with self._lock:
            active_count = len(self.active_futures)
        
        stats = {
            'task_stats': {
                'submitted': self.task_stats.tasks_submitted,
                'completed': self.task_stats.tasks_completed,
                'failed': self.task_stats.tasks_failed,
                'active': active_count,
                'success_rate': (self.task_stats.tasks_completed - self.task_stats.tasks_failed) / max(1, self.task_stats.tasks_completed),
                'average_duration': self.task_stats.average_duration,
                'max_duration': self.task_stats.max_duration,
                'min_duration': self.task_stats.min_duration if self.task_stats.min_duration != float('inf') else 0.0
            },
            'thread_pool': self.thread_pool.get_stats()
        }
        
        if self.process_pool:
            stats['process_pool'] = self.process_pool.get_stats()
        
        return stats
    
    def shutdown(self, timeout: float = 30.0):
        """
        Gracefully shutdown the processor.
        
        Args:
            timeout: Maximum time to wait for active tasks
        """
        logger.info("Shutting down concurrent processor")
        
        # Wait for active tasks to complete
        if not self.wait_for_completion(timeout=timeout):
            logger.warning(f"Shutdown timeout after {timeout}s, some tasks may be incomplete")
        
        # Shutdown pools
        self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        logger.info("Concurrent processor shutdown complete")


# Convenience functions for common concurrent processing patterns
def process_files_concurrently(file_paths: List[str], 
                             processor_func: Callable[[str], Any],
                             max_workers: int = None) -> List[TaskResult]:
    """
    Process multiple files concurrently.
    
    Args:
        file_paths: List of file paths to process
        processor_func: Function that processes a single file
        max_workers: Maximum number of concurrent workers
        
    Returns:
        List of TaskResults
    """
    processor = ConcurrentProcessor(thread_workers=max_workers)
    
    try:
        # Submit all file processing tasks
        tasks = [(f"file_{i}", processor_func, (path,), {}) 
                for i, path in enumerate(file_paths)]
        
        task_ids = processor.submit_batch_io_tasks(tasks)
        
        # Wait for completion and collect results
        results = processor.get_results(task_ids)
        
        return results
        
    finally:
        processor.shutdown()


def process_chunks_concurrently(data_chunks: List[Any],
                               processor_func: Callable[[Any], Any],
                               max_workers: int = None) -> List[TaskResult]:
    """
    Process data chunks concurrently.
    
    Args:
        data_chunks: List of data chunks to process
        processor_func: Function that processes a single chunk
        max_workers: Maximum number of concurrent workers
        
    Returns:
        List of TaskResults
    """
    processor = ConcurrentProcessor(process_workers=max_workers)
    
    try:
        # Submit all chunk processing tasks
        tasks = [(f"chunk_{i}", processor_func, (chunk,), {}) 
                for i, chunk in enumerate(data_chunks)]
        
        task_ids = processor.submit_batch_cpu_tasks(tasks)
        
        # Wait for completion and collect results
        results = processor.get_results(task_ids)
        
        return results
        
    finally:
        processor.shutdown()