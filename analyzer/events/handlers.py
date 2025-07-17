#!/usr/bin/env python3
"""
Event Handlers for Code Architecture Analyzer

Provides specialized event handlers that process events for logging,
metrics collection, caching, and other cross-cutting concerns.
"""

import logging
import time
import json
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict, deque
from abc import ABC, abstractmethod
from pathlib import Path

from .events import (
    Event, 
    AnalysisEvent,
    ProjectDiscoveredEvent,
    FileParseEvent,
    ActorDetectedEvent,
    GraphGeneratedEvent,
    ErrorEvent,
    ProgressEvent,
    MetricsEvent,
    CacheEvent
)

logger = logging.getLogger(__name__)


class EventHandler(ABC):
    """Abstract base class for event handlers."""
    
    @abstractmethod
    def handle(self, event: Event) -> None:
        """Handle an event."""
        pass
    
    def can_handle(self, event: Event) -> bool:
        """Check if this handler can process the event."""
        return True
    
    def get_name(self) -> str:
        """Get handler name."""
        return self.__class__.__name__


class LoggingEventHandler(EventHandler):
    """Event handler that logs events with configurable detail levels."""
    
    def __init__(self, 
                 log_level: int = logging.INFO,
                 include_metadata: bool = True,
                 include_event_data: bool = False,
                 event_type_filters: Optional[Set[str]] = None):
        """
        Initialize logging event handler.
        
        Args:
            log_level: Logging level for events
            include_metadata: Whether to include event metadata in logs
            include_event_data: Whether to include full event data
            event_type_filters: Set of event types to log (None = all)
        """
        self.log_level = log_level
        self.include_metadata = include_metadata
        self.include_event_data = include_event_data
        self.event_type_filters = event_type_filters
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def handle(self, event: Event) -> None:
        """Log the event."""
        if not self.can_handle(event):
            return
        
        # Build log message
        message_parts = [f"Event: {event.event_type}"]
        
        if hasattr(event, 'stage') and event.stage:
            message_parts.append(f"Stage: {event.stage}")
        
        if hasattr(event, 'success'):
            message_parts.append(f"Success: {event.success}")
        
        if hasattr(event, 'duration_seconds') and event.duration_seconds:
            message_parts.append(f"Duration: {event.duration_seconds:.3f}s")
        
        # Add event-specific information
        if isinstance(event, ProjectDiscoveredEvent):
            message_parts.append(f"Files: {event.python_files}/{event.total_files}")
            message_parts.append(f"Lines: {event.total_lines}")
        elif isinstance(event, FileParseEvent):
            message_parts.append(f"File: {event.module_name}")
            message_parts.append(f"Functions: {event.functions_found}")
        elif isinstance(event, ActorDetectedEvent):
            message_parts.append(f"Module: {event.module_name}")
            message_parts.append(f"Actors: {event.actors_detected}")
        elif isinstance(event, GraphGeneratedEvent):
            message_parts.append(f"Nodes: {event.nodes_generated}")
            message_parts.append(f"Relationships: {event.relationships_generated}")
        elif isinstance(event, ErrorEvent):
            message_parts.append(f"Error: {event.error_message}")
            message_parts.append(f"Component: {event.component}")
        
        message = " | ".join(message_parts)
        
        # Add metadata if requested
        if self.include_metadata:
            metadata = {
                'event_id': event.event_id,
                'timestamp': event.timestamp,
                'source': event.source,
                'correlation_id': event.correlation_id
            }
            message += f" | Metadata: {metadata}"
        
        # Add full event data if requested
        if self.include_event_data:
            try:
                event_data = event.to_dict()
                message += f" | Data: {json.dumps(event_data, indent=2)}"
            except Exception as e:
                message += f" | Data serialization failed: {e}"
        
        # Log at appropriate level
        if isinstance(event, ErrorEvent):
            self.logger.error(message)
        elif hasattr(event, 'success') and not event.success:
            self.logger.warning(message)
        else:
            self.logger.log(self.log_level, message)
    
    def can_handle(self, event: Event) -> bool:
        """Check if this event should be logged."""
        if self.event_type_filters:
            return event.event_type in self.event_type_filters
        return True


class MetricsEventHandler(EventHandler):
    """Event handler that collects and aggregates metrics from events."""
    
    def __init__(self, 
                 metrics_window_size: int = 1000,
                 enable_histograms: bool = True):
        """
        Initialize metrics event handler.
        
        Args:
            metrics_window_size: Size of rolling window for metrics
            enable_histograms: Whether to collect histogram data
        """
        self.metrics_window_size = metrics_window_size
        self.enable_histograms = enable_histograms
        
        # Metric storage
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=metrics_window_size))
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=metrics_window_size))
        
        # Event-specific metrics
        self.event_counts: Dict[str, int] = defaultdict(int)
        self.stage_durations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        self.start_time = time.time()
    
    def handle(self, event: Event) -> None:
        """Process event for metrics collection."""
        # Count all events
        self.event_counts[event.event_type] += 1
        self.counters['events_total'] += 1
        
        # Process analysis events
        if isinstance(event, AnalysisEvent):
            self._handle_analysis_event(event)
        
        # Process error events
        elif isinstance(event, ErrorEvent):
            self._handle_error_event(event)
        
        # Process progress events
        elif isinstance(event, ProgressEvent):
            self._handle_progress_event(event)
        
        # Process metrics events
        elif isinstance(event, MetricsEvent):
            self._handle_metrics_event(event)
        
        # Process cache events
        elif isinstance(event, CacheEvent):
            self._handle_cache_event(event)
    
    def _handle_analysis_event(self, event: AnalysisEvent):
        """Handle analysis-specific events."""
        stage = event.stage
        
        if event.success:
            self.counters[f'stage_{stage}_success'] += 1
        else:
            self.counters[f'stage_{stage}_failure'] += 1
        
        if event.duration_seconds:
            self.stage_durations[stage].append(event.duration_seconds)
            self.timers[f'stage_{stage}_duration'].append(event.duration_seconds)
        
        # Event-specific metrics
        if isinstance(event, ProjectDiscoveredEvent):
            self.gauges['files_discovered'] = event.total_files
            self.gauges['python_files_discovered'] = event.python_files
            self.gauges['total_lines_discovered'] = event.total_lines
            
        elif isinstance(event, FileParseEvent):
            self.counters['functions_parsed'] += event.functions_found
            self.counters['classes_parsed'] += event.classes_found
            self.counters['imports_parsed'] += event.imports_found
            
        elif isinstance(event, ActorDetectedEvent):
            self.counters['actors_detected'] += event.actors_detected
            self.counters['high_confidence_actors'] += event.high_confidence_actors
            self.counters['ambiguous_actors'] += event.ambiguous_actors
            
        elif isinstance(event, GraphGeneratedEvent):
            self.gauges['nodes_generated'] = event.nodes_generated
            self.gauges['relationships_generated'] = event.relationships_generated
            self.gauges['flow_relationships'] = event.flow_relationships
    
    def _handle_error_event(self, event: ErrorEvent):
        """Handle error events."""
        self.error_counts[event.error_type] += 1
        self.error_counts[event.component] += 1
        self.counters['errors_total'] += 1
        
        if not event.recoverable:
            self.counters['critical_errors'] += 1
    
    def _handle_progress_event(self, event: ProgressEvent):
        """Handle progress events."""
        self.gauges[f'progress_{event.stage}'] = event.progress_percentage
        
        if event.estimated_remaining_seconds:
            self.gauges[f'eta_{event.stage}'] = event.estimated_remaining_seconds
    
    def _handle_metrics_event(self, event: MetricsEvent):
        """Handle custom metrics events."""
        metric_name = event.metric_name
        
        if event.metric_type == 'counter':
            self.counters[metric_name] += int(event.metric_value)
        elif event.metric_type == 'gauge':
            self.gauges[metric_name] = float(event.metric_value)
        elif event.metric_type == 'timer':
            self.timers[metric_name].append(float(event.metric_value))
        elif event.metric_type == 'histogram' and self.enable_histograms:
            self.histograms[metric_name].append(float(event.metric_value))
    
    def _handle_cache_event(self, event: CacheEvent):
        """Handle cache events."""
        self.counters[f'cache_{event.operation}'] += 1
        
        if event.hit_rate is not None:
            self.gauges[f'cache_hit_rate_{event.cache_type}'] = event.hit_rate
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        summary = {
            'collection_duration': time.time() - self.start_time,
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'event_counts': dict(self.event_counts),
            'error_counts': dict(self.error_counts)
        }
        
        # Add timer statistics
        timer_stats = {}
        for name, times in self.timers.items():
            if times:
                timer_stats[name] = {
                    'count': len(times),
                    'total': sum(times),
                    'average': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times)
                }
        summary['timers'] = timer_stats
        
        # Add stage duration statistics
        stage_stats = {}
        for stage, durations in self.stage_durations.items():
            if durations:
                stage_stats[stage] = {
                    'count': len(durations),
                    'total_duration': sum(durations),
                    'average_duration': sum(durations) / len(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations)
                }
        summary['stage_performance'] = stage_stats
        
        return summary
    
    def reset_metrics(self):
        """Reset all collected metrics."""
        self.counters.clear()
        self.gauges.clear()
        self.timers.clear()
        self.histograms.clear()
        self.event_counts.clear()
        self.stage_durations.clear()
        self.error_counts.clear()
        self.start_time = time.time()


class CacheEventHandler(EventHandler):
    """Event handler that manages cache-related events and optimization."""
    
    def __init__(self, 
                 cache_service,
                 auto_optimize: bool = True,
                 optimization_threshold: float = 0.5):
        """
        Initialize cache event handler.
        
        Args:
            cache_service: Cache service instance to manage
            auto_optimize: Whether to automatically optimize cache based on events
            optimization_threshold: Hit rate threshold for optimization
        """
        self.cache_service = cache_service
        self.auto_optimize = auto_optimize
        self.optimization_threshold = optimization_threshold
        
        # Cache statistics
        self.cache_operations: Dict[str, int] = defaultdict(int)
        self.cache_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def handle(self, event: Event) -> None:
        """Handle cache-related events."""
        if isinstance(event, CacheEvent):
            self._handle_cache_event(event)
        elif isinstance(event, AnalysisEvent):
            self._handle_analysis_event(event)
    
    def _handle_cache_event(self, event: CacheEvent):
        """Process cache operation events."""
        operation = event.operation
        self.cache_operations[operation] += 1
        
        if event.hit_rate is not None:
            self.cache_performance[event.cache_type].append(event.hit_rate)
            
            # Auto-optimize if hit rate is low
            if (self.auto_optimize and 
                event.hit_rate < self.optimization_threshold and
                operation in ['miss', 'evict']):
                self._optimize_cache(event.cache_type, event.hit_rate)
    
    def _handle_analysis_event(self, event: AnalysisEvent):
        """Handle analysis events for cache optimization."""
        # Cache analysis results based on event data
        if event.success and hasattr(event, 'duration_seconds'):
            cache_key = f"analysis_{event.stage}_{hash(event.project_path)}"
            
            # Cache stage completion for future optimizations
            cache_data = {
                'stage': event.stage,
                'duration': event.duration_seconds,
                'timestamp': event.timestamp
            }
            
            try:
                self.cache_service.put(cache_key, cache_data, ttl=3600)
            except Exception as e:
                logger.warning(f"Failed to cache analysis result: {e}")
    
    def _optimize_cache(self, cache_type: str, current_hit_rate: float):
        """Optimize cache settings based on performance."""
        logger.info(f"Optimizing cache {cache_type} (hit rate: {current_hit_rate:.2%})")
        
        # Simple optimization strategy - could be enhanced
        try:
            stats = self.cache_service.get_statistics()
            utilization = stats.get('utilization', 0)
            
            if utilization > 0.9:  # Cache is full
                # Trigger manual cleanup of expired entries
                logger.info("Cache utilization high, triggering cleanup")
                # This would depend on cache service implementation
                
        except Exception as e:
            logger.warning(f"Cache optimization failed: {e}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache-related statistics."""
        stats = {
            'operations': dict(self.cache_operations),
            'performance': {}
        }
        
        # Calculate performance statistics
        for cache_type, hit_rates in self.cache_performance.items():
            if hit_rates:
                stats['performance'][cache_type] = {
                    'average_hit_rate': sum(hit_rates) / len(hit_rates),
                    'min_hit_rate': min(hit_rates),
                    'max_hit_rate': max(hit_rates),
                    'measurements': len(hit_rates)
                }
        
        return stats


class FileEventHandler(EventHandler):
    """Event handler that writes events to files for audit trails and analysis."""
    
    def __init__(self, 
                 output_directory: str,
                 file_format: str = "jsonl",
                 rotate_size_mb: int = 100,
                 max_files: int = 10):
        """
        Initialize file event handler.
        
        Args:
            output_directory: Directory to write event files
            file_format: Format for event files (jsonl, json, csv)
            rotate_size_mb: Size threshold for file rotation
            max_files: Maximum number of files to keep
        """
        self.output_directory = Path(output_directory)
        self.file_format = file_format
        self.rotate_size_mb = rotate_size_mb
        self.max_files = max_files
        
        # Ensure output directory exists
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Current file info
        self.current_file: Optional[Path] = None
        self.current_file_handle = None
        
        self._open_new_file()
    
    def handle(self, event: Event) -> None:
        """Write event to file."""
        try:
            if self._should_rotate_file():
                self._rotate_file()
            
            if self.file_format == "jsonl":
                self._write_jsonl(event)
            elif self.file_format == "json":
                self._write_json(event)
            else:
                logger.warning(f"Unsupported file format: {self.file_format}")
        
        except Exception as e:
            logger.error(f"Failed to write event to file: {e}")
    
    def _write_jsonl(self, event: Event):
        """Write event as JSON lines."""
        event_data = event.to_dict()
        self.current_file_handle.write(json.dumps(event_data) + '\n')
        self.current_file_handle.flush()
    
    def _write_json(self, event: Event):
        """Write event as JSON (append to array)."""
        # This is a simplified implementation
        event_data = event.to_dict()
        self.current_file_handle.write(json.dumps(event_data, indent=2) + '\n')
        self.current_file_handle.flush()
    
    def _should_rotate_file(self) -> bool:
        """Check if current file should be rotated."""
        if not self.current_file or not self.current_file.exists():
            return True
        
        size_mb = self.current_file.stat().st_size / (1024 * 1024)
        return size_mb >= self.rotate_size_mb
    
    def _rotate_file(self):
        """Rotate to a new file."""
        if self.current_file_handle:
            self.current_file_handle.close()
        
        self._cleanup_old_files()
        self._open_new_file()
    
    def _open_new_file(self):
        """Open a new event file."""
        timestamp = int(time.time())
        filename = f"events_{timestamp}.{self.file_format}"
        self.current_file = self.output_directory / filename
        
        self.current_file_handle = open(self.current_file, 'w')
        logger.info(f"Opened new event file: {self.current_file}")
    
    def _cleanup_old_files(self):
        """Remove old event files beyond max_files limit."""
        event_files = list(self.output_directory.glob(f"events_*.{self.file_format}"))
        
        if len(event_files) >= self.max_files:
            # Sort by modification time and remove oldest
            event_files.sort(key=lambda f: f.stat().st_mtime)
            files_to_remove = event_files[:-self.max_files + 1]
            
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                    logger.debug(f"Removed old event file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove old event file {file_path}: {e}")
    
    def close(self):
        """Close current file handle."""
        if self.current_file_handle:
            self.current_file_handle.close()
            self.current_file_handle = None