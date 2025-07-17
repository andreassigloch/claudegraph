#!/usr/bin/env python3
"""
Event Definitions for Code Architecture Analyzer

Defines the event types and data structures used throughout the analysis
process to enable reactive and decoupled processing.
"""

import time
import uuid
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod


@dataclass
class Event(ABC):
    """Base class for all events in the analysis system."""
    
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    event_type: str = field(init=False)
    source: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set event type based on class name."""
        self.event_type = self.__class__.__name__
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'timestamp': self.timestamp,
            'source': self.source,
            'correlation_id': self.correlation_id,
            'metadata': self.metadata,
            'data': self._get_event_data()
        }
    
    @abstractmethod
    def _get_event_data(self) -> Dict[str, Any]:
        """Get event-specific data."""
        pass


@dataclass
class AnalysisEvent(Event):
    """Base class for analysis-related events."""
    
    project_path: str = ""
    stage: str = ""
    success: bool = True
    duration_seconds: Optional[float] = None
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'project_path': self.project_path,
            'stage': self.stage,
            'success': self.success,
            'duration_seconds': self.duration_seconds
        }


@dataclass
class ProjectDiscoveredEvent(AnalysisEvent):
    """Event fired when project structure is discovered."""
    
    total_files: int = 0
    python_files: int = 0
    total_lines: int = 0
    total_size_bytes: int = 0
    directories: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        super().__post_init__()
        self.stage = "project_discovery"
    
    def _get_event_data(self) -> Dict[str, Any]:
        data = super()._get_event_data()
        data.update({
            'total_files': self.total_files,
            'python_files': self.python_files,
            'total_lines': self.total_lines,
            'total_size_bytes': self.total_size_bytes,
            'directories': self.directories,
            'errors': self.errors
        })
        return data


@dataclass
class FileParseEvent(AnalysisEvent):
    """Event fired when a file is parsed."""
    
    file_path: str = ""
    module_name: str = ""
    functions_found: int = 0
    classes_found: int = 0
    imports_found: int = 0
    lines_of_code: int = 0
    parse_errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        super().__post_init__()
        self.stage = "ast_parsing"
    
    def _get_event_data(self) -> Dict[str, Any]:
        data = super()._get_event_data()
        data.update({
            'file_path': self.file_path,
            'module_name': self.module_name,
            'functions_found': self.functions_found,
            'classes_found': self.classes_found,
            'imports_found': self.imports_found,
            'lines_of_code': self.lines_of_code,
            'parse_errors': self.parse_errors
        })
        return data


@dataclass
class ActorDetectedEvent(AnalysisEvent):
    """Event fired when actors are detected in a module."""
    
    module_name: str = ""
    actors_detected: int = 0
    high_confidence_actors: int = 0
    ambiguous_actors: int = 0
    actor_types: List[str] = field(default_factory=list)
    detection_method: str = "pattern_matching"
    
    def __post_init__(self):
        super().__post_init__()
        self.stage = "actor_detection"
    
    def _get_event_data(self) -> Dict[str, Any]:
        data = super()._get_event_data()
        data.update({
            'module_name': self.module_name,
            'actors_detected': self.actors_detected,
            'high_confidence_actors': self.high_confidence_actors,
            'ambiguous_actors': self.ambiguous_actors,
            'actor_types': self.actor_types,
            'detection_method': self.detection_method
        })
        return data


@dataclass
class GraphGeneratedEvent(AnalysisEvent):
    """Event fired when graph generation is completed."""
    
    nodes_generated: int = 0
    relationships_generated: int = 0
    flow_relationships: int = 0
    actor_nodes: int = 0
    function_nodes: int = 0
    module_nodes: int = 0
    system_nodes: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        self.stage = "graph_generation"
    
    def _get_event_data(self) -> Dict[str, Any]:
        data = super()._get_event_data()
        data.update({
            'nodes_generated': self.nodes_generated,
            'relationships_generated': self.relationships_generated,
            'flow_relationships': self.flow_relationships,
            'actor_nodes': self.actor_nodes,
            'function_nodes': self.function_nodes,
            'module_nodes': self.module_nodes,
            'system_nodes': self.system_nodes
        })
        return data


@dataclass
class CacheEvent(Event):
    """Event fired for cache operations."""
    
    operation: str = ""  # hit, miss, put, delete, evict
    cache_key: str = ""
    cache_type: str = ""
    hit_rate: Optional[float] = None
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'operation': self.operation,
            'cache_key': self.cache_key,
            'cache_type': self.cache_type,
            'hit_rate': self.hit_rate
        }


@dataclass
class ErrorEvent(Event):
    """Event fired when errors occur during analysis."""
    
    error_type: str = ""
    error_message: str = ""
    component: str = ""
    stage: str = ""
    file_path: Optional[str] = None
    recoverable: bool = True
    stack_trace: Optional[str] = None
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'error_type': self.error_type,
            'error_message': self.error_message,
            'component': self.component,
            'stage': self.stage,
            'file_path': self.file_path,
            'recoverable': self.recoverable,
            'stack_trace': self.stack_trace
        }


@dataclass
class ProgressEvent(Event):
    """Event fired to report analysis progress."""
    
    stage: str = ""
    current_step: int = 0
    total_steps: int = 0
    progress_percentage: float = 0.0
    estimated_remaining_seconds: Optional[float] = None
    current_operation: str = ""
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'stage': self.stage,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'progress_percentage': self.progress_percentage,
            'estimated_remaining_seconds': self.estimated_remaining_seconds,
            'current_operation': self.current_operation
        }


@dataclass
class MetricsEvent(Event):
    """Event fired to report performance metrics."""
    
    metric_name: str = ""
    metric_value: Union[int, float, str] = 0
    metric_type: str = "counter"  # counter, gauge, histogram, timer
    tags: Dict[str, str] = field(default_factory=dict)
    
    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'metric_type': self.metric_type,
            'tags': self.tags
        }


# Event factory functions for convenience
def create_project_discovered_event(project_structure, duration: Optional[float] = None, 
                                   correlation_id: Optional[str] = None) -> ProjectDiscoveredEvent:
    """Create a ProjectDiscoveredEvent from project structure."""
    return ProjectDiscoveredEvent(
        project_path=str(project_structure.root_path),
        total_files=project_structure.total_files,
        python_files=len(project_structure.python_files),
        total_lines=project_structure.total_lines,
        total_size_bytes=project_structure.total_size_bytes,
        directories=[str(d) for d in project_structure.directories],
        errors=project_structure.errors,
        duration_seconds=duration,
        correlation_id=correlation_id,
        source="project_discoverer"
    )


def create_file_parse_event(ast_result, duration: Optional[float] = None,
                           correlation_id: Optional[str] = None) -> FileParseEvent:
    """Create a FileParseEvent from AST result."""
    return FileParseEvent(
        project_path="",  # Would need to be passed in
        file_path=ast_result.file_path,
        module_name=ast_result.module_name,
        functions_found=len(ast_result.functions),
        classes_found=len(ast_result.classes),
        imports_found=len(ast_result.imports),
        lines_of_code=getattr(ast_result, 'lines_of_code', 0),
        parse_errors=ast_result.errors,
        duration_seconds=duration,
        correlation_id=correlation_id,
        source="ast_parser"
    )


def create_actor_detected_event(actor_result, duration: Optional[float] = None,
                               correlation_id: Optional[str] = None) -> ActorDetectedEvent:
    """Create an ActorDetectedEvent from actor detection result."""
    actor_types = list(set(actor.actor_type for actor in actor_result.detected_actors))
    
    return ActorDetectedEvent(
        project_path="",  # Would need to be passed in
        module_name=actor_result.module_name,
        actors_detected=len(actor_result.detected_actors),
        high_confidence_actors=len(actor_result.high_confidence_matches),
        ambiguous_actors=len(actor_result.ambiguous_matches),
        actor_types=actor_types,
        duration_seconds=duration,
        correlation_id=correlation_id,
        source="actor_detector"
    )


def create_graph_generated_event(graph_data, duration: Optional[float] = None,
                                correlation_id: Optional[str] = None) -> GraphGeneratedEvent:
    """Create a GraphGeneratedEvent from graph data."""
    # Count different node types
    node_counts = {}
    for node in graph_data.nodes:
        node_type = node.type
        node_counts[node_type] = node_counts.get(node_type, 0) + 1
    
    # Count flow relationships
    flow_relationships = len([rel for rel in graph_data.relationships if rel.type == 'flow'])
    
    return GraphGeneratedEvent(
        project_path="",  # Would need to be passed in
        nodes_generated=len(graph_data.nodes),
        relationships_generated=len(graph_data.relationships),
        flow_relationships=flow_relationships,
        actor_nodes=node_counts.get('ACTOR', 0),
        function_nodes=node_counts.get('FUNC', 0),
        module_nodes=node_counts.get('MOD', 0),
        system_nodes=node_counts.get('SYS', 0),
        duration_seconds=duration,
        correlation_id=correlation_id,
        source="node_generator"
    )