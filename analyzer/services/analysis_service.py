#!/usr/bin/env python3
"""
Analysis Service for Code Architecture Analyzer

Orchestrates the complete analysis workflow using Command pattern and service injection.
Replaces the monolithic DeterministicAnalyzer.analyze method with modular commands.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Protocol
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod

from ..core.ast_parser import ASTParseResult
from ..core.project_discoverer import ProjectStructure, ProjectDiscoverer
from ..core.analyzer import DeterministicResult, AnalysisStats
from .cache_service import CacheService
from ..events import EventBus, LoggingEventHandler, MetricsEventHandler
from ..events.events import (
    create_project_discovered_event,
    create_file_parse_event,
    create_actor_detected_event,
    create_graph_generated_event,
    ErrorEvent,
    ProgressEvent
)

logger = logging.getLogger(__name__)


class AnalysisCommand(ABC):
    """Abstract base class for analysis commands."""
    
    @abstractmethod
    def execute(self, context: 'AnalysisContext') -> bool:
        """Execute the command and return success status."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get command name for logging."""
        pass


@dataclass
class AnalysisContext:
    """Context object passed between analysis commands."""
    project_path: str
    result: DeterministicResult
    config: Dict[str, Any]
    services: Dict[str, Any]
    
    # Analysis artifacts
    project_structure: Optional[ProjectStructure] = None
    ast_results: List[ASTParseResult] = None
    actor_results: Optional[List[Any]] = None
    graph_data: Optional[Any] = None


class ProjectDiscoveryCommand(AnalysisCommand):
    """Command for project structure discovery."""
    
    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service
    
    def execute(self, context: AnalysisContext) -> bool:
        """Execute project discovery."""
        logger.info("Stage 1: Project discovery")
        
        try:
            # Check cache first
            cache_key = f"project_discovery:{context.project_path}"
            cached_structure = self.cache_service.get(cache_key)
            
            if cached_structure:
                logger.debug("Using cached project structure")
                context.project_structure = cached_structure
            else:
                # Discover project structure
                discoverer = ProjectDiscoverer(context.config)
                context.project_structure = discoverer.discover_project(context.project_path)
                
                # Cache the result
                self.cache_service.put(cache_key, context.project_structure, ttl=3600)
            
            # Update result
            context.result.project_structure = context.project_structure
            context.result.stats.files_discovered = context.project_structure.total_files
            context.result.errors.extend(context.project_structure.errors)
            
            # Validate we found Python files
            if not context.project_structure.python_files:
                logger.error("No Python files found in project")
                context.result.errors.append("No Python files found in project")
                return False
            
            logger.info(f"Discovered {context.project_structure.total_files} Python files, "
                       f"{context.project_structure.total_lines} lines of code")
            
            return True
            
        except Exception as e:
            error_msg = f"Project discovery failed: {e}"
            logger.error(error_msg)
            context.result.errors.append(error_msg)
            return False
    
    def get_name(self) -> str:
        return "ProjectDiscovery"


class ASTParsingCommand(AnalysisCommand):
    """Command for AST parsing."""
    
    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service
    
    def execute(self, context: AnalysisContext) -> bool:
        """Execute AST parsing."""
        logger.info("Stage 2: AST parsing")
        
        try:
            from ..core.ast_parser import ASTParser
            
            ast_parser = ASTParser(context.config)
            context.ast_results = []
            
            files_to_parse = context.project_structure.python_files
            
            for i, project_file in enumerate(files_to_parse):
                try:
                    # Check cache
                    cache_key = f"ast_parse:{project_file.path}:{project_file.size_bytes}"
                    cached_ast = self.cache_service.get(cache_key)
                    
                    if cached_ast:
                        logger.debug(f"Using cached AST for {project_file.relative_path}")
                        ast_result = cached_ast
                    else:
                        logger.debug(f"Parsing file {i+1}/{len(files_to_parse)}: {project_file.relative_path}")
                        
                        ast_result = ast_parser.parse_file(
                            project_file.path, 
                            project_file.module_name
                        )
                        
                        # Cache the result
                        if ast_result:
                            self.cache_service.put(cache_key, ast_result, ttl=1800)
                    
                    if ast_result:
                        context.ast_results.append(ast_result)
                        
                        # Update statistics
                        context.result.stats.files_parsed += 1
                        context.result.stats.functions_found += len(ast_result.functions)
                        context.result.stats.classes_found += len(ast_result.classes)
                        context.result.stats.imports_found += len(ast_result.imports)
                        
                except Exception as e:
                    logger.warning(f"Failed to parse {project_file.path}: {e}")
                    context.result.stats.parse_errors += 1
                    context.result.errors.append(f"Parse error in {project_file.path}: {e}")
            
            # Update result
            context.result.ast_results = context.ast_results
            
            # Validate we parsed some files
            if not context.ast_results:
                logger.error("No files could be parsed successfully")
                context.result.errors.append("No files could be parsed successfully")
                return False
            
            logger.info(f"Parsed {len(context.ast_results)} files successfully")
            return True
            
        except Exception as e:
            error_msg = f"AST parsing failed: {e}"
            logger.error(error_msg)
            context.result.errors.append(error_msg)
            return False
    
    def get_name(self) -> str:
        return "ASTParsing"


class ActorDetectionCommand(AnalysisCommand):
    """Command for actor detection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def execute(self, context: AnalysisContext) -> bool:
        """Execute actor detection."""
        logger.info("Stage 3: Actor detection")
        
        try:
            from ..detection.pattern_matcher import PatternMatcher
            
            pattern_matcher = PatternMatcher(self.config)
            context.actor_results = []
            
            for ast_result in context.ast_results:
                try:
                    actor_result = pattern_matcher.detect_actors(ast_result)
                    context.actor_results.append(actor_result)
                except Exception as e:
                    logger.warning(f"Actor detection failed for {ast_result.module_name}: {e}")
            
            # Update result
            context.result.actor_results = context.actor_results
            
            # Update statistics
            if context.actor_results:
                total_actors = sum(len(getattr(result, 'detected_actors', [])) 
                                 for result in context.actor_results)
                context.result.stats.actors_detected = total_actors
            
            logger.info(f"Detected actors in {len(context.actor_results or [])} modules")
            return True
            
        except Exception as e:
            error_msg = f"Actor detection failed: {e}"
            logger.error(error_msg)
            context.result.errors.append(error_msg)
            return False
    
    def get_name(self) -> str:
        return "ActorDetection"


class GraphGenerationCommand(AnalysisCommand):
    """Command for graph generation."""
    
    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service
    
    def execute(self, context: AnalysisContext) -> bool:
        """Execute graph generation."""
        logger.info("Stage 4: Graph generation")
        
        try:
            from ..core.node_generator import NodeGenerator
            
            # Check cache
            cache_key = f"graph:{context.project_path}:{len(context.ast_results)}"
            cached_graph = self.cache_service.get(cache_key)
            
            if cached_graph:
                logger.debug("Using cached graph data")
                context.graph_data = cached_graph
            else:
                node_generator = NodeGenerator(context.config)
                context.graph_data = node_generator.generate_graph(
                    context.project_structure,
                    context.ast_results,
                    context.actor_results
                )
                
                # Cache the result
                if context.graph_data:
                    self.cache_service.put(cache_key, context.graph_data, ttl=1800)
            
            # Update result
            context.result.graph_data = context.graph_data
            
            # Validate we generated nodes
            if not context.graph_data or not context.graph_data.get('graph', {}).get('nodes'):
                logger.error("No nodes could be generated")
                context.result.errors.append("No nodes could be generated")
                return False
            
            # Update statistics
            graph = context.graph_data.get('graph', {})
            context.result.stats.nodes_generated = len(graph.get('nodes', []))
            context.result.stats.relationships_generated = len(graph.get('relationships', []))
            
            logger.info(f"Generated {context.result.stats.nodes_generated} nodes and "
                       f"{context.result.stats.relationships_generated} relationships")
            return True
            
        except Exception as e:
            error_msg = f"Graph generation failed: {e}"
            logger.error(error_msg)
            context.result.errors.append(error_msg)
            return False
    
    def get_name(self) -> str:
        return "GraphGeneration"


class StatisticsCommand(AnalysisCommand):
    """Command for calculating final statistics."""
    
    def execute(self, context: AnalysisContext) -> bool:
        """Calculate final statistics."""
        logger.info("Stage 5: Statistics calculation")
        
        try:
            # Additional statistics can be calculated here
            context.result.stats.warnings = len([e for e in context.result.errors if 'warning' in e.lower()])
            
            return True
            
        except Exception as e:
            error_msg = f"Statistics calculation failed: {e}"
            logger.error(error_msg)
            context.result.errors.append(error_msg)
            return False
    
    def get_name(self) -> str:
        return "Statistics"


class AnalysisService:
    """
    Service that orchestrates the complete analysis workflow.
    
    Uses the Command pattern to break down the monolithic analyze method
    into discrete, testable, and cacheable commands.
    """
    
    def __init__(self, 
                 cache_service: CacheService,
                 config: Optional[Dict[str, Any]] = None,
                 event_bus: Optional[EventBus] = None):
        """Initialize analysis service with dependencies."""
        self.cache_service = cache_service
        self.config = config or {}
        
        # Event-driven architecture
        events_config = self.config.get('events', {})
        self.enable_events = events_config.get('enabled', True)
        
        if self.enable_events:
            self.event_bus = event_bus or EventBus()
            self._setup_event_handlers()
            logger.info("Event-driven processing enabled")
        else:
            self.event_bus = None
        
        # Analysis configuration
        analyzers_config = self.config.get('analyzers', {})
        self.enable_actor_detection = analyzers_config.get('enable_actor_detection', True)
        self.continue_on_error = self.config.get('error_handling', {}).get('continue_on_error', True)
        
        # Build command pipeline
        self.commands = self._build_command_pipeline()
        
        logger.info("Analysis service initialized with command pipeline")
    
    def _setup_event_handlers(self):
        """Setup default event handlers for reactive processing."""
        try:
            # Setup logging handler
            log_level = logging.DEBUG if self.config.get('debug', False) else logging.INFO
            logging_handler = LoggingEventHandler(
                log_level=log_level,
                include_metadata=True,
                include_event_data=False
            )
            
            from ..events.events import AnalysisEvent, ErrorEvent
            self.event_bus.subscribe(
                [AnalysisEvent, ErrorEvent], 
                logging_handler.handle,
                priority=100  # High priority for logging
            )
            
            # Setup metrics handler
            metrics_handler = MetricsEventHandler()
            self.event_bus.subscribe(
                AnalysisEvent,
                metrics_handler.handle,
                priority=50
            )
            
            # Store handlers for later access
            self.logging_handler = logging_handler
            self.metrics_handler = metrics_handler
            
            logger.debug("Event handlers configured")
            
        except Exception as e:
            logger.warning(f"Failed to setup event handlers: {e}")
    
    def _publish_event(self, event):
        """Publish an event if event bus is enabled."""
        if self.enable_events and self.event_bus:
            try:
                self.event_bus.publish(event)
            except Exception as e:
                logger.warning(f"Failed to publish event: {e}")
    
    def get_metrics_summary(self) -> Optional[Dict[str, Any]]:
        """Get analysis metrics from event processing."""
        if hasattr(self, 'metrics_handler'):
            return self.metrics_handler.get_metrics_summary()
        return None
    
    def _publish_command_events(self, command, context: AnalysisContext, duration: float, correlation_id: str):
        """Publish events specific to command completion."""
        try:
            command_name = command.get_name()
            
            if command_name == "ProjectDiscovery" and context.project_structure:
                event = create_project_discovered_event(
                    context.project_structure, 
                    duration, 
                    correlation_id
                )
                self._publish_event(event)
                
            elif command_name == "GraphGeneration" and context.graph_data:
                event = create_graph_generated_event(
                    context.graph_data,
                    duration,
                    correlation_id
                )
                self._publish_event(event)
                
            # For AST parsing and actor detection, we'd need to publish per-file events
            # This is a simplified version that publishes aggregate events
            
        except Exception as e:
            logger.warning(f"Failed to publish command events: {e}")
    
    def analyze(self, project_path: str) -> DeterministicResult:
        """
        Perform complete analysis using command pipeline.
        
        Args:
            project_path: Path to the Python project root
            
        Returns:
            DeterministicResult with analysis results and metadata
        """
        start_time = time.time()
        project_path = str(Path(project_path).resolve())
        
        logger.info(f"Starting analysis of: {project_path}")
        
        # Initialize result and context
        result = DeterministicResult(
            project_structure=ProjectStructure(root_path=Path(project_path))
        )
        
        context = AnalysisContext(
            project_path=project_path,
            result=result,
            config=self.config,
            services={
                'cache': self.cache_service
            }
        )
        
        # Execute command pipeline
        correlation_id = f"analysis_{int(time.time() * 1000000)}"
        
        for i, command in enumerate(self.commands):
            command_start = time.time()
            
            try:
                logger.debug(f"Executing command: {command.get_name()}")
                
                # Publish progress event
                if self.enable_events:
                    progress_event = ProgressEvent(
                        stage=command.get_name(),
                        current_step=i + 1,
                        total_steps=len(self.commands),
                        progress_percentage=(i / len(self.commands)) * 100,
                        current_operation=f"Executing {command.get_name()}",
                        correlation_id=correlation_id,
                        source="analysis_service"
                    )
                    self._publish_event(progress_event)
                
                success = command.execute(context)
                command_duration = time.time() - command_start
                
                # Publish command completion events based on type
                if success and self.enable_events:
                    self._publish_command_events(command, context, command_duration, correlation_id)
                
                if not success:
                    # Publish error event
                    if self.enable_events:
                        error_event = ErrorEvent(
                            error_type="CommandFailure",
                            error_message=f"Command {command.get_name()} failed",
                            component="analysis_service",
                            stage=command.get_name(),
                            recoverable=self.continue_on_error,
                            correlation_id=correlation_id,
                            source="analysis_service"
                        )
                        self._publish_event(error_event)
                    
                    if not self.continue_on_error:
                        logger.error(f"Command {command.get_name()} failed, stopping analysis")
                        break
                    else:
                        logger.warning(f"Command {command.get_name()} failed, continuing with next command")
                
            except Exception as e:
                command_duration = time.time() - command_start
                error_msg = f"Command {command.get_name()} failed: {e}"
                logger.error(error_msg, exc_info=True)
                context.result.errors.append(error_msg)
                
                # Publish error event
                if self.enable_events:
                    error_event = ErrorEvent(
                        error_type=type(e).__name__,
                        error_message=str(e),
                        component="analysis_service",
                        stage=command.get_name(),
                        recoverable=self.continue_on_error,
                        stack_trace=str(e),
                        correlation_id=correlation_id,
                        source="analysis_service"
                    )
                    self._publish_event(error_event)
                
                if not self.continue_on_error:
                    break
        
        # Finalize result
        context.result.stats.analysis_time_seconds = time.time() - start_time
        
        logger.info(f"Analysis completed in {context.result.stats.analysis_time_seconds:.2f}s")
        if context.result.errors:
            logger.warning(f"Analysis completed with {len(context.result.errors)} errors")
        
        return context.result
    
    def _build_command_pipeline(self) -> List[AnalysisCommand]:
        """Build the command pipeline based on configuration."""
        commands = []
        
        analyzers_config = self.config.get('analyzers', {})
        
        # Project discovery (always enabled)
        if analyzers_config.get('project_discovery', True):
            commands.append(ProjectDiscoveryCommand(self.cache_service))
        
        # AST parsing
        if analyzers_config.get('ast_parser', True):
            commands.append(ASTParsingCommand(self.cache_service))
        
        # Actor detection (optional)
        if self.enable_actor_detection:
            commands.append(ActorDetectionCommand(self.config))
        
        # Graph generation
        if analyzers_config.get('node_generator', True):
            commands.append(GraphGenerationCommand(self.cache_service))
        
        # Statistics (always enabled)
        commands.append(StatisticsCommand())
        
        return commands