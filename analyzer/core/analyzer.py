#!/usr/bin/env python3
"""
Deterministic Analyzer for Code Architecture Analyzer

Coordinates all core components to perform deterministic analysis of Python projects.
Extracts project structure, parses AST, and generates initial ontology graph.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .project_discoverer import ProjectDiscoverer, ProjectStructure
from .ast_parser import ASTParser, ASTParseResult
from .node_generator import NodeGenerator, GraphData, OntologyNode, OntologyRelationship
from ..detection.pattern_matcher import PatternMatcher, ActorDetectionResult
from ..llm.actor_enhancer import ActorEnhancementService
from ..llm.client import LLMManager

logger = logging.getLogger(__name__)


@dataclass
class AnalysisStats:
    """Statistics from deterministic analysis."""
    files_discovered: int = 0
    files_parsed: int = 0
    functions_found: int = 0
    classes_found: int = 0
    imports_found: int = 0
    actors_detected: int = 0
    high_confidence_actors: int = 0
    ambiguous_actors: int = 0
    nodes_generated: int = 0
    relationships_generated: int = 0
    flow_relationships_generated: int = 0
    fchain_nodes_generated: int = 0
    parse_errors: int = 0
    warnings: int = 0
    analysis_time_seconds: float = 0.0


@dataclass
class DeterministicResult:
    """Complete result from deterministic analysis."""
    project_structure: ProjectStructure
    ast_results: List[ASTParseResult] = field(default_factory=list)
    actor_results: List[ActorDetectionResult] = field(default_factory=list)
    graph_data: Optional[GraphData] = None
    stats: AnalysisStats = field(default_factory=AnalysisStats)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_successful(self) -> bool:
        """Check if analysis was successful."""
        return (
            self.project_structure.total_files > 0 and
            len(self.ast_results) > 0 and
            self.graph_data is not None and
            len(self.graph_data.nodes) > 0
        )
    
    def get_total_errors(self) -> int:
        """Get total number of errors across all components."""
        total = len(self.errors)
        total += len(self.project_structure.errors)
        for ast_result in self.ast_results:
            total += len(ast_result.errors)
        if self.graph_data:
            total += len(self.graph_data.errors)
        return total


class DeterministicAnalyzer:
    """
    Main deterministic analyzer that coordinates all core components.
    
    Performs project discovery, AST parsing, and initial graph generation
    without requiring external AI or user input.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize deterministic analyzer with configuration."""
        self.config = config or {}
        
        # Check if service-oriented architecture is enabled
        service_config = self.config.get('services', {})
        self.use_service_architecture = service_config.get('enabled', False)
        
        # Initialize LLM client for enhancement service
        self.llm_client = None
        enhancement_config = self.config.get('llm', {}).get('actor_enhancement', {})
        if enhancement_config.get('enabled', True):
            try:
                self.llm_client = LLMManager(self.config).get_client()
                logger.info("LLM client initialized for actor enhancement")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client: {e}")
        
        # Initialize enhancement service
        self.enhancement_service = None
        if self.llm_client and enhancement_config.get('enabled', True):
            try:
                self.enhancement_service = ActorEnhancementService(self.config, self.llm_client)
                logger.info("Actor enhancement service initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize enhancement service: {e}")
        
        # Initialize service layer if enabled
        self.analysis_service = None
        if self.use_service_architecture:
            try:
                from ..services import AnalysisService, CacheService
                cache_service = CacheService(config=self.config)
                self.analysis_service = AnalysisService(cache_service, self.config)
                logger.info("Service-oriented architecture enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize service architecture: {e}")
                self.use_service_architecture = False
        
        # Initialize traditional components (used when service architecture is disabled)
        self.project_discoverer = ProjectDiscoverer(config)
        self.ast_parser = ASTParser(config)
        self.pattern_matcher = PatternMatcher(config, self.enhancement_service)
        self.node_generator = NodeGenerator(config, self.llm_client)
        
        # Analysis settings
        deterministic_config = self.config.get('deterministic', {})
        self.confidence_threshold = deterministic_config.get('confidence_threshold', 0.8)
        self.analyzers_config = deterministic_config.get('analyzers', {})
        
        # Actor detection settings
        self.enable_actor_detection = self.analyzers_config.get('pattern_matcher', True)
        
        # Performance settings
        performance_config = self.config.get('performance', {})
        self.max_workers = performance_config.get('max_workers', 4)
        self.chunk_size = performance_config.get('chunk_size', 1000)
        
        # Error handling
        error_config = self.config.get('error_handling', {})
        self.continue_on_error = error_config.get('continue_on_error', True)
        self.partial_results = error_config.get('partial_results', True)
        
        logger.info("Deterministic analyzer initialized")
    
    def analyze(self, project_path: str) -> DeterministicResult:
        """
        Perform complete deterministic analysis of a Python project.
        
        Args:
            project_path: Path to the Python project root
            
        Returns:
            DeterministicResult with analysis results and metadata
        """
        # Use service architecture if enabled
        if self.use_service_architecture and self.analysis_service:
            logger.info("Using service-oriented analysis architecture")
            return self.analysis_service.analyze(project_path)
        
        # Traditional monolithic analysis
        start_time = time.time()
        project_path = str(Path(project_path).resolve())
        
        logger.info(f"Starting deterministic analysis of: {project_path}")
        
        result = DeterministicResult(
            project_structure=ProjectStructure(root_path=Path(project_path))
        )
        
        try:
            # Stage 1: Project Discovery
            if self.analyzers_config.get('project_discovery', True):
                result.project_structure = self._discover_project(project_path, result)
                if not result.project_structure.python_files:
                    error_msg = "No Python files found in project"
                    logger.error(error_msg)
                    result.errors.append(error_msg)
                    return result
            
            # Stage 2: AST Parsing
            if self.analyzers_config.get('ast_parser', True):
                result.ast_results = self._parse_ast(result.project_structure, result)
                if not result.ast_results:
                    error_msg = "No files could be parsed successfully"
                    logger.error(error_msg)
                    result.errors.append(error_msg)
                    return result
            
            # Stage 3: Actor Detection
            if self.enable_actor_detection:
                result.actor_results = self._detect_actors(result.ast_results, result)
            
            # Stage 4: Node Generation
            if self.analyzers_config.get('node_generator', True):
                result.graph_data = self._generate_graph(result.project_structure, result.ast_results, result.actor_results, result)
                if not result.graph_data or not result.graph_data.nodes:
                    error_msg = "No nodes could be generated"
                    logger.error(error_msg)
                    result.errors.append(error_msg)
                    return result
            
            # Stage 5: Statistics and Metadata
            self._calculate_statistics(result)
            self._generate_metadata(result, project_path)
            
            # Calculate analysis time
            result.stats.analysis_time_seconds = time.time() - start_time
            
            logger.info(f"Deterministic analysis completed successfully in {result.stats.analysis_time_seconds:.2f}s")
            logger.info(f"Generated {result.stats.nodes_generated} nodes and {result.stats.relationships_generated} relationships")
            
        except Exception as e:
            error_msg = f"Deterministic analysis failed: {e}"
            logger.error(error_msg, exc_info=True)
            result.errors.append(error_msg)
            result.stats.analysis_time_seconds = time.time() - start_time
        
        return result
    
    def _discover_project(self, project_path: str, result: DeterministicResult) -> ProjectStructure:
        """Perform project discovery stage."""
        logger.info("Stage 1: Project discovery")
        
        try:
            project_structure = self.project_discoverer.discover_project(project_path)
            
            # Update statistics
            result.stats.files_discovered = project_structure.total_files
            
            # Collect errors and warnings
            result.errors.extend(project_structure.errors)
            
            logger.info(f"Discovered {project_structure.total_files} Python files, "
                       f"{project_structure.total_lines} lines of code")
            
            return project_structure
            
        except Exception as e:
            error_msg = f"Project discovery failed: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            return ProjectStructure(root_path=Path(project_path))
    
    def _parse_ast(self, project_structure: ProjectStructure, result: DeterministicResult) -> List[ASTParseResult]:
        """Perform AST parsing stage."""
        logger.info("Stage 2: AST parsing")
        
        ast_results = []
        files_to_parse = project_structure.python_files
        
        try:
            # Process files
            for i, project_file in enumerate(files_to_parse):
                try:
                    logger.debug(f"Parsing file {i+1}/{len(files_to_parse)}: {project_file.relative_path}")
                    
                    ast_result = self.ast_parser.parse_file(
                        project_file.path, 
                        project_file.module_name
                    )
                    
                    if ast_result:
                        ast_results.append(ast_result)
                        
                        # Update statistics
                        result.stats.functions_found += len(ast_result.functions)
                        result.stats.classes_found += len(ast_result.classes)
                        result.stats.imports_found += len(ast_result.imports)
                        
                        # Collect errors and warnings
                        result.errors.extend(ast_result.errors)
                        result.warnings.extend(ast_result.warnings)
                        
                        if ast_result.errors:
                            result.stats.parse_errors += 1
                    
                except Exception as e:
                    error_msg = f"Failed to parse {project_file.path}: {e}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)
                    result.stats.parse_errors += 1
                    
                    if not self.continue_on_error:
                        break
            
            result.stats.files_parsed = len(ast_results)
            
            logger.info(f"Parsed {len(ast_results)} files successfully, "
                       f"found {result.stats.functions_found} functions and {result.stats.classes_found} classes")
            
        except Exception as e:
            error_msg = f"AST parsing stage failed: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
        
        return ast_results
    
    def _detect_actors(self, ast_results: List[ASTParseResult], result: DeterministicResult) -> List[ActorDetectionResult]:
        """Perform actor detection stage."""
        logger.info("Stage 3: Actor detection")
        
        actor_results = []
        
        try:
            for ast_result in ast_results:
                try:
                    logger.debug(f"Detecting actors in {ast_result.module_name}")
                    
                    actor_result = self.pattern_matcher.detect_actors(ast_result)
                    actor_results.append(actor_result)
                    
                    # Update statistics
                    result.stats.actors_detected += len(actor_result.detected_actors)
                    result.stats.high_confidence_actors += len(actor_result.high_confidence_matches)
                    result.stats.ambiguous_actors += len(actor_result.ambiguous_matches)
                    
                except Exception as e:
                    error_msg = f"Failed to detect actors in {ast_result.module_name}: {e}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)
                    
                    if not self.continue_on_error:
                        break
            
            logger.info(f"Actor detection completed: {result.stats.actors_detected} actors detected, "
                       f"{result.stats.high_confidence_actors} high confidence")
            
        except Exception as e:
            error_msg = f"Actor detection stage failed: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
        
        return actor_results
    
    def _generate_graph(self, project_structure: ProjectStructure, ast_results: List[ASTParseResult],
                       actor_results: List[ActorDetectionResult], result: DeterministicResult) -> Optional[GraphData]:
        """Perform graph generation stage."""
        logger.info("Stage 4: Graph generation")
        
        try:
            graph_data = self.node_generator.generate_graph(project_structure, ast_results, actor_results)
            
            if graph_data:
                # Update statistics
                result.stats.nodes_generated = len(graph_data.nodes)
                result.stats.relationships_generated = len(graph_data.relationships)
                
                # Collect errors and warnings
                result.errors.extend(graph_data.errors)
                result.warnings.extend(graph_data.warnings)
                
                logger.info(f"Generated {len(graph_data.nodes)} nodes and {len(graph_data.relationships)} relationships")
            
            return graph_data
            
        except Exception as e:
            error_msg = f"Graph generation failed: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            return None
    
    def _calculate_statistics(self, result: DeterministicResult) -> None:
        """Calculate comprehensive analysis statistics."""
        try:
            stats = result.stats
            
            # Count warnings
            stats.warnings = len(result.warnings)
            if result.project_structure:
                stats.warnings += len([e for e in result.project_structure.errors if 'warning' in e.lower()])
            
            for ast_result in result.ast_results:
                stats.warnings += len(ast_result.warnings)
            
            if result.graph_data:
                stats.warnings += len(result.graph_data.warnings)
                
                # Count flow relationships and FCHAIN nodes
                stats.flow_relationships_generated = len([
                    rel for rel in result.graph_data.relationships 
                    if rel.type == 'flow' and rel.properties.get('FlowDescr')
                ])
                
                stats.fchain_nodes_generated = len([
                    node for node in result.graph_data.nodes 
                    if node.type == 'FCHAIN'
                ])
            
            logger.debug(f"Analysis statistics calculated: {stats}")
            
        except Exception as e:
            logger.warning(f"Error calculating statistics: {e}")
    
    def _generate_metadata(self, result: DeterministicResult, project_path: str) -> None:
        """Generate comprehensive metadata for the analysis."""
        try:
            metadata = {
                'analysis_version': '1.0.0',
                'analyzer_type': 'deterministic',
                'timestamp': datetime.utcnow().isoformat(),
                'project_path': project_path,
                'project_name': Path(project_path).name,
                'configuration': {
                    'confidence_threshold': self.confidence_threshold,
                    'continue_on_error': self.continue_on_error,
                    'partial_results': self.partial_results,
                    'analyzers_enabled': self.analyzers_config
                },
                'statistics': {
                    'files_discovered': result.stats.files_discovered,
                    'files_parsed': result.stats.files_parsed,
                    'functions_found': result.stats.functions_found,
                    'classes_found': result.stats.classes_found,
                    'imports_found': result.stats.imports_found,
                    'actors_detected': result.stats.actors_detected,
                    'high_confidence_actors': result.stats.high_confidence_actors,
                    'ambiguous_actors': result.stats.ambiguous_actors,
                    'nodes_generated': result.stats.nodes_generated,
                    'relationships_generated': result.stats.relationships_generated,
                    'flow_relationships_generated': result.stats.flow_relationships_generated,
                    'fchain_nodes_generated': result.stats.fchain_nodes_generated,
                    'parse_errors': result.stats.parse_errors,
                    'warnings': result.stats.warnings,
                    'analysis_time_seconds': result.stats.analysis_time_seconds
                },
                'success_metrics': {
                    'discovery_success_rate': (result.stats.files_discovered / max(1, len(result.project_structure.python_files))) if result.project_structure else 0,
                    'parse_success_rate': (result.stats.files_parsed / max(1, result.stats.files_discovered)),
                    'actor_detection_success': result.stats.actors_detected > 0,
                    'node_generation_success': bool(result.graph_data and result.graph_data.nodes),
                    'overall_success': result.is_successful()
                }
            }
            
            # Add component-specific metadata
            if result.project_structure and result.project_structure.metadata:
                metadata['project_metadata'] = result.project_structure.metadata
            
            if result.actor_results:
                metadata['actor_detection_metadata'] = {
                    'pattern_matcher_stats': self.pattern_matcher.get_detection_statistics(),
                    'detection_summary': {
                        'modules_analyzed': len(result.actor_results),
                        'total_matches': sum(len(ar.detected_actors) for ar in result.actor_results),
                        'confidence_distribution': self._get_confidence_distribution(result.actor_results)
                    }
                }
            
            if result.graph_data and result.graph_data.metadata:
                metadata['graph_metadata'] = result.graph_data.metadata
            
            result.metadata = metadata
            
        except Exception as e:
            logger.warning(f"Error generating metadata: {e}")
            result.metadata = {'error': str(e)}
    
    def validate_project(self, project_path: str) -> Tuple[bool, List[str]]:
        """
        Validate that a project is suitable for deterministic analysis.
        
        Args:
            project_path: Path to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        try:
            return self.project_discoverer.validate_project_path(project_path)
        except Exception as e:
            return False, [f"Validation failed: {e}"]
    
    def get_analysis_preview(self, project_path: str) -> Dict[str, Any]:
        """
        Get a quick preview of what would be analyzed without full processing.
        
        Args:
            project_path: Path to preview
            
        Returns:
            Dictionary with preview information
        """
        try:
            # Quick project structure scan
            project_structure = self.project_discoverer.discover_project(project_path)
            
            preview = {
                'project_name': Path(project_path).name,
                'total_files': project_structure.total_files,
                'total_lines': project_structure.total_lines,
                'total_size_mb': round(project_structure.total_size_bytes / (1024 * 1024), 2),
                'directories': len(project_structure.directories),
                'file_types': {},
                'main_files': [],
                'test_files': [],
                'estimated_analysis_time': self._estimate_analysis_time(project_structure),
                'complexity_estimate': self._estimate_complexity(project_structure)
            }
            
            # Analyze file types and characteristics
            for pf in project_structure.python_files:
                ext = pf.path.suffix
                preview['file_types'][ext] = preview['file_types'].get(ext, 0) + 1
                
                if pf.is_main:
                    preview['main_files'].append(str(pf.relative_path))
                if pf.is_test:
                    preview['test_files'].append(str(pf.relative_path))
            
            return preview
            
        except Exception as e:
            return {'error': str(e)}
    
    def _estimate_analysis_time(self, project_structure: ProjectStructure) -> float:
        """Estimate analysis time based on project size."""
        # Rough estimate: ~0.1 seconds per 100 lines of code
        base_time = project_structure.total_lines * 0.001
        file_overhead = project_structure.total_files * 0.05
        return round(base_time + file_overhead, 1)
    
    def _estimate_complexity(self, project_structure: ProjectStructure) -> str:
        """Estimate project complexity."""
        if project_structure.total_lines < 1000:
            return "Low"
        elif project_structure.total_lines < 10000:
            return "Medium"
        elif project_structure.total_lines < 25000:
            return "High"
        else:
            return "Very High"
    
    def _get_confidence_distribution(self, actor_results: List[ActorDetectionResult]) -> Dict[str, int]:
        """Get confidence distribution from actor detection results."""
        distribution = {'high': 0, 'medium': 0, 'low': 0}
        
        for actor_result in actor_results:
            for match in actor_result.detected_actors:
                if match.confidence >= 0.8:
                    distribution['high'] += 1
                elif match.confidence >= 0.6:
                    distribution['medium'] += 1
                else:
                    distribution['low'] += 1
        
        return distribution