#!/usr/bin/env python3
"""
Analyzer Pipeline Module for Code Architecture Analyzer

Orchestrates the complete analysis pipeline from project discovery through
graph export, coordinating all analysis components.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from .core.project_discoverer import ProjectDiscoverer
from .core.ast_parser import ASTParser
from .core.pycg_integration import PyCGIntegration
from .detection.pattern_matcher import PatternMatcher
from .graph.builder import OntologyGraphBuilder, GraphBuildResult
from .llm.client import LLMManager

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""
    success: bool = False
    project_name: str = ""
    files_analyzed: int = 0
    functions_found: int = 0
    classes_found: int = 0
    imports_found: int = 0
    actors_detected: int = 0
    nodes_generated: int = 0
    relationships_generated: int = 0
    analysis_time: float = 0.0
    graph_result: Optional[GraphBuildResult] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Pipeline configuration settings."""
    # Analysis settings
    continue_on_error: bool = True
    partial_results: bool = True
    confidence_threshold: float = 0.8
    
    # Component toggles
    enable_pycg: bool = True
    enable_llm: bool = False
    enable_actor_detection: bool = True
    
    # Performance settings
    max_files: int = 1000
    timeout_seconds: int = 300
    
    # Output settings
    include_metadata: bool = True
    generate_statistics: bool = True


class AnalyzerPipeline:
    """
    Main analyzer pipeline that orchestrates the complete analysis workflow.
    
    Coordinates project discovery, AST parsing, actor detection, call graph
    analysis, and graph building into a unified analysis pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize analyzer pipeline with configuration."""
        self.config = config or {}
        self.pipeline_config = self._load_pipeline_config()
        
        # Initialize components
        self._initialize_components()
        
        # Pipeline state
        self.current_project: Optional[str] = None
        self.analysis_start_time: Optional[float] = None
        
        logger.info("Analyzer pipeline initialized")
    
    def _load_pipeline_config(self) -> PipelineConfig:
        """Load pipeline configuration from config dict."""
        pipeline_settings = self.config.get('pipeline', {})
        
        return PipelineConfig(
            continue_on_error=pipeline_settings.get('continue_on_error', True),
            partial_results=pipeline_settings.get('partial_results', True),
            confidence_threshold=pipeline_settings.get('confidence_threshold', 0.8),
            enable_pycg=pipeline_settings.get('enable_pycg', True),
            enable_llm=pipeline_settings.get('enable_llm', False),
            enable_actor_detection=pipeline_settings.get('enable_actor_detection', True),
            max_files=pipeline_settings.get('max_files', 1000),
            timeout_seconds=pipeline_settings.get('timeout_seconds', 300),
            include_metadata=pipeline_settings.get('include_metadata', True),
            generate_statistics=pipeline_settings.get('generate_statistics', True)
        )
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            # Core components (always required)
            self.project_discoverer = ProjectDiscoverer(self.config)
            self.ast_parser = ASTParser(self.config)
            self.graph_builder = OntologyGraphBuilder(self.config)
            
            # Optional components
            if self.pipeline_config.enable_actor_detection:
                self.pattern_matcher = PatternMatcher(self.config)
            else:
                self.pattern_matcher = None
            
            if self.pipeline_config.enable_pycg:
                self.pycg_integration = PyCGIntegration(self.config)
            else:
                self.pycg_integration = None
            
            if self.pipeline_config.enable_llm:
                self.llm_manager = LLMManager(self.config)
            else:
                self.llm_manager = None
            
            logger.info("Pipeline components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {e}")
            raise
    
    def analyze_project(self, project_path: str, output_path: Optional[str] = None) -> PipelineResult:
        """
        Execute complete analysis pipeline on a project.
        
        Args:
            project_path: Path to the Python project to analyze
            output_path: Optional path for JSON output
            
        Returns:
            PipelineResult with analysis results and statistics
        """
        self.analysis_start_time = time.time()
        self.current_project = project_path
        
        try:
            logger.info(f"Starting pipeline analysis of project: {project_path}")
            
            # Use existing DeterministicAnalyzer for now (simpler integration)
            from .core.analyzer import DeterministicAnalyzer
            analyzer = DeterministicAnalyzer(self.config)
            
            # Run the existing analyzer
            analysis_result = analyzer.analyze(project_path)
            
            # Convert to pipeline result format
            return PipelineResult(
                success=True,  # If we get here, it succeeded
                project_name=Path(project_path).name,
                files_analyzed=len(analysis_result.ast_results),
                functions_found=sum(len(result.functions) for result in analysis_result.ast_results.values() if hasattr(result, 'functions')),
                classes_found=sum(len(result.classes) for result in analysis_result.ast_results.values() if hasattr(result, 'classes')),
                imports_found=sum(len(result.imports) for result in analysis_result.ast_results.values() if hasattr(result, 'imports')),
                actors_detected=len(analysis_result.nodes),  # Use generated nodes as proxy
                nodes_generated=len(analysis_result.nodes),
                relationships_generated=len(analysis_result.relationships),
                analysis_time=time.time() - self.analysis_start_time,
                statistics={
                    'pipeline': {
                        'analysis_time_seconds': time.time() - self.analysis_start_time,
                        'components_enabled': {
                            'ast_parser': True,
                            'actor_detection': True,
                            'graph_building': True
                        }
                    },
                    'analysis': analysis_result.statistics
                }
            )
            
        except Exception as e:
            logger.error(f"Pipeline analysis failed: {e}")
            return PipelineResult(
                success=False,
                error_message=str(e),
                analysis_time=time.time() - self.analysis_start_time if self.analysis_start_time else 0
            )
    
    
    def validate_project(self, project_path: str) -> Tuple[bool, List[str]]:
        """Validate project before analysis."""
        try:
            # Use existing analyzer's validation
            from .core.analyzer import DeterministicAnalyzer
            analyzer = DeterministicAnalyzer(self.config)
            return analyzer.validate_project(project_path)
        except Exception as e:
            return False, [f"Validation error: {e}"]
    
    def get_project_preview(self, project_path: str) -> Dict[str, Any]:
        """Get quick project preview without full analysis."""
        try:
            project_structure = self.project_discoverer.discover_project(project_path)
            
            if not project_structure.python_files:
                return {"error": "No Python files found"}
            
            python_file_paths = [str(pf.path) for pf in project_structure.python_files]
            total_size = sum(Path(f).stat().st_size for f in python_file_paths if Path(f).exists())
            
            return {
                "project_name": Path(project_path).name,
                "total_files": len(python_file_paths),
                "total_lines": project_structure.total_lines,
                "total_size": total_size,
                "directories": len(set(Path(f).parent for f in python_file_paths)),
                "estimated_time": min(max(len(python_file_paths) * 0.1, 0.5), 30),
                "complexity": "Low" if len(python_file_paths) < 10 else "Medium" if len(python_file_paths) < 50 else "High",
                "main_files": [Path(f).name for f in python_file_paths if "main" in Path(f).name.lower()][:5],
                "file_types": {".py": len(python_file_paths)}
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and configuration."""
        return {
            "pipeline_config": {
                "continue_on_error": self.pipeline_config.continue_on_error,
                "partial_results": self.pipeline_config.partial_results,
                "confidence_threshold": self.pipeline_config.confidence_threshold,
                "max_files": self.pipeline_config.max_files,
                "timeout_seconds": self.pipeline_config.timeout_seconds
            },
            "components_enabled": {
                "project_discoverer": self.project_discoverer is not None,
                "ast_parser": self.ast_parser is not None,
                "pattern_matcher": self.pattern_matcher is not None,
                "pycg_integration": self.pycg_integration is not None,
                "llm_manager": self.llm_manager is not None,
                "graph_builder": self.graph_builder is not None
            },
            "current_analysis": {
                "project": self.current_project,
                "running": self.analysis_start_time is not None,
                "elapsed_time": time.time() - self.analysis_start_time if self.analysis_start_time else 0
            }
        }