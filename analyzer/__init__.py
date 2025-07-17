#!/usr/bin/env python3
"""
Code Architecture Analyzer

Automated extraction and visualization of code architectures from Python projects
using ontology-based graph analysis.
"""

__version__ = "0.1.0"
__author__ = "Andreas"
__email__ = "andreas@example.com"
__description__ = "Automated extraction and visualization of code architectures from Python projects"

import logging
import sys
from typing import Optional

# Configure default logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Get package logger
logger = logging.getLogger(__name__)

# Version info
VERSION_INFO = tuple(map(int, __version__.split('.')))

def get_version() -> str:
    """Get the current version string."""
    return __version__

def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )
    
    logger.info(f"Logging configured at {level} level")

# Module imports - import key classes for easy access
try:
    from .core.project_discoverer import ProjectDiscoverer
    from .core.ast_parser import ASTParser
    from .detection.pattern_matcher import PatternMatcher
    from .llm.client import LLMManager, LLMClientFactory
    from .graph.builder import OntologyGraphBuilder
    
    # Main pipeline class
    from .pipeline import AnalyzerPipeline
    
    __all__ = [
        'AnalyzerPipeline',
        'ProjectDiscoverer', 
        'ASTParser',
        'PatternMatcher',
        'LLMManager',
        'LLMClientFactory',
        'OntologyGraphBuilder',
        'get_version',
        'setup_logging',
        '__version__'
    ]
    
except ImportError as e:
    # During development, some modules might not exist yet
    logger.warning(f"Could not import all modules: {e}")
    __all__ = [
        'get_version',
        'setup_logging', 
        '__version__'
    ]

logger.info(f"Code Architecture Analyzer v{__version__} initialized")