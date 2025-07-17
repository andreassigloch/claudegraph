#!/usr/bin/env python3
"""
CLI Interface for Code Architecture Analyzer

Provides command-line interface for running code analysis and generating
ontology-compliant graphs from Python projects.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# CLI module imports
try:
    from .main import main, cli
    from .config import load_unified_config, validate_config
    
    __all__ = [
        'main',
        'cli', 
        'load_unified_config',
        'validate_config'
    ]
    
except ImportError as e:
    # During development, some modules might not exist yet
    logger.warning(f"Could not import all CLI modules: {e}")
    __all__ = []

def get_version() -> str:
    """Get CLI version."""
    from analyzer import __version__
    return __version__

logger.info("CLI interface initialized")