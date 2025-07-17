#!/usr/bin/env python3
"""
Core Analysis Components

This module contains the fundamental analysis components for deterministic
code structure extraction and processing.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Core component imports
try:
    from .project_discoverer import ProjectDiscoverer
    from .ast_parser import ASTParser
    from .pycg_integration import PyCGIntegration
    from .node_generator import NodeGenerator
    
    __all__ = [
        'ProjectDiscoverer',
        'ASTParser', 
        'PyCGIntegration',
        'NodeGenerator',
        'DeterministicAnalyzer'
    ]
    
    # Import main analyzer class
    from .analyzer import DeterministicAnalyzer
    
except ImportError as e:
    # During development, some modules might not exist yet
    logger.warning(f"Could not import all core modules: {e}")
    __all__ = []


def get_available_analyzers() -> List[str]:
    """Get list of available analyzer components."""
    available = []
    
    try:
        from .project_discoverer import ProjectDiscoverer
        available.append('ProjectDiscoverer')
    except ImportError:
        pass
    
    try:
        from .ast_parser import ASTParser
        available.append('ASTParser')
    except ImportError:
        pass
        
    try:
        from .pycg_integration import PyCGIntegration
        available.append('PyCGIntegration')
    except ImportError:
        pass
        
    try:
        from .node_generator import NodeGenerator
        available.append('NodeGenerator')
    except ImportError:
        pass
    
    return available


def validate_core_dependencies() -> bool:
    """Validate that all required core dependencies are available."""
    required_modules = ['ast', 'pathlib', 'subprocess', 'json']
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            logger.error(f"Required module {module} not available")
            return False
    
    return True


# Initialize core logging
logger.info("Core analysis components initialized")

# Validate dependencies on import
if not validate_core_dependencies():
    logger.warning("Some core dependencies are missing")