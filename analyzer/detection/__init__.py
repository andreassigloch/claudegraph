#!/usr/bin/env python3
"""
Detection Components for Code Architecture Analyzer

This module contains pattern matching and actor detection components for
identifying external dependencies and system interactions in Python code.
"""

import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Detection component imports
try:
    from .pattern_matcher import PatternMatcher
    from .http_detector import HttpDetector
    from .database_detector import DatabaseDetector
    from .filesystem_detector import FilesystemDetector
    from .endpoint_detector import EndpointDetector
    from .actor_classifier import ActorClassifier
    
    __all__ = [
        'PatternMatcher',
        'HttpDetector',
        'DatabaseDetector', 
        'FilesystemDetector',
        'EndpointDetector',
        'ActorClassifier',
        'get_available_detectors',
        'validate_detection_config',
        'create_detector_pipeline'
    ]
    
except ImportError as e:
    # During development, some modules might not exist yet
    logger.warning(f"Could not import all detection modules: {e}")
    __all__ = [
        'get_available_detectors',
        'validate_detection_config'
    ]


def get_available_detectors() -> List[str]:
    """Get list of available detector components."""
    available = []
    
    try:
        from .pattern_matcher import PatternMatcher
        available.append('PatternMatcher')
    except ImportError:
        pass
    
    try:
        from .http_detector import HttpDetector
        available.append('HttpDetector')
    except ImportError:
        pass
        
    try:
        from .database_detector import DatabaseDetector
        available.append('DatabaseDetector')
    except ImportError:
        pass
        
    try:
        from .filesystem_detector import FilesystemDetector
        available.append('FilesystemDetector')
    except ImportError:
        pass
        
    try:
        from .endpoint_detector import EndpointDetector
        available.append('EndpointDetector')
    except ImportError:
        pass
        
    try:
        from .actor_classifier import ActorClassifier
        available.append('ActorClassifier')
    except ImportError:
        pass
    
    return available


def validate_detection_config(config: Dict[str, Any]) -> bool:
    """Validate detection configuration parameters."""
    try:
        detection_config = config.get('detection', {})
        
        # Check confidence thresholds
        confidence_threshold = detection_config.get('confidence_threshold', 0.8)
        if not isinstance(confidence_threshold, (int, float)) or not 0 <= confidence_threshold <= 1:
            logger.error("detection.confidence_threshold must be between 0 and 1")
            return False
        
        # Check enabled detectors
        enabled_detectors = detection_config.get('enabled_detectors', {})
        if enabled_detectors and not isinstance(enabled_detectors, dict):
            logger.error("detection.enabled_detectors must be a dictionary")
            return False
        
        # Check pattern configuration
        patterns_config = detection_config.get('patterns', {})
        if patterns_config and not isinstance(patterns_config, dict):
            logger.error("detection.patterns must be a dictionary")
            return False
        
        logger.debug("Detection configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Detection configuration validation failed: {e}")
        return False


def create_detector_pipeline(config: Dict[str, Any]) -> Optional[List[Any]]:
    """Create detector pipeline based on configuration."""
    try:
        detection_config = config.get('detection', {})
        enabled_detectors = detection_config.get('enabled_detectors', {
            'http': True,
            'database': True,
            'filesystem': True,
            'endpoint': True
        })
        
        pipeline = []
        
        # Add detectors based on configuration
        if enabled_detectors.get('http', True):
            try:
                from .http_detector import HttpDetector
                pipeline.append(HttpDetector(config))
            except ImportError:
                logger.warning("HttpDetector not available")
        
        if enabled_detectors.get('database', True):
            try:
                from .database_detector import DatabaseDetector
                pipeline.append(DatabaseDetector(config))
            except ImportError:
                logger.warning("DatabaseDetector not available")
        
        if enabled_detectors.get('filesystem', True):
            try:
                from .filesystem_detector import FilesystemDetector
                pipeline.append(FilesystemDetector(config))
            except ImportError:
                logger.warning("FilesystemDetector not available")
        
        if enabled_detectors.get('endpoint', True):
            try:
                from .endpoint_detector import EndpointDetector
                pipeline.append(EndpointDetector(config))
            except ImportError:
                logger.warning("EndpointDetector not available")
        
        logger.info(f"Created detection pipeline with {len(pipeline)} detectors")
        return pipeline if pipeline else None
        
    except Exception as e:
        logger.error(f"Failed to create detector pipeline: {e}")
        return None


def get_supported_actor_types() -> List[str]:
    """Get list of supported actor types for detection."""
    return [
        'HttpClient',
        'Database', 
        'FileSystem',
        'WebEndpoint',
        'MessageQueue',
        'ConfigManager',
        'CloudService',
        'ExternalApi',
        'Cache',
        'Monitor'
    ]


def get_detection_statistics(detectors: List[Any]) -> Dict[str, Any]:
    """Get statistics from detector pipeline."""
    stats = {
        'total_detectors': len(detectors) if detectors else 0,
        'detector_types': [],
        'patterns_loaded': 0,
        'confidence_levels': {}
    }
    
    if not detectors:
        return stats
    
    try:
        for detector in detectors:
            detector_name = detector.__class__.__name__
            stats['detector_types'].append(detector_name)
            
            # Get pattern count if available
            if hasattr(detector, 'get_pattern_count'):
                stats['patterns_loaded'] += detector.get_pattern_count()
            
            # Get confidence info if available
            if hasattr(detector, 'get_confidence_threshold'):
                stats['confidence_levels'][detector_name] = detector.get_confidence_threshold()
    
    except Exception as e:
        logger.warning(f"Error collecting detection statistics: {e}")
    
    return stats


# Initialize detection logging
logger.info("Detection components initialized")

# Validate detection environment
available_detectors = get_available_detectors()
if not available_detectors:
    logger.warning("No detection components are currently available")
else:
    logger.debug(f"Available detectors: {', '.join(available_detectors)}")