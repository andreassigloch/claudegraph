#!/usr/bin/env python3
"""
Service Layer for Code Architecture Analyzer

This module provides service-oriented architecture components that decouple
business logic from infrastructure concerns. Services handle coordination
of multiple components while maintaining clean interfaces.

Services included:
- AnalysisService: Core analysis orchestration
- DetectionService: Actor detection coordination  
- CacheService: Caching abstraction
- FileService: File system abstraction
"""

from .analysis_service import AnalysisService, AnalysisContext
from .detection_service import DetectionService, DetectionStrategy
from .cache_service import CacheService, CacheBackend
from .file_service import FileService, FileProvider, FileMetadata

__all__ = [
    'AnalysisService',
    'AnalysisContext',
    'DetectionService',
    'DetectionStrategy', 
    'CacheService',
    'CacheBackend',
    'FileService',
    'FileProvider',
    'FileMetadata'
]