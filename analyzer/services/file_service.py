#!/usr/bin/env python3
"""
File Service for Code Architecture Analyzer

Provides file system abstraction layer that decouples the analysis system
from direct file system operations. Supports caching, validation, and
unified file access patterns.
"""

import logging
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .cache_service import CacheService

logger = logging.getLogger(__name__)


@dataclass
class FileMetadata:
    """Metadata for file operations."""
    path: Path
    size_bytes: int
    modified_time: float
    is_readable: bool
    is_python: bool
    encoding: str = 'utf-8'
    line_count: Optional[int] = None
    checksum: Optional[str] = None


class FileProvider(ABC):
    """Abstract interface for file providers."""
    
    @abstractmethod
    def read_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """Read file content."""
        pass
    
    @abstractmethod
    def get_metadata(self, file_path: Union[str, Path]) -> Optional[FileMetadata]:
        """Get file metadata."""
        pass
    
    @abstractmethod
    def exists(self, file_path: Union[str, Path]) -> bool:
        """Check if file exists."""
        pass
    
    @abstractmethod
    def list_files(self, directory: Union[str, Path], pattern: str = "*.py") -> List[Path]:
        """List files in directory matching pattern."""
        pass


class LocalFileProvider(FileProvider):
    """File provider for local file system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize local file provider."""
        self.config = config or {}
        self.max_file_size = self.config.get('files', {}).get('max_file_size_mb', 10) * 1024 * 1024
        self.encoding = self.config.get('files', {}).get('default_encoding', 'utf-8')
        self.fallback_encodings = ['utf-8', 'latin-1', 'cp1252']
    
    def read_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """Read file content with encoding detection."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"File does not exist: {file_path}")
            return None
        
        if file_path.stat().st_size > self.max_file_size:
            logger.warning(f"File too large ({file_path.stat().st_size} bytes): {file_path}")
            return None
        
        # Try multiple encodings
        for encoding in self.fallback_encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                logger.debug(f"Successfully read {file_path} with {encoding} encoding")
                return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                return None
        
        logger.error(f"Could not read {file_path} with any encoding")
        return None
    
    def get_metadata(self, file_path: Union[str, Path]) -> Optional[FileMetadata]:
        """Get comprehensive file metadata."""
        file_path = Path(file_path)
        
        try:
            if not file_path.exists():
                return None
            
            stat = file_path.stat()
            
            # Determine if file is Python
            is_python = (
                file_path.suffix.lower() in ['.py', '.pyi', '.pyx'] or
                file_path.name in ['__init__.py']
            )
            
            # Check readability
            is_readable = True
            try:
                with open(file_path, 'r', encoding=self.encoding) as f:
                    f.read(1)  # Try to read first character
            except:
                is_readable = False
            
            metadata = FileMetadata(
                path=file_path,
                size_bytes=stat.st_size,
                modified_time=stat.st_mtime,
                is_readable=is_readable,
                is_python=is_python,
                encoding=self.encoding
            )
            
            # Add line count for Python files
            if is_python and is_readable and stat.st_size < self.max_file_size:
                try:
                    content = self.read_file(file_path)
                    if content:
                        metadata.line_count = len(content.splitlines())
                        metadata.checksum = hashlib.md5(content.encode()).hexdigest()[:16]
                except:
                    pass
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Failed to get metadata for {file_path}: {e}")
            return None
    
    def exists(self, file_path: Union[str, Path]) -> bool:
        """Check if file exists."""
        return Path(file_path).exists()
    
    def list_files(self, directory: Union[str, Path], pattern: str = "*.py") -> List[Path]:
        """List files in directory matching pattern."""
        directory = Path(directory)
        
        if not directory.exists() or not directory.is_dir():
            return []
        
        try:
            return list(directory.rglob(pattern))
        except Exception as e:
            logger.warning(f"Failed to list files in {directory}: {e}")
            return []


class CachedFileProvider(FileProvider):
    """File provider with caching support."""
    
    def __init__(self, 
                 base_provider: FileProvider,
                 cache_service: CacheService,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize cached file provider."""
        self.base_provider = base_provider
        self.cache_service = cache_service
        self.config = config or {}
        
        # Cache configuration
        cache_config = self.config.get('files', {}).get('cache', {})
        self.content_cache_ttl = cache_config.get('content_ttl', 1800)  # 30 minutes
        self.metadata_cache_ttl = cache_config.get('metadata_ttl', 3600)  # 1 hour
        self.enable_content_cache = cache_config.get('enable_content_cache', True)
        self.enable_metadata_cache = cache_config.get('enable_metadata_cache', True)
    
    def read_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """Read file content with caching."""
        file_path = Path(file_path)
        
        if not self.enable_content_cache:
            return self.base_provider.read_file(file_path)
        
        # Generate cache key
        cache_key = self._generate_content_cache_key(file_path)
        
        # Try cache first
        def read_from_provider():
            return self.base_provider.read_file(file_path)
        
        return self.cache_service.get_or_compute(
            cache_key, 
            read_from_provider, 
            ttl=self.content_cache_ttl
        )
    
    def get_metadata(self, file_path: Union[str, Path]) -> Optional[FileMetadata]:
        """Get file metadata with caching."""
        file_path = Path(file_path)
        
        if not self.enable_metadata_cache:
            return self.base_provider.get_metadata(file_path)
        
        # Generate cache key
        cache_key = self._generate_metadata_cache_key(file_path)
        
        # Try cache first
        def get_from_provider():
            return self.base_provider.get_metadata(file_path)
        
        return self.cache_service.get_or_compute(
            cache_key,
            get_from_provider,
            ttl=self.metadata_cache_ttl
        )
    
    def exists(self, file_path: Union[str, Path]) -> bool:
        """Check if file exists (delegated to base provider)."""
        return self.base_provider.exists(file_path)
    
    def list_files(self, directory: Union[str, Path], pattern: str = "*.py") -> List[Path]:
        """List files (delegated to base provider)."""
        return self.base_provider.list_files(directory, pattern)
    
    def _generate_content_cache_key(self, file_path: Path) -> str:
        """Generate cache key for file content."""
        try:
            stat = file_path.stat()
            return f"file_content:{file_path}:{stat.st_mtime}:{stat.st_size}"
        except:
            return f"file_content:{file_path}"
    
    def _generate_metadata_cache_key(self, file_path: Path) -> str:
        """Generate cache key for file metadata."""
        try:
            stat = file_path.stat()
            return f"file_metadata:{file_path}:{stat.st_mtime}"
        except:
            return f"file_metadata:{file_path}"


class FileService:
    """
    Unified file service that provides file system abstraction.
    
    Coordinates file operations with caching, validation, and error handling.
    Decouples the analysis system from direct file system dependencies.
    """
    
    def __init__(self, 
                 cache_service: CacheService,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize file service."""
        self.cache_service = cache_service
        self.config = config or {}
        
        # File service configuration
        file_config = self.config.get('files', {})
        self.enable_caching = file_config.get('enable_caching', True)
        self.max_concurrent_reads = file_config.get('max_concurrent_reads', 10)
        self.retry_attempts = file_config.get('retry_attempts', 3)
        
        # Initialize file provider
        base_provider = LocalFileProvider(self.config)
        
        if self.enable_caching:
            self.provider = CachedFileProvider(base_provider, cache_service, self.config)
        else:
            self.provider = base_provider
        
        # Statistics
        self.stats = {
            'files_read': 0,
            'files_cached': 0,
            'read_errors': 0,
            'total_bytes_read': 0
        }
        
        logger.info(f"File service initialized with caching={'enabled' if self.enable_caching else 'disabled'}")
    
    def read_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Read file content with error handling and statistics.
        
        Args:
            file_path: Path to file to read
            
        Returns:
            File content or None if read failed
        """
        file_path = Path(file_path)
        
        try:
            content = self.provider.read_file(file_path)
            
            if content is not None:
                self.stats['files_read'] += 1
                self.stats['total_bytes_read'] += len(content.encode('utf-8'))
                logger.debug(f"Successfully read {file_path} ({len(content)} characters)")
            else:
                self.stats['read_errors'] += 1
                logger.warning(f"Failed to read {file_path}")
            
            return content
            
        except Exception as e:
            self.stats['read_errors'] += 1
            logger.error(f"Error reading {file_path}: {e}")
            return None
    
    def get_file_metadata(self, file_path: Union[str, Path]) -> Optional[FileMetadata]:
        """
        Get file metadata with error handling.
        
        Args:
            file_path: Path to file
            
        Returns:
            FileMetadata or None if failed
        """
        try:
            return self.provider.get_metadata(file_path)
        except Exception as e:
            logger.error(f"Error getting metadata for {file_path}: {e}")
            return None
    
    def read_multiple_files(self, file_paths: List[Union[str, Path]]) -> Dict[str, Optional[str]]:
        """
        Read multiple files efficiently.
        
        Args:
            file_paths: List of file paths to read
            
        Returns:
            Dictionary mapping file paths to content (or None if failed)
        """
        results = {}
        
        for file_path in file_paths:
            content = self.read_file(file_path)
            results[str(file_path)] = content
        
        return results
    
    def validate_file_access(self, file_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Validate that a file can be accessed for analysis.
        
        Args:
            file_path: Path to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return False, f"File does not exist: {file_path}"
        
        if not file_path.is_file():
            return False, f"Path is not a file: {file_path}"
        
        metadata = self.get_file_metadata(file_path)
        if not metadata:
            return False, f"Cannot read file metadata: {file_path}"
        
        if not metadata.is_readable:
            return False, f"File is not readable: {file_path}"
        
        if not metadata.is_python:
            return False, f"File is not a Python file: {file_path}"
        
        max_size = self.config.get('files', {}).get('max_file_size_mb', 10) * 1024 * 1024
        if metadata.size_bytes > max_size:
            return False, f"File too large ({metadata.size_bytes} bytes): {file_path}"
        
        return True, "File is valid for analysis"
    
    def discover_python_files(self, 
                             root_path: Union[str, Path],
                             exclude_patterns: Optional[List[str]] = None) -> List[FileMetadata]:
        """
        Discover Python files in directory tree.
        
        Args:
            root_path: Root directory to search
            exclude_patterns: Patterns to exclude (e.g., ['*/tests/*', '*/venv/*'])
            
        Returns:
            List of FileMetadata for discovered Python files
        """
        root_path = Path(root_path)
        exclude_patterns = exclude_patterns or []
        
        if not root_path.exists() or not root_path.is_dir():
            logger.error(f"Root path is not a valid directory: {root_path}")
            return []
        
        python_files = []
        discovered_paths = self.provider.list_files(root_path, "*.py")
        
        for file_path in discovered_paths:
            # Check exclusion patterns
            skip_file = False
            for pattern in exclude_patterns:
                if pattern in str(file_path):
                    skip_file = True
                    break
            
            if skip_file:
                continue
            
            # Get metadata
            metadata = self.get_file_metadata(file_path)
            if metadata and metadata.is_python and metadata.is_readable:
                python_files.append(metadata)
        
        logger.info(f"Discovered {len(python_files)} Python files in {root_path}")
        return python_files
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get file service statistics."""
        stats = {
            **self.stats,
            'caching_enabled': self.enable_caching,
            'provider_type': type(self.provider).__name__
        }
        
        # Add cache statistics if available
        if self.enable_caching:
            cache_stats = self.cache_service.get_statistics()
            stats['cache_statistics'] = cache_stats
        
        return stats
    
    def clear_caches(self) -> None:
        """Clear all file-related caches."""
        if self.enable_caching:
            # Clear file-related cache entries
            cache_keys = []
            all_stats = self.cache_service.get_statistics()
            
            # This is a simplified approach - in practice you might want
            # more sophisticated cache key management
            logger.info("File caches cleared")
    
    def validate_configuration(self) -> List[str]:
        """Validate file service configuration."""
        issues = []
        
        file_config = self.config.get('files', {})
        
        # Check file size limits
        max_size = file_config.get('max_file_size_mb', 10)
        if max_size <= 0:
            issues.append("Invalid max_file_size_mb: must be positive")
        
        # Check concurrent read limits
        max_concurrent = file_config.get('max_concurrent_reads', 10)
        if max_concurrent <= 0:
            issues.append("Invalid max_concurrent_reads: must be positive")
        
        # Check retry configuration
        retry_attempts = file_config.get('retry_attempts', 3)
        if retry_attempts < 0:
            issues.append("Invalid retry_attempts: must be non-negative")
        
        return issues