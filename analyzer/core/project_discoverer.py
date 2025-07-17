#!/usr/bin/env python3
"""
Project Discoverer for Code Architecture Analyzer

Discovers and validates Python project structure, handling file patterns
and size constraints according to configuration.
"""

import os
import logging
import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict


logger = logging.getLogger(__name__)


@dataclass
class ProjectFile:
    """Represents a discovered Python file in the project."""
    path: Path
    relative_path: Path
    size_bytes: int
    line_count: int = 0
    is_test: bool = False
    is_main: bool = False
    module_name: str = ""


@dataclass
class ProjectStructure:
    """Complete project structure information."""
    root_path: Path
    python_files: List[ProjectFile] = field(default_factory=list)
    total_files: int = 0
    total_lines: int = 0
    total_size_bytes: int = 0
    directories: List[Path] = field(default_factory=list)
    excluded_files: List[Path] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProjectDiscoverer:
    """
    Discovers and analyzes Python project structure.
    
    Handles file discovery, filtering, and validation according to
    configuration patterns and size limits.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize project discoverer with configuration."""
        self.config = config or {}
        
        # Load patterns from config
        project_config = self.config.get('project', {})
        self.include_patterns = project_config.get('include_patterns', ['*.py', '**/*.py'])
        self.exclude_patterns = project_config.get('exclude_patterns', [
            '__pycache__/**', '*.pyc', '.git/**', '.venv/**', 'venv/**',
            'env/**', '.pytest_cache/**', 'node_modules/**', 'build/**',
            'dist/**', '*.egg-info/**'
        ])
        
        # Size limits
        self.max_loc = project_config.get('max_loc', 25000)
        self.max_file_size_kb = project_config.get('max_file_size_kb', 500)
        self.max_file_size_bytes = self.max_file_size_kb * 1024
        
        logger.debug(f"ProjectDiscoverer initialized with {len(self.include_patterns)} include patterns")
    
    def discover_project(self, project_path: str) -> ProjectStructure:
        """
        Discover and analyze complete project structure.
        
        Args:
            project_path: Path to the Python project root
            
        Returns:
            ProjectStructure with discovered files and metadata
        """
        project_path = Path(project_path).resolve()
        
        if not project_path.exists():
            error_msg = f"Project path does not exist: {project_path}"
            logger.error(error_msg)
            return ProjectStructure(
                root_path=project_path,
                errors=[error_msg]
            )
        
        if not project_path.is_dir():
            error_msg = f"Project path is not a directory: {project_path}"
            logger.error(error_msg)
            return ProjectStructure(
                root_path=project_path,
                errors=[error_msg]
            )
        
        logger.info(f"Discovering project structure in: {project_path}")
        
        structure = ProjectStructure(root_path=project_path)
        
        try:
            # Discover all Python files
            all_files = self._find_python_files(project_path)
            logger.debug(f"Found {len(all_files)} potential Python files")
            
            # Filter files by patterns
            filtered_files = self._filter_files(all_files, project_path)
            logger.debug(f"After filtering: {len(filtered_files)} files remain")
            
            # Process each file
            for file_path in filtered_files:
                try:
                    project_file = self._process_file(file_path, project_path)
                    if project_file:
                        structure.python_files.append(project_file)
                        structure.total_size_bytes += project_file.size_bytes
                        structure.total_lines += project_file.line_count
                except Exception as e:
                    error_msg = f"Error processing file {file_path}: {e}"
                    logger.warning(error_msg)
                    structure.errors.append(error_msg)
            
            # Update statistics
            structure.total_files = len(structure.python_files)
            
            # Discover directories
            structure.directories = self._find_directories(project_path)
            
            # Validate project constraints
            self._validate_project_constraints(structure)
            
            # Add metadata
            structure.metadata = self._generate_metadata(structure)
            
            logger.info(f"Project discovery completed: {structure.total_files} files, "
                       f"{structure.total_lines} lines, {len(structure.errors)} errors")
            
        except Exception as e:
            error_msg = f"Failed to discover project structure: {e}"
            logger.error(error_msg)
            structure.errors.append(error_msg)
        
        return structure
    
    def _find_python_files(self, root_path: Path) -> List[Path]:
        """Find all potential Python files in the project."""
        python_files = []
        
        try:
            # Walk through all files recursively
            for pattern in self.include_patterns:
                if '**' in pattern:
                    # Recursive pattern
                    files = root_path.glob(pattern)
                else:
                    # Non-recursive pattern
                    files = root_path.glob(pattern)
                
                for file_path in files:
                    if file_path.is_file() and file_path not in python_files:
                        python_files.append(file_path)
            
        except Exception as e:
            logger.error(f"Error finding Python files: {e}")
        
        return python_files
    
    def _filter_files(self, files: List[Path], root_path: Path) -> List[Path]:
        """Filter files based on exclude patterns."""
        filtered_files = []
        
        for file_path in files:
            try:
                # Get relative path for pattern matching
                relative_path = file_path.relative_to(root_path)
                relative_str = str(relative_path)
                
                # Check exclude patterns
                excluded = False
                for pattern in self.exclude_patterns:
                    if fnmatch.fnmatch(relative_str, pattern) or fnmatch.fnmatch(str(file_path), pattern):
                        excluded = True
                        logger.debug(f"Excluding file {relative_path} (matches pattern: {pattern})")
                        break
                    
                    # Check if any parent directory matches
                    for parent in relative_path.parents:
                        if fnmatch.fnmatch(str(parent), pattern.rstrip('/**')):
                            excluded = True
                            logger.debug(f"Excluding file {relative_path} (parent matches: {pattern})")
                            break
                    
                    if excluded:
                        break
                
                if not excluded:
                    filtered_files.append(file_path)
                
            except Exception as e:
                logger.warning(f"Error filtering file {file_path}: {e}")
        
        return filtered_files
    
    def _process_file(self, file_path: Path, root_path: Path) -> Optional[ProjectFile]:
        """Process a single Python file and extract metadata."""
        try:
            # Check file size
            size_bytes = file_path.stat().st_size
            if size_bytes > self.max_file_size_bytes:
                logger.warning(f"Skipping large file {file_path} ({size_bytes} bytes)")
                return None
            
            # Read file and count lines
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    line_count = len(content.splitlines())
            except Exception as e:
                logger.warning(f"Could not read file {file_path}: {e}")
                line_count = 0
            
            # Calculate relative path
            relative_path = file_path.relative_to(root_path)
            
            # Determine file characteristics
            is_test = self._is_test_file(file_path)
            is_main = self._is_main_file(file_path, content if 'content' in locals() else "")
            module_name = self._extract_module_name(relative_path)
            
            return ProjectFile(
                path=file_path,
                relative_path=relative_path,
                size_bytes=size_bytes,
                line_count=line_count,
                is_test=is_test,
                is_main=is_main,
                module_name=module_name
            )
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None
    
    def _is_test_file(self, file_path: Path) -> bool:
        """Determine if a file is a test file."""
        name = file_path.name.lower()
        return (
            name.startswith('test_') or 
            name.endswith('_test.py') or
            'test' in str(file_path.parent).lower()
        )
    
    def _is_main_file(self, file_path: Path, content: str) -> bool:
        """Determine if a file is a main entry point."""
        name = file_path.name.lower()
        if name in ['main.py', '__main__.py', 'app.py', 'run.py']:
            return True
        
        # Check content for main patterns
        if content and ('if __name__ == "__main__"' in content or 'if __name__ == \'__main__\'' in content):
            return True
        
        return False
    
    def _extract_module_name(self, relative_path: Path) -> str:
        """Extract Python module name from relative path."""
        # Convert path to module name (use Unix path format, remove .py)
        parts = relative_path.parts
        if parts[-1].endswith('.py'):
            parts = parts[:-1] + (parts[-1][:-3],)
        
        # Handle __init__.py files
        if parts[-1] == '__init__':
            parts = parts[:-1]
        
        return '/'.join(parts) if parts else ''
    
    def _find_directories(self, root_path: Path) -> List[Path]:
        """Find all directories in the project."""
        directories = []
        
        try:
            for item in root_path.rglob('*'):
                if item.is_dir():
                    # Check if directory should be excluded
                    relative_path = item.relative_to(root_path)
                    excluded = False
                    
                    for pattern in self.exclude_patterns:
                        if fnmatch.fnmatch(str(relative_path), pattern.rstrip('/**')):
                            excluded = True
                            break
                    
                    if not excluded:
                        directories.append(item)
        
        except Exception as e:
            logger.error(f"Error finding directories: {e}")
        
        return directories
    
    def _validate_project_constraints(self, structure: ProjectStructure) -> None:
        """Validate project against size and other constraints."""
        # Check total lines of code
        if structure.total_lines > self.max_loc:
            warning_msg = f"Project exceeds maximum lines of code: {structure.total_lines} > {self.max_loc}"
            logger.warning(warning_msg)
            structure.errors.append(warning_msg)
        
        # Check if project has any Python files
        if structure.total_files == 0:
            warning_msg = "No Python files found in project"
            logger.warning(warning_msg)
            structure.errors.append(warning_msg)
    
    def _generate_metadata(self, structure: ProjectStructure) -> Dict[str, Any]:
        """Generate metadata about the discovered project."""
        metadata = {
            'discovery_version': '1.0.0',
            'file_extensions': self._get_file_extensions(structure.python_files),
            'directory_count': len(structure.directories),
            'test_file_count': sum(1 for f in structure.python_files if f.is_test),
            'main_file_count': sum(1 for f in structure.python_files if f.is_main),
            'average_file_size': structure.total_size_bytes / structure.total_files if structure.total_files > 0 else 0,
            'average_lines_per_file': structure.total_lines / structure.total_files if structure.total_files > 0 else 0,
            'largest_file': self._get_largest_file(structure.python_files),
            'module_distribution': self._get_module_distribution(structure.python_files)
        }
        
        return metadata
    
    def _get_file_extensions(self, files: List[ProjectFile]) -> Dict[str, int]:
        """Get distribution of file extensions."""
        extensions = defaultdict(int)
        for file in files:
            ext = file.path.suffix.lower()
            extensions[ext] += 1
        return dict(extensions)
    
    def _get_largest_file(self, files: List[ProjectFile]) -> Optional[Dict[str, Any]]:
        """Get information about the largest file."""
        if not files:
            return None
        
        largest = max(files, key=lambda f: f.size_bytes)
        return {
            'path': str(largest.relative_path),
            'size_bytes': largest.size_bytes,
            'line_count': largest.line_count
        }
    
    def _get_module_distribution(self, files: List[ProjectFile]) -> Dict[str, int]:
        """Get distribution of files by top-level module."""
        distribution = defaultdict(int)
        for file in files:
            if file.module_name:
                top_module = file.module_name.split('.')[0]
                distribution[top_module] += 1
        return dict(distribution)
    
    def validate_project_path(self, project_path: str) -> Tuple[bool, List[str]]:
        """
        Validate that a project path is suitable for analysis.
        
        Args:
            project_path: Path to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        path = Path(project_path)
        
        if not path.exists():
            errors.append(f"Path does not exist: {project_path}")
        elif not path.is_dir():
            errors.append(f"Path is not a directory: {project_path}")
        else:
            # Check for Python files
            python_files = list(path.glob('**/*.py'))
            if not python_files:
                errors.append("No Python files found in project")
            
            # Check read permissions
            try:
                list(path.iterdir())
            except PermissionError:
                errors.append(f"No read permission for directory: {project_path}")
        
        return len(errors) == 0, errors