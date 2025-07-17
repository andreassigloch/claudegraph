#!/usr/bin/env python3
"""
Filesystem Detector for Code Architecture Analyzer

Specialized detector for identifying filesystem usage patterns in Python code.
Detects pathlib, os, shutil, built-in file operations and file I/O patterns.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from pathlib import Path

from .pattern_matcher import BaseDetector, PatternRule, DetectionMatch, ActorType, ConfidenceLevel
from ..core.ast_parser import ASTParseResult, FunctionInfo, ImportInfo

logger = logging.getLogger(__name__)


@dataclass
class FilesystemPattern:
    """Filesystem-specific pattern information."""
    library_name: str
    operations: List[str]
    confidence: float
    async_support: bool = False
    path_manipulation: bool = False
    file_io: bool = False


class FilesystemDetector(BaseDetector):
    """
    Specialized detector for filesystem operation patterns.
    
    Identifies usage of filesystem libraries and extracts information about
    file operations, path manipulation, and I/O operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize filesystem detector with configuration."""
        super().__init__(config)
        
        # Filesystem-specific configuration
        fs_config = self.config.get('detection', {}).get('filesystem', {})
        self.detect_file_paths = fs_config.get('detect_file_paths', True)
        self.detect_file_extensions = fs_config.get('detect_file_extensions', True)
        self.min_path_confidence = fs_config.get('min_path_confidence', 0.6)
        
        # File path patterns for detecting file paths in strings
        self.file_path_pattern = re.compile(
            r'(?:[a-zA-Z]:[/\\]|/|\.{1,2}/|~/)[a-zA-Z0-9_./\\-]+(?:\.[a-zA-Z0-9]+)?',
            re.IGNORECASE
        )
        
        # File extension pattern
        self.file_extension_pattern = re.compile(r'\.\w{1,10}$')
        
        # Common file operations
        self.file_operations = {
            'READ': ['read', 'open', 'load', 'get', 'fetch'],
            'WRITE': ['write', 'save', 'dump', 'put', 'create'],
            'COPY': ['copy', 'duplicate', 'backup'],
            'MOVE': ['move', 'rename', 'relocate'],
            'DELETE': ['delete', 'remove', 'unlink', 'rmtree'],
            'LIST': ['list', 'listdir', 'glob', 'scan'],
            'CHECK': ['exists', 'isfile', 'isdir', 'stat']
        }
        
        logger.debug("Filesystem detector initialized")
    
    def _load_patterns(self) -> List[PatternRule]:
        """Load filesystem detection patterns."""
        patterns = []
        
        # Get confidence scores from config
        base_confidence = self.config.get('deterministic', {}).get('pattern_confidence', {}).get('file_system', 0.98)
        
        # Pathlib patterns (modern Python)
        patterns.append(PatternRule(
            name="pathlib_library",
            actor_type=ActorType.FILE_SYSTEM,
            confidence=base_confidence,
            imports=["pathlib", "pathlib.*"],
            function_calls=[
                "Path", "PurePath", "WindowsPath", "PosixPath",
                "Path.open", "Path.read_text", "Path.write_text", "Path.read_bytes", "Path.write_bytes",
                "Path.exists", "Path.is_file", "Path.is_dir", "Path.stat",
                "Path.mkdir", "Path.rmdir", "Path.unlink", "Path.rename",
                "Path.glob", "Path.rglob", "Path.iterdir",
                "Path.parent", "Path.name", "Path.suffix", "Path.stem"
            ],
            keywords=["pathlib", "Path", "file", "directory"]
        ))
        
        # Built-in file operations
        patterns.append(PatternRule(
            name="builtin_file_ops",
            actor_type=ActorType.FILE_SYSTEM,
            confidence=base_confidence - 0.1,
            function_calls=[
                "open(", "file.read", "file.write", "file.close",
                "file.readline", "file.readlines", "file.writelines",
                "file.seek", "file.tell", "file.flush"
            ],
            keywords=["open", "file", "read", "write"]
        ))
        
        # os and os.path patterns
        patterns.append(PatternRule(
            name="os_path_library",
            actor_type=ActorType.FILE_SYSTEM,
            confidence=base_confidence - 0.05,
            imports=["os", "os.path", "os.*"],
            function_calls=[
                "os.path.join", "os.path.exists", "os.path.isfile", "os.path.isdir",
                "os.path.dirname", "os.path.basename", "os.path.splitext",
                "os.path.abspath", "os.path.relpath", "os.path.normpath",
                "os.listdir", "os.makedirs", "os.removedirs", "os.rename",
                "os.remove", "os.rmdir", "os.walk", "os.scandir",
                "os.getcwd", "os.chdir", "os.stat", "os.chmod"
            ],
            keywords=["os", "path", "directory", "file"]
        ))
        
        # shutil patterns (high-level file operations)
        patterns.append(PatternRule(
            name="shutil_library",
            actor_type=ActorType.FILE_SYSTEM,
            confidence=base_confidence,
            imports=["shutil"],
            function_calls=[
                "shutil.copy", "shutil.copy2", "shutil.copyfile", "shutil.copystat",
                "shutil.copytree", "shutil.move", "shutil.rmtree",
                "shutil.disk_usage", "shutil.which", "shutil.make_archive",
                "shutil.unpack_archive", "shutil.get_archive_formats"
            ],
            keywords=["shutil", "copy", "move", "archive"]
        ))
        
        # glob patterns (file pattern matching)
        patterns.append(PatternRule(
            name="glob_library",
            actor_type=ActorType.FILE_SYSTEM,
            confidence=base_confidence - 0.05,
            imports=["glob"],
            function_calls=[
                "glob.glob", "glob.iglob", "glob.escape"
            ],
            keywords=["glob", "pattern", "wildcard"]
        ))
        
        # tempfile patterns (temporary files)
        patterns.append(PatternRule(
            name="tempfile_library",
            actor_type=ActorType.FILE_SYSTEM,
            confidence=base_confidence - 0.1,
            imports=["tempfile"],
            function_calls=[
                "tempfile.TemporaryFile", "tempfile.NamedTemporaryFile",
                "tempfile.TemporaryDirectory", "tempfile.mkstemp", "tempfile.mkdtemp",
                "tempfile.gettempdir", "tempfile.gettempprefix"
            ],
            keywords=["tempfile", "temporary", "temp"]
        ))
        
        # File format specific patterns
        patterns.extend([
            PatternRule(
                name="json_file_ops",
                actor_type=ActorType.FILE_SYSTEM,
                confidence=base_confidence - 0.1,
                imports=["json"],
                function_calls=["json.load", "json.dump", "json.loads", "json.dumps"],
                keywords=["json", "load", "dump"],
                file_extensions=[".json"]
            ),
            PatternRule(
                name="csv_file_ops",
                actor_type=ActorType.FILE_SYSTEM,
                confidence=base_confidence - 0.1,
                imports=["csv"],
                function_calls=[
                    "csv.reader", "csv.writer", "csv.DictReader", "csv.DictWriter",
                    "csv.Sniffer", "reader.readrow", "writer.writerow"
                ],
                keywords=["csv", "reader", "writer"],
                file_extensions=[".csv"]
            ),
            PatternRule(
                name="yaml_file_ops",
                actor_type=ActorType.FILE_SYSTEM,
                confidence=base_confidence - 0.1,
                imports=["yaml", "pyyaml", "ruamel.yaml"],
                function_calls=[
                    "yaml.load", "yaml.dump", "yaml.safe_load", "yaml.safe_dump",
                    "yaml.load_all", "yaml.dump_all"
                ],
                keywords=["yaml", "load", "dump"],
                file_extensions=[".yaml", ".yml"]
            ),
            PatternRule(
                name="pickle_file_ops",
                actor_type=ActorType.FILE_SYSTEM,
                confidence=base_confidence - 0.05,
                imports=["pickle", "cPickle"],
                function_calls=[
                    "pickle.load", "pickle.dump", "pickle.loads", "pickle.dumps"
                ],
                keywords=["pickle", "serialize"],
                file_extensions=[".pkl", ".pickle"]
            )
        ])
        
        # IO module patterns
        patterns.append(PatternRule(
            name="io_library",
            actor_type=ActorType.FILE_SYSTEM,
            confidence=base_confidence - 0.1,
            imports=["io"],
            function_calls=[
                "io.open", "io.StringIO", "io.BytesIO", "io.TextIOWrapper",
                "io.BufferedReader", "io.BufferedWriter"
            ],
            keywords=["io", "stream", "buffer"]
        ))
        
        logger.info(f"Loaded {len(patterns)} filesystem detection patterns")
        return patterns
    
    def _detect_function_calls(self, functions: List[FunctionInfo]) -> List[DetectionMatch]:
        """Detect filesystem usage from function calls."""
        return self._detect_filesystem_calls(functions)
    
    def _detect_content_patterns(self, ast_result: ASTParseResult) -> List[DetectionMatch]:
        """Detect file paths in string literals."""
        if self.detect_file_paths:
            return self._detect_file_paths_in_code(ast_result)
        return []
    
    def _detect_filesystem_calls(self, functions: List[FunctionInfo]) -> List[DetectionMatch]:
        """Detect filesystem operations within functions."""
        matches = []
        
        for func_info in functions:
            for call in func_info.calls:
                # Check against filesystem patterns
                for pattern in self.patterns:
                    if self._match_filesystem_call(call, pattern):
                        match = DetectionMatch(
                            actor_type=ActorType.FILE_SYSTEM,
                            confidence=pattern.confidence,
                            pattern_name=pattern.name,
                            evidence={
                                'function_call': call,
                                'operation_type': self._extract_operation_type(call),
                                'containing_function': func_info.name,
                                'is_async': func_info.is_async
                            },
                            context={
                                'detection_method': 'filesystem_call_analysis',
                                'library_type': self._get_library_type(pattern.name),
                                'operation_category': self._get_operation_category(pattern.name),
                                'call_context': 'function'
                            },
                            line_numbers=[func_info.line_number],
                            function_name=func_info.name
                        )
                        matches.append(match)
        
        return matches
    
    def _detect_file_paths_in_code(self, ast_result: ASTParseResult) -> List[DetectionMatch]:
        """Detect file paths in string literals throughout the code."""
        matches = []
        
        try:
            # Read the file content to search for file paths
            with open(ast_result.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find all file paths
            path_matches = self.file_path_pattern.finditer(content)
            
            for path_match in path_matches:
                file_path = path_match.group()
                line_number = content[:path_match.start()].count('\n') + 1
                
                # Skip very short matches that might be false positives
                if len(file_path) < 3:
                    continue
                
                # Determine file type from extension
                file_extension = self._extract_file_extension(file_path)
                file_type = self._classify_file_type(file_extension)
                
                match = DetectionMatch(
                    actor_type=ActorType.FILE_SYSTEM,
                    confidence=self.min_path_confidence,
                    pattern_name="file_path_literal",
                    evidence={
                        'file_path': file_path,
                        'file_extension': file_extension,
                        'file_type': file_type,
                        'string_type': 'path'
                    },
                    context={
                        'detection_method': 'file_path_analysis',
                        'path_type': self._classify_path_type(file_path)
                    },
                    line_numbers=[line_number]
                )
                matches.append(match)
        
        except Exception as e:
            logger.warning(f"Failed to detect file paths in {ast_result.file_path}: {e}")
        
        return matches
    
    def _match_filesystem_call(self, call: str, pattern: PatternRule) -> bool:
        """Check if function call matches filesystem pattern."""
        if not pattern.function_calls:
            return False
        
        for call_pattern in pattern.function_calls:
            if call == call_pattern:
                return True
            if call.endswith(call_pattern.split('.')[-1]):  # Match method name
                return True
            if call_pattern in call:  # Substring match for complex calls
                return True
        
        return False
    
    def _extract_operation_type(self, call: str) -> Optional[str]:
        """Extract filesystem operation type from function call."""
        call_lower = call.lower()
        
        for operation_type, keywords in self.file_operations.items():
            if any(keyword in call_lower for keyword in keywords):
                return operation_type
        
        return 'UNKNOWN'
    
    def _extract_file_extension(self, file_path: str) -> Optional[str]:
        """Extract file extension from path."""
        match = self.file_extension_pattern.search(file_path)
        return match.group() if match else None
    
    def _classify_file_type(self, extension: Optional[str]) -> str:
        """Classify file type based on extension."""
        if not extension:
            return 'unknown'
        
        ext_lower = extension.lower()
        
        # Data formats
        if ext_lower in ['.json', '.xml', '.yaml', '.yml', '.toml']:
            return 'data'
        elif ext_lower in ['.csv', '.tsv', '.xlsx', '.xls']:
            return 'spreadsheet'
        elif ext_lower in ['.txt', '.log', '.md', '.rst']:
            return 'text'
        elif ext_lower in ['.py', '.js', '.java', '.cpp', '.c', '.h']:
            return 'code'
        elif ext_lower in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg']:
            return 'image'
        elif ext_lower in ['.mp4', '.avi', '.mkv', '.mov']:
            return 'video'
        elif ext_lower in ['.mp3', '.wav', '.flac', '.ogg']:
            return 'audio'
        elif ext_lower in ['.pdf', '.doc', '.docx', '.ppt', '.pptx']:
            return 'document'
        elif ext_lower in ['.zip', '.tar', '.gz', '.rar', '.7z']:
            return 'archive'
        elif ext_lower in ['.db', '.sqlite', '.sql']:
            return 'database'
        else:
            return 'unknown'
    
    def _classify_path_type(self, path: str) -> str:
        """Classify path type (absolute, relative, etc.)."""
        if path.startswith('/'):
            return 'absolute_unix'
        elif path.match(r'^[a-zA-Z]:'):
            return 'absolute_windows'
        elif path.startswith('./'):
            return 'relative_current'
        elif path.startswith('../'):
            return 'relative_parent'
        elif path.startswith('~/'):
            return 'home_relative'
        else:
            return 'relative'
    
    def _get_library_type(self, pattern_name: str) -> str:
        """Get library type from pattern name."""
        library_map = {
            'pathlib_library': 'pathlib',
            'builtin_file_ops': 'builtin',
            'os_path_library': 'os',
            'shutil_library': 'shutil',
            'glob_library': 'glob',
            'tempfile_library': 'tempfile',
            'json_file_ops': 'json',
            'csv_file_ops': 'csv',
            'yaml_file_ops': 'yaml',
            'pickle_file_ops': 'pickle',
            'io_library': 'io'
        }
        return library_map.get(pattern_name, 'unknown')
    
    def _get_operation_category(self, pattern_name: str) -> str:
        """Get operation category from pattern name."""
        category_map = {
            'pathlib_library': 'path_manipulation',
            'builtin_file_ops': 'file_io',
            'os_path_library': 'path_manipulation',
            'shutil_library': 'file_operations',
            'glob_library': 'pattern_matching',
            'tempfile_library': 'temporary_files',
            'json_file_ops': 'structured_data',
            'csv_file_ops': 'structured_data',
            'yaml_file_ops': 'structured_data',
            'pickle_file_ops': 'serialization',
            'io_library': 'stream_io'
        }
        return category_map.get(pattern_name, 'general')
    
    def _enhance_matches(self, matches: List[DetectionMatch], ast_result: ASTParseResult) -> None:
        """Enhance matches with additional context and information."""
        for match in matches:
            # Add module context
            match.module_name = ast_result.module_name
            
            # Enhance evidence based on detection method
            if match.context.get('detection_method') == 'filesystem_import_analysis':
                self._enhance_import_match(match, ast_result)
            elif match.context.get('detection_method') == 'filesystem_call_analysis':
                self._enhance_call_match(match, ast_result)
            elif match.context.get('detection_method') == 'file_path_analysis':
                self._enhance_path_match(match)
    
    def _enhance_import_match(self, match: DetectionMatch, ast_result: ASTParseResult) -> None:
        """Enhance import-based matches with usage context."""
        imported_module = match.evidence.get('import_module', '')
        
        # Count usage frequency in functions
        usage_count = 0
        for func_info in ast_result.functions:
            for call in func_info.calls:
                if imported_module in call or match.evidence.get('import_name', '') in call:
                    usage_count += 1
        
        match.evidence['usage_frequency'] = usage_count
        match.context['import_usage'] = 'active' if usage_count > 0 else 'imported_only'
        
        # Determine if it's async usage
        if any(async_indicator in imported_module.lower() for async_indicator in ['aio', 'async']):
            match.context['async_usage'] = True
    
    def _enhance_call_match(self, match: DetectionMatch, ast_result: ASTParseResult) -> None:
        """Enhance call-based matches with function context."""
        function_name = match.function_name
        
        if function_name:
            # Find the function info
            func_info = next((f for f in ast_result.functions if f.name == function_name), None)
            if func_info:
                match.evidence['function_complexity'] = func_info.complexity
                match.evidence['function_args'] = func_info.args
                match.evidence['is_async_function'] = func_info.is_async
                
                # Check for error handling patterns
                error_handling = any('except' in call or 'try' in call for call in func_info.calls)
                match.context['has_error_handling'] = error_handling
                
                # Check for context manager usage (with statements)
                context_manager = any('with ' in call for call in func_info.calls)
                match.context['uses_context_manager'] = context_manager
    
    def _enhance_path_match(self, match: DetectionMatch) -> None:
        """Enhance path-based matches with path analysis."""
        file_path = match.evidence.get('file_path', '')
        
        if file_path:
            # Security analysis
            if any(suspicious in file_path.lower() for suspicious in ['../', '~/', '/etc/', '/var/']):
                match.context['security_concern'] = True
            
            # Path complexity
            path_depth = file_path.count('/') + file_path.count('\\')
            match.context['path_depth'] = path_depth
            match.context['path_complexity'] = 'deep' if path_depth > 3 else 'shallow'
    
    def get_filesystem_statistics(self, matches: List[DetectionMatch]) -> Dict[str, Any]:
        """Get filesystem-specific statistics from matches."""
        stats = {
            'total_filesystem_matches': len(matches),
            'library_distribution': {},
            'operation_distribution': {},
            'file_type_distribution': {},
            'path_type_distribution': {},
            'async_usage': 0,
            'context_manager_usage': 0,
            'file_paths_detected': 0
        }
        
        for match in matches:
            # Library distribution
            library = match.context.get('library_type', 'unknown')
            stats['library_distribution'][library] = stats['library_distribution'].get(library, 0) + 1
            
            # Operation distribution
            operation = match.evidence.get('operation_type')
            if operation:
                stats['operation_distribution'][operation] = stats['operation_distribution'].get(operation, 0) + 1
            
            # File type distribution
            file_type = match.evidence.get('file_type')
            if file_type:
                stats['file_type_distribution'][file_type] = stats['file_type_distribution'].get(file_type, 0) + 1
            
            # Path type distribution
            path_type = match.context.get('path_type')
            if path_type:
                stats['path_type_distribution'][path_type] = stats['path_type_distribution'].get(path_type, 0) + 1
            
            # Async usage
            if match.evidence.get('is_async', False) or match.context.get('async_usage', False):
                stats['async_usage'] += 1
            
            # Context manager usage
            if match.context.get('uses_context_manager', False):
                stats['context_manager_usage'] += 1
            
            # File paths detected
            if match.evidence.get('string_type') == 'path':
                stats['file_paths_detected'] += 1
        
        return stats