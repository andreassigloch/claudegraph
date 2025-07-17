#!/usr/bin/env python3
"""
Pattern Matcher for Code Architecture Analyzer

Base pattern matching framework for detecting external actors and dependencies
in Python code based on imports, function calls, and decorators.
"""

import ast
import re
import logging
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from ..core.ast_parser import ASTParseResult, FunctionInfo, ImportInfo, DecoratorInfo
from ..llm.actor_enhancer import ActorEnhancementService, ActorDetection, EnhancedActor

# DatabaseDetector removed to avoid circular import - patterns added directly below

logger = logging.getLogger(__name__)


class ActorType(Enum):
    """Enumeration of supported actor types."""
    HTTP_CLIENT = "HttpClient"
    DATABASE = "Database"
    FILE_SYSTEM = "FileSystem"
    WEB_ENDPOINT = "WebEndpoint"
    MESSAGE_QUEUE = "MessageQueue"
    CONFIG_MANAGER = "ConfigManager"
    CLOUD_SERVICE = "CloudService"
    EXTERNAL_API = "ExternalApi"
    CACHE = "Cache"
    MONITOR = "Monitor"
    UNKNOWN = "Unknown"


class ConfidenceLevel(Enum):
    """Confidence levels for pattern matches."""
    VERY_HIGH = 0.95
    HIGH = 0.85
    MEDIUM = 0.70
    LOW = 0.50
    VERY_LOW = 0.30


@dataclass
class PatternRule:
    """Defines a pattern matching rule for actor detection."""
    name: str
    actor_type: ActorType
    confidence: float
    imports: List[str] = field(default_factory=list)
    function_calls: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    url_patterns: List[str] = field(default_factory=list)
    file_extensions: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    regex_patterns: List[str] = field(default_factory=list)
    context_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionMatch:
    """Represents a pattern match result."""
    actor_type: ActorType
    confidence: float
    pattern_name: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    line_numbers: List[int] = field(default_factory=list)
    function_name: Optional[str] = None
    module_name: Optional[str] = None
    # LLM enhancement attributes
    import_names: List[str] = field(default_factory=list)
    function_calls: List[str] = field(default_factory=list)
    decorator_names: List[str] = field(default_factory=list)
    source_code: Optional[str] = None


@dataclass
class ActorDetectionResult:
    """Complete result of actor detection for a module."""
    module_name: str
    file_path: str
    detected_actors: List[DetectionMatch] = field(default_factory=list)
    high_confidence_matches: List[DetectionMatch] = field(default_factory=list)
    ambiguous_matches: List[DetectionMatch] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)


class BaseDetector(ABC):
    """Abstract base class for specific actor detectors."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.patterns = self._load_patterns()
        self.confidence_threshold = self.config.get('detection', {}).get('confidence_threshold', 0.8)
    
    @abstractmethod
    def _load_patterns(self) -> List[PatternRule]:
        """Load detector-specific patterns."""
        pass
    
    def detect(self, ast_result: ASTParseResult) -> List[DetectionMatch]:
        """Unified detection workflow for all detectors."""
        matches = []
        
        try:
            # Detect from imports
            import_matches = self._detect_imports(ast_result.imports)
            matches.extend(import_matches)
            
            # Detect from function calls
            function_matches = self._detect_function_calls(ast_result.functions)
            matches.extend(function_matches)
            
            # Detect from strings/content (detector-specific)
            content_matches = self._detect_content_patterns(ast_result)
            matches.extend(content_matches)
            
            # Enhance matches with additional context
            self._enhance_matches(matches, ast_result)
            
            logger.debug(f"{self.__class__.__name__} found {len(matches)} matches in {ast_result.module_name}")
            
        except Exception as e:
            logger.error(f"{self.__class__.__name__} detection failed for {ast_result.module_name}: {e}")
        
        return matches
    
    def _detect_imports(self, imports: List[ImportInfo]) -> List[DetectionMatch]:
        """Detect actors from import statements - unified implementation."""
        matches = []
        
        for import_info in imports:
            for pattern in self.patterns:
                if self._match_import(import_info, pattern):
                    match = DetectionMatch(
                        actor_type=pattern.actor_type,
                        confidence=pattern.confidence,
                        pattern_name=pattern.name,
                        evidence={
                            'import_module': import_info.module,
                            'import_name': import_info.name,
                            'import_alias': import_info.alias,
                            'is_from_import': import_info.is_from_import
                        },
                        line_numbers=[import_info.line_number],
                        module_name=import_info.module
                    )
                    matches.append(match)
        
        return matches
    
    def _match_import(self, import_info: ImportInfo, pattern: PatternRule) -> bool:
        """Unified import matching logic."""
        # Check direct module match
        if import_info.module in pattern.imports:
            return True
            
        # Check wildcard patterns
        for pattern_import in pattern.imports:
            if '*' in pattern_import:
                if re.match(pattern_import.replace('*', '.*'), import_info.module):
                    return True
        
        # Check from-import patterns
        if import_info.is_from_import and import_info.name:
            full_import = f"{import_info.module}.{import_info.name}"
            if full_import in pattern.imports:
                return True
        
        return False
    
    @abstractmethod
    def _detect_function_calls(self, functions: List[FunctionInfo]) -> List[DetectionMatch]:
        """Detect actors from function calls - detector-specific implementation."""
        pass
    
    @abstractmethod  
    def _detect_content_patterns(self, ast_result: ASTParseResult) -> List[DetectionMatch]:
        """Detect actors from content patterns (strings, URLs, etc.) - detector-specific."""
        pass
    
    def _enhance_matches(self, matches: List[DetectionMatch], ast_result: ASTParseResult) -> None:
        """Enhance matches with additional context - unified implementation."""
        for match in matches:
            # Count usage frequency
            usage_count = sum(1 for func in ast_result.functions 
                            if any(call in [fc.name for fc in func.function_calls] 
                                  for call in self._get_pattern_function_calls(match.pattern_name)))
            
            match.evidence['usage_frequency'] = usage_count
            match.evidence['module_size'] = len(ast_result.functions)
            
            # Adjust confidence based on usage
            if usage_count > 3:
                match.confidence = min(0.99, match.confidence * 1.1)
            elif usage_count == 0:
                match.confidence = max(0.3, match.confidence * 0.8)
    
    def _get_pattern_function_calls(self, pattern_name: str) -> List[str]:
        """Get function calls for a specific pattern."""
        for pattern in self.patterns:
            if pattern.name == pattern_name:
                return pattern.function_calls
        return []
    
    def get_pattern_count(self) -> int:
        """Get number of loaded patterns."""
        return len(self.patterns)
    
    def get_confidence_threshold(self) -> float:
        """Get confidence threshold for this detector."""
        return self.confidence_threshold


class PatternMatcher:
    """
    Main pattern matching engine for actor detection.
    
    Coordinates multiple specialized detectors and provides unified
    actor detection across different pattern types.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, enhancement_service: Optional[ActorEnhancementService] = None):
        """Initialize pattern matcher with configuration."""
        self.config = config or {}
        self.enhancement_service = enhancement_service
        
        # Detection settings
        detection_config = self.config.get('detection', {})
        self.confidence_threshold = detection_config.get('confidence_threshold', 0.8)
        self.min_evidence_count = detection_config.get('min_evidence_count', 1)
        self.aggregate_scores = detection_config.get('aggregate_scores', True)
        
        # Pattern confidence scores from config
        pattern_confidence = self.config.get('deterministic', {}).get('pattern_confidence', {})
        self.default_confidences = {
            'http_client': pattern_confidence.get('http_client', 0.95),
            'database': pattern_confidence.get('database', 0.90),
            'file_system': pattern_confidence.get('file_system', 0.98),
            'web_endpoint': pattern_confidence.get('web_endpoint', 0.99),
            'message_queue': pattern_confidence.get('message_queue', 0.85),
            'cloud_service': pattern_confidence.get('cloud_service', 0.80)
        }
        
        # Load base patterns
        self.base_patterns = self._load_base_patterns()
        
        # Statistics
        self.detection_stats = {
            'total_detections': 0,
            'high_confidence_detections': 0,
            'ambiguous_detections': 0,
            'pattern_matches': {},
            'actor_type_counts': {}
        }
        
        logger.info(f"Loaded {len(self.base_patterns)} base detection patterns")
    
    def detect_actors(self, ast_result: ASTParseResult) -> ActorDetectionResult:
        """
        Detect all actors in an AST parsing result.
        
        Args:
            ast_result: Parsed AST data from a Python file
            
        Returns:
            ActorDetectionResult with all detected actors
        """
        result = ActorDetectionResult(
            module_name=ast_result.module_name,
            file_path=str(ast_result.file_path)
        )
        
        try:
            # Run detection on imports
            import_matches = self._detect_from_imports(ast_result.imports)
            result.detected_actors.extend(import_matches)
            
            # Run detection on function calls
            function_matches = self._detect_from_functions(ast_result.functions)
            result.detected_actors.extend(function_matches)
            
            # Run detection on decorators
            decorator_matches = self._detect_from_decorators(ast_result)
            result.detected_actors.extend(decorator_matches)
            
            # Aggregate and classify results
            self._classify_matches(result)
            
            # Enhance actors with meaningful names and descriptions
            if self.enhancement_service:
                try:
                    result = self._enhance_detected_actors(result, ast_result)
                except Exception as e:
                    logger.warning(f"Actor enhancement failed for {ast_result.module_name}: {e}")
            
            # Generate statistics
            result.statistics = self._generate_detection_stats(result)
            
            # Update global stats
            self._update_global_stats(result)
            
            logger.debug(f"Detected {len(result.detected_actors)} actors in {ast_result.module_name}")
            
        except Exception as e:
            logger.error(f"Actor detection failed for {ast_result.module_name}: {e}")
            result.statistics['error'] = str(e)
        
        return result
    
    def _detect_from_imports(self, imports: List[ImportInfo]) -> List[DetectionMatch]:
        """Detect actors based on import statements."""
        matches = []
        
        for import_info in imports:
            for pattern in self.base_patterns:
                if self._match_import_pattern(import_info, pattern):
                    match = DetectionMatch(
                        actor_type=pattern.actor_type,
                        confidence=pattern.confidence,
                        pattern_name=pattern.name,
                        evidence={
                            'import_module': import_info.module,
                            'import_name': import_info.name,
                            'is_from_import': import_info.is_from_import
                        },
                        context={'detection_method': 'import_analysis'},
                        line_numbers=[import_info.line_number],
                        module_name=import_info.module,
                        import_names=[import_info.name or import_info.module],
                        function_calls=[],
                        decorator_names=[],
                        source_code=f"import {import_info.module}" if not import_info.is_from_import else f"from {import_info.module} import {import_info.name}"
                    )
                    matches.append(match)
        
        return matches
    
    def _detect_from_functions(self, functions: List[FunctionInfo]) -> List[DetectionMatch]:
        """Detect actors based on function calls within functions."""
        matches = []
        
        for func_info in functions:
            for call in func_info.calls:
                for pattern in self.base_patterns:
                    if self._match_function_call_pattern(call, pattern):
                        match = DetectionMatch(
                            actor_type=pattern.actor_type,
                            confidence=pattern.confidence,
                            pattern_name=pattern.name,
                            evidence={
                                'function_call': call,
                                'containing_function': func_info.name
                            },
                            context={'detection_method': 'function_call_analysis'},
                            line_numbers=[func_info.line_number],
                            function_name=func_info.name,
                            import_names=[],
                            function_calls=[call],
                            decorator_names=[],
                            source_code=call
                        )
                        matches.append(match)
        
        return matches
    
    def _detect_from_decorators(self, ast_result: ASTParseResult) -> List[DetectionMatch]:
        """Detect actors based on decorators (especially web endpoints)."""
        matches = []
        
        for func_info in ast_result.functions:
            for decorator in func_info.decorators:
                for pattern in self.base_patterns:
                    if self._match_decorator_pattern(decorator, pattern):
                        match = DetectionMatch(
                            actor_type=pattern.actor_type,
                            confidence=pattern.confidence,
                            pattern_name=pattern.name,
                            evidence={
                                'decorator_name': decorator.name,
                                'decorator_args': decorator.args,
                                'function_name': func_info.name
                            },
                            context={'detection_method': 'decorator_analysis'},
                            line_numbers=[func_info.line_number],
                            function_name=func_info.name,
                            import_names=[],
                            function_calls=[],
                            decorator_names=[decorator.name],
                            source_code=f"@{decorator.name}"
                        )
                        matches.append(match)
        
        return matches
    
    def _match_import_pattern(self, import_info: ImportInfo, pattern: PatternRule) -> bool:
        """Check if import matches a pattern rule."""
        if not pattern.imports:
            return False
        
        # Check module name
        for import_pattern in pattern.imports:
            if self._matches_pattern(import_info.module, import_pattern):
                return True
            
            # For from imports, also check the imported name
            if import_info.is_from_import and import_info.name:
                full_import = f"{import_info.module}.{import_info.name}"
                if self._matches_pattern(full_import, import_pattern):
                    return True
        
        return False
    
    def _match_function_call_pattern(self, call: str, pattern: PatternRule) -> bool:
        """Check if function call matches a pattern rule."""
        if not pattern.function_calls:
            return False
        
        for call_pattern in pattern.function_calls:
            if self._matches_pattern(call, call_pattern):
                return True
        
        return False
    
    def _match_decorator_pattern(self, decorator: DecoratorInfo, pattern: PatternRule) -> bool:
        """Check if decorator matches a pattern rule."""
        if not pattern.decorators:
            return False
        
        for decorator_pattern in pattern.decorators:
            if self._matches_pattern(decorator.name, decorator_pattern):
                return True
        
        return False
    
    def _matches_pattern(self, text: str, pattern: str) -> bool:
        """Check if text matches a pattern (supports wildcards and regex)."""
        if not text or not pattern:
            return False
        
        # Exact match
        if pattern == text:
            return True
        
        # Wildcard match
        if '*' in pattern:
            import fnmatch
            return fnmatch.fnmatch(text, pattern)
        
        # Substring match
        if pattern in text:
            return True
        
        # Regex match for complex patterns
        if pattern.startswith('^') or pattern.endswith('$'):
            try:
                return bool(re.match(pattern, text))
            except re.error:
                pass
        
        return False
    
    def _classify_matches(self, result: ActorDetectionResult) -> None:
        """Classify matches into high confidence and ambiguous categories."""
        for match in result.detected_actors:
            if match.confidence >= self.confidence_threshold:
                result.high_confidence_matches.append(match)
            else:
                result.ambiguous_matches.append(match)
    
    def _generate_detection_stats(self, result: ActorDetectionResult) -> Dict[str, Any]:
        """Generate statistics for detection result."""
        stats = {
            'total_matches': len(result.detected_actors),
            'high_confidence_count': len(result.high_confidence_matches),
            'ambiguous_count': len(result.ambiguous_matches),
            'confidence_distribution': {},
            'actor_type_distribution': {},
            'detection_methods': {},
            'average_confidence': 0.0
        }
        
        if not result.detected_actors:
            return stats
        
        # Calculate confidence distribution
        confidences = [match.confidence for match in result.detected_actors]
        stats['average_confidence'] = sum(confidences) / len(confidences)
        
        # Count by actor type
        for match in result.detected_actors:
            actor_type = match.actor_type.value
            stats['actor_type_distribution'][actor_type] = stats['actor_type_distribution'].get(actor_type, 0) + 1
        
        # Count by detection method
        for match in result.detected_actors:
            method = match.context.get('detection_method', 'unknown')
            stats['detection_methods'][method] = stats['detection_methods'].get(method, 0) + 1
        
        return stats
    
    def _enhance_detected_actors(self, result: ActorDetectionResult, ast_result: ASTParseResult) -> ActorDetectionResult:
        """Enhance detected actors with meaningful names and descriptions using LLM."""
        enhanced_matches = []
        
        for match in result.detected_actors:
            try:
                # Create actor detection for enhancement
                detection = self._create_actor_detection(match, ast_result)
                
                # Generate original generic name
                original_name = self._generate_generic_name(match)
                
                # Enhance with LLM
                enhanced = self.enhancement_service.enhance_actor(detection, original_name)
                
                # Update match with enhanced information
                enhanced_match = self._update_match_with_enhancement(match, enhanced)
                enhanced_matches.append(enhanced_match)
                
            except Exception as e:
                logger.debug(f"Failed to enhance actor {match.pattern_name}: {e}")
                enhanced_matches.append(match)  # Keep original on failure
        
        result.detected_actors = enhanced_matches
        return result
    
    def _create_actor_detection(self, match: DetectionMatch, ast_result: ASTParseResult) -> ActorDetection:
        """Create ActorDetection from DetectionMatch for enhancement."""
        # Extract code snippet and context
        code_snippet = self._extract_code_snippet(match, ast_result)
        function_context = self._extract_function_context(match, ast_result)
        url_or_target = self._extract_target_info(match)
        
        return ActorDetection(
            type=match.actor_type.value,
            library=self._extract_library_name(match),
            code_snippet=code_snippet,
            function_context=function_context,
            file_path=ast_result.file_path,
            confidence=match.confidence,
            url_or_target=url_or_target
        )
    
    def _extract_code_snippet(self, match: DetectionMatch, ast_result: ASTParseResult) -> str:
        """Extract relevant code snippet for the match."""
        # Try to get the actual code that triggered the match
        if hasattr(match, 'source_code') and match.source_code:
            return match.source_code[:200]  # Limit length
        
        # Fallback to constructing from available data
        if match.import_names:
            return f"import {', '.join(match.import_names)}"
        elif match.function_calls:
            return f"{match.function_calls[0]}(...)"
        elif match.decorator_names:
            return f"@{match.decorator_names[0]}"
        
        return f"{match.actor_type.value} usage detected"
    
    def _extract_function_context(self, match: DetectionMatch, ast_result: ASTParseResult) -> str:
        """Extract function context where the match was found."""
        # Try to find the function that contains this match
        if hasattr(match, 'function_name') and match.function_name:
            return match.function_name
        
        # Look for function calls that might be relevant
        if match.function_calls:
            return match.function_calls[0].split('.')[0] if '.' in match.function_calls[0] else match.function_calls[0]
        
        return ""
    
    def _extract_target_info(self, match: DetectionMatch) -> str:
        """Extract target URL or identifier from the match."""
        # Look for URL patterns in function calls or import names
        if match.function_calls:
            for call in match.function_calls:
                if 'http' in call.lower() or 'api' in call.lower():
                    return call
        
        if match.import_names:
            for import_name in match.import_names:
                if any(service in import_name.lower() for service in ['stripe', 'github', 'aws', 'google']):
                    return import_name
        
        return ""
    
    def _extract_library_name(self, match: DetectionMatch) -> str:
        """Extract the primary library name from the match."""
        if match.import_names:
            # Get the first/primary import
            primary_import = match.import_names[0]
            return primary_import.split('.')[0]  # Get base library name
        
        if match.function_calls:
            # Extract library from function call
            call = match.function_calls[0]
            if '.' in call:
                return call.split('.')[0]
        
        return match.actor_type.value.lower()
    
    def _generate_generic_name(self, match: DetectionMatch) -> str:
        """Generate the original generic name that would have been used."""
        base_name = match.actor_type.value.lower()
        
        # Add 'unknown' suffix like the original system
        if match.import_names or match.function_calls:
            return f"{base_name}unknown"
        
        return f"{base_name}unknown"
    
    def _update_match_with_enhancement(self, match: DetectionMatch, enhanced: EnhancedActor) -> DetectionMatch:
        """Update DetectionMatch with enhancement information."""
        # Add enhancement metadata to the match
        if not hasattr(match, 'enhanced_name'):
            match.enhanced_name = enhanced.name
        if not hasattr(match, 'enhanced_description'):
            match.enhanced_description = enhanced.description
        if not hasattr(match, 'enhancement_confidence'):
            match.enhancement_confidence = enhanced.enhancement_confidence
        if not hasattr(match, 'was_enhanced'):
            match.was_enhanced = enhanced.enhanced
        
        return match
    
    def _update_global_stats(self, result: ActorDetectionResult) -> None:
        """Update global detection statistics."""
        self.detection_stats['total_detections'] += len(result.detected_actors)
        self.detection_stats['high_confidence_detections'] += len(result.high_confidence_matches)
        self.detection_stats['ambiguous_detections'] += len(result.ambiguous_matches)
        
        # Update actor type counts
        for match in result.detected_actors:
            actor_type = match.actor_type.value
            self.detection_stats['actor_type_counts'][actor_type] = self.detection_stats['actor_type_counts'].get(actor_type, 0) + 1
    
    def _load_base_patterns(self) -> List[PatternRule]:
        """Load base detection patterns for common actors."""
        patterns = []
        
        # HTTP Client patterns
        patterns.extend([
            PatternRule(
                name="requests_library",
                actor_type=ActorType.HTTP_CLIENT,
                confidence=self.default_confidences['http_client'],
                imports=["requests", "requests.*"],
                function_calls=["requests.get", "requests.post", "requests.put", "requests.delete", "requests.request"]
            ),
            PatternRule(
                name="urllib_library",
                actor_type=ActorType.HTTP_CLIENT,
                confidence=self.default_confidences['http_client'] - 0.05,
                imports=["urllib.request", "urllib.*"],
                function_calls=["urllib.request.urlopen", "urllib.request.Request"]
            ),
            PatternRule(
                name="httpx_library",
                actor_type=ActorType.HTTP_CLIENT,
                confidence=self.default_confidences['http_client'],
                imports=["httpx", "httpx.*"],
                function_calls=["httpx.get", "httpx.post", "httpx.Client"]
            ),
            PatternRule(
                name="aiohttp_library",
                actor_type=ActorType.HTTP_CLIENT,
                confidence=self.default_confidences['http_client'],
                imports=["aiohttp", "aiohttp.*"],
                function_calls=["aiohttp.ClientSession", "session.get", "session.post"]
            )
        ])
        
        # Database patterns
        patterns.extend([
            PatternRule(
                name="sqlite3_library",
                actor_type=ActorType.DATABASE,
                confidence=self.default_confidences['database'],
                imports=["sqlite3"],
                function_calls=["sqlite3.connect", "cursor.execute", "connection.execute"]
            ),
            PatternRule(
                name="psycopg2_library",
                actor_type=ActorType.DATABASE,
                confidence=self.default_confidences['database'],
                imports=["psycopg2", "psycopg2.*"],
                function_calls=["psycopg2.connect", "cursor.execute"]
            ),
            PatternRule(
                name="pymongo_library",
                actor_type=ActorType.DATABASE,
                confidence=self.default_confidences['database'],
                imports=["pymongo", "pymongo.*"],
                function_calls=["MongoClient", "collection.find", "collection.insert"]
            ),
            PatternRule(
                name="redis_library",
                actor_type=ActorType.DATABASE,
                confidence=self.default_confidences['database'],
                imports=["redis"],
                function_calls=["redis.Redis", "redis.StrictRedis", "redis_client.get", "redis_client.set"]
            )
        ])
        
        # File System patterns
        patterns.extend([
            PatternRule(
                name="pathlib_library",
                actor_type=ActorType.FILE_SYSTEM,
                confidence=self.default_confidences['file_system'],
                imports=["pathlib", "pathlib.*"],
                function_calls=["Path", "Path.open", "Path.read_text", "Path.write_text"]
            ),
            PatternRule(
                name="builtin_file_ops",
                actor_type=ActorType.FILE_SYSTEM,
                confidence=self.default_confidences['file_system'] - 0.1,
                function_calls=["open(", "file.read", "file.write", "file.close"]
            ),
            PatternRule(
                name="os_path_library",
                actor_type=ActorType.FILE_SYSTEM,
                confidence=self.default_confidences['file_system'] - 0.05,
                imports=["os.path", "os"],
                function_calls=["os.path.join", "os.path.exists", "os.listdir", "os.makedirs"]
            ),
            PatternRule(
                name="shutil_library",
                actor_type=ActorType.FILE_SYSTEM,
                confidence=self.default_confidences['file_system'],
                imports=["shutil"],
                function_calls=["shutil.copy", "shutil.move", "shutil.rmtree"]
            )
        ])
        
        # Web Endpoint patterns
        patterns.extend([
            PatternRule(
                name="flask_routes",
                actor_type=ActorType.WEB_ENDPOINT,
                confidence=self.default_confidences['web_endpoint'],
                imports=["flask", "flask.*"],
                decorators=["@app.route", "@router.route", "@bp.route"],
                function_calls=["Flask", "Blueprint"]
            ),
            PatternRule(
                name="fastapi_routes",
                actor_type=ActorType.WEB_ENDPOINT,
                confidence=self.default_confidences['web_endpoint'],
                imports=["fastapi", "fastapi.*"],
                decorators=["@app.get", "@app.post", "@app.put", "@app.delete", "@router.get", "@router.post"],
                function_calls=["FastAPI", "APIRouter"]
            ),
            PatternRule(
                name="django_views",
                actor_type=ActorType.WEB_ENDPOINT,
                confidence=self.default_confidences['web_endpoint'] - 0.05,
                imports=["django.*", "django.http"],
                class_names=["View", "APIView", "ViewSet"],
                function_calls=["HttpResponse", "JsonResponse"]
            )
        ])
        
        # Message Queue patterns
        patterns.extend([
            PatternRule(
                name="celery_tasks",
                actor_type=ActorType.MESSAGE_QUEUE,
                confidence=self.default_confidences['message_queue'],
                imports=["celery", "celery.*"],
                decorators=["@celery.task", "@task", "@shared_task"],
                function_calls=["Celery", "apply_async"]
            ),
            PatternRule(
                name="rabbitmq_library",
                actor_type=ActorType.MESSAGE_QUEUE,
                confidence=self.default_confidences['message_queue'],
                imports=["pika", "kombu"],
                function_calls=["pika.BlockingConnection", "Connection"]
            )
        ])
        
        # Cloud Service patterns
        patterns.extend([
            PatternRule(
                name="aws_boto3",
                actor_type=ActorType.CLOUD_SERVICE,
                confidence=self.default_confidences['cloud_service'],
                imports=["boto3", "botocore"],
                function_calls=["boto3.client", "boto3.resource", "boto3.Session"]
            ),
            PatternRule(
                name="google_cloud",
                actor_type=ActorType.CLOUD_SERVICE,
                confidence=self.default_confidences['cloud_service'],
                imports=["google.cloud.*", "google.auth"],
                function_calls=["storage.Client", "pubsub.PublisherClient"]
            )
        ])
        
        # Database patterns (added directly to avoid circular imports)
        patterns.extend([
            PatternRule(
                name="sqlite3_builtin",
                actor_type=ActorType.DATABASE,
                confidence=self.default_confidences['database'],
                imports=["sqlite3"],
                function_calls=[
                    "sqlite3.connect", "sqlite3.Connection", "connection.execute",
                    "connection.executemany", "connection.executescript", "cursor.execute",
                    "cursor.executemany", "cursor.fetchone", "cursor.fetchall", "cursor.fetchmany"
                ],
                keywords=["sqlite3", "cursor", "execute", "fetch"]
            ),
            PatternRule(
                name="neo4j_graph_database",
                actor_type=ActorType.DATABASE,
                confidence=self.default_confidences['database'],
                imports=["neo4j", "neo4j.*", "neo4j.exceptions"],
                function_calls=[
                    "GraphDatabase.driver", "neo4j.GraphDatabase.driver", "driver.session",
                    "session.run", "session.execute_read", "session.execute_write",
                    "tx.run", "result.single", "result.data", "result.consume",
                    "session.begin_transaction", "session.close", "driver.close"
                ],
                keywords=["neo4j", "GraphDatabase", "cypher", "session", "tx", "graph"]
            ),
            PatternRule(
                name="psycopg2_postgresql",
                actor_type=ActorType.DATABASE,
                confidence=self.default_confidences['database'],
                imports=["psycopg2", "psycopg2.*", "psycopg2.pool"],
                function_calls=[
                    "psycopg2.connect", "psycopg2.Connection", "cursor.execute",
                    "cursor.executemany", "cursor.fetchone", "cursor.fetchall",
                    "connection.commit", "connection.rollback", "SimpleConnectionPool"
                ],
                keywords=["psycopg2", "postgresql", "postgres", "cursor"]
            ),
            PatternRule(
                name="pymongo_mongodb",
                actor_type=ActorType.DATABASE,
                confidence=self.default_confidences['database'],
                imports=["pymongo", "pymongo.*"],
                function_calls=[
                    "MongoClient", "client.get_database", "db.get_collection",
                    "collection.insert_one", "collection.find", "collection.update_one",
                    "collection.delete_one", "collection.aggregate"
                ],
                keywords=["pymongo", "mongodb", "mongo", "collection"]
            )
        ])
        
        logger.info(f"Loaded {len(patterns)} base detection patterns")
        return patterns
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get global detection statistics."""
        return self.detection_stats.copy()
    
    def get_supported_actor_types(self) -> List[str]:
        """Get list of supported actor types."""
        return [actor_type.value for actor_type in ActorType]
    
    def get_pattern_count(self) -> int:
        """Get total number of loaded patterns."""
        return len(self.base_patterns)