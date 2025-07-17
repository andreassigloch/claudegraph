#!/usr/bin/env python3
"""
HTTP Client Detector for Code Architecture Analyzer

Specialized detector for identifying HTTP client usage patterns in Python code.
Detects requests, urllib, httpx, aiohttp and other HTTP client libraries.
"""

import re
import ast
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse

from .pattern_matcher import BaseDetector, PatternRule, DetectionMatch, ActorType, ConfidenceLevel
from ..core.ast_parser import ASTParseResult, FunctionInfo, ImportInfo

logger = logging.getLogger(__name__)


@dataclass
class HttpPattern:
    """HTTP-specific pattern information."""
    library_name: str
    methods: List[str]
    confidence: float
    async_support: bool = False
    session_based: bool = False
    auth_patterns: List[str] = None


class HttpDetector(BaseDetector):
    """
    Specialized detector for HTTP client patterns.
    
    Identifies usage of HTTP client libraries and extracts information about
    HTTP requests, methods, URLs, and authentication patterns.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize HTTP detector with configuration."""
        super().__init__(config)
        
        # HTTP-specific configuration
        http_config = self.config.get('detection', {}).get('http', {})
        self.detect_urls = http_config.get('detect_urls', True)
        self.extract_endpoints = http_config.get('extract_endpoints', True)
        self.min_url_confidence = http_config.get('min_url_confidence', 0.7)
        
        # URL pattern for detecting HTTP URLs in strings
        self.url_pattern = re.compile(
            r'https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?',
            re.IGNORECASE
        )
        
        # HTTP method patterns
        self.http_methods = {
            'GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS', 'TRACE'
        }
        
        logger.debug("HTTP detector initialized")
    
    def _load_patterns(self) -> List[PatternRule]:
        """Load HTTP client detection patterns."""
        patterns = []
        
        # Get confidence scores from config
        base_confidence = self.config.get('deterministic', {}).get('pattern_confidence', {}).get('http_client', 0.95)
        
        # Requests library patterns
        patterns.append(PatternRule(
            name="requests_library",
            actor_type=ActorType.HTTP_CLIENT,
            confidence=base_confidence,
            imports=["requests", "requests.*"],
            function_calls=[
                "requests.get", "requests.post", "requests.put", "requests.delete",
                "requests.patch", "requests.head", "requests.options", "requests.request"
            ],
            keywords=["requests", "session", "response"]
        ))
        
        # Requests session patterns
        patterns.append(PatternRule(
            name="requests_session",
            actor_type=ActorType.HTTP_CLIENT,
            confidence=base_confidence,
            imports=["requests"],
            function_calls=[
                "requests.Session", "requests.session", "session.get", "session.post",
                "session.put", "session.delete", "session.request"
            ],
            keywords=["Session", "session"]
        ))
        
        # urllib patterns
        patterns.append(PatternRule(
            name="urllib_request",
            actor_type=ActorType.HTTP_CLIENT,
            confidence=base_confidence - 0.05,
            imports=["urllib.request", "urllib.parse", "urllib.error", "urllib"],
            function_calls=[
                "urllib.request.urlopen", "urllib.request.Request", "urllib.request.urlretrieve",
                "urlopen", "Request"
            ],
            keywords=["urllib", "urlopen", "Request"]
        ))
        
        # httpx patterns (modern requests alternative)
        patterns.append(PatternRule(
            name="httpx_library",
            actor_type=ActorType.HTTP_CLIENT,
            confidence=base_confidence,
            imports=["httpx", "httpx.*"],
            function_calls=[
                "httpx.get", "httpx.post", "httpx.put", "httpx.delete",
                "httpx.Client", "httpx.AsyncClient", "client.get", "client.post"
            ],
            keywords=["httpx", "Client", "AsyncClient"]
        ))
        
        # aiohttp patterns (async HTTP)
        patterns.append(PatternRule(
            name="aiohttp_client",
            actor_type=ActorType.HTTP_CLIENT,
            confidence=base_confidence,
            imports=["aiohttp", "aiohttp.client", "aiohttp.*"],
            function_calls=[
                "aiohttp.ClientSession", "aiohttp.request", "ClientSession",
                "session.get", "session.post", "session.put", "session.delete"
            ],
            keywords=["aiohttp", "ClientSession", "async"]
        ))
        
        # http.client (built-in) patterns
        patterns.append(PatternRule(
            name="http_client_builtin",
            actor_type=ActorType.HTTP_CLIENT,
            confidence=base_confidence - 0.15,
            imports=["http.client", "http"],
            function_calls=[
                "http.client.HTTPConnection", "http.client.HTTPSConnection",
                "HTTPConnection", "HTTPSConnection", "request", "getresponse"
            ],
            keywords=["HTTPConnection", "HTTPSConnection"]
        ))
        
        # tornado HTTP client
        patterns.append(PatternRule(
            name="tornado_httpclient",
            actor_type=ActorType.HTTP_CLIENT,
            confidence=base_confidence - 0.05,
            imports=["tornado.httpclient", "tornado.*"],
            function_calls=[
                "tornado.httpclient.HTTPClient", "tornado.httpclient.AsyncHTTPClient",
                "HTTPClient", "AsyncHTTPClient", "fetch"
            ],
            keywords=["tornado", "httpclient", "fetch"]
        ))
        
        # grequests (asynchronous requests)
        patterns.append(PatternRule(
            name="grequests_library",
            actor_type=ActorType.HTTP_CLIENT,
            confidence=base_confidence - 0.1,
            imports=["grequests"],
            function_calls=[
                "grequests.get", "grequests.post", "grequests.map", "grequests.imap"
            ],
            keywords=["grequests", "async"]
        ))
        
        logger.info(f"Loaded {len(patterns)} HTTP detection patterns")
        return patterns
    
    def _detect_function_calls(self, functions: List[FunctionInfo]) -> List[DetectionMatch]:
        """Detect HTTP usage from function calls."""
        return self._detect_http_calls(functions)
    
    def _detect_content_patterns(self, ast_result: ASTParseResult) -> List[DetectionMatch]:
        """Detect URLs in string literals."""
        if self.detect_urls:
            return self._detect_urls_in_code(ast_result)
        return []
    
    
    def _detect_http_calls(self, functions: List[FunctionInfo]) -> List[DetectionMatch]:
        """Detect HTTP client calls within functions."""
        matches = []
        
        for func_info in functions:
            for call in func_info.calls:
                # Check against HTTP patterns
                for pattern in self.patterns:
                    if self._match_http_call(call, pattern):
                        match = DetectionMatch(
                            actor_type=ActorType.HTTP_CLIENT,
                            confidence=pattern.confidence,
                            pattern_name=pattern.name,
                            evidence={
                                'function_call': call,
                                'http_method': self._extract_http_method(call),
                                'containing_function': func_info.name,
                                'is_async': func_info.is_async
                            },
                            context={
                                'detection_method': 'http_call_analysis',
                                'library_type': self._get_library_type(pattern.name),
                                'call_context': 'function'
                            },
                            line_numbers=[func_info.line_number],
                            function_name=func_info.name
                        )
                        matches.append(match)
        
        return matches
    
    def _detect_urls_in_code(self, ast_result: ASTParseResult) -> List[DetectionMatch]:
        """Detect HTTP URLs in string literals throughout the code."""
        matches = []
        
        try:
            # Read the file content to search for URLs
            with open(ast_result.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find all HTTP URLs
            url_matches = self.url_pattern.finditer(content)
            
            for url_match in url_matches:
                url = url_match.group()
                line_number = content[:url_match.start()].count('\n') + 1
                
                # Parse URL for additional context
                parsed_url = urlparse(url)
                
                match = DetectionMatch(
                    actor_type=ActorType.HTTP_CLIENT,
                    confidence=self.min_url_confidence,
                    pattern_name="url_literal",
                    evidence={
                        'url': url,
                        'scheme': parsed_url.scheme,
                        'netloc': parsed_url.netloc,
                        'path': parsed_url.path,
                        'query': parsed_url.query
                    },
                    context={
                        'detection_method': 'url_literal_analysis',
                        'url_type': 'hardcoded'
                    },
                    line_numbers=[line_number]
                )
                matches.append(match)
        
        except Exception as e:
            logger.warning(f"Failed to detect URLs in {ast_result.file_path}: {e}")
        
        return matches
    
    
    def _match_http_call(self, call: str, pattern: PatternRule) -> bool:
        """Check if function call matches HTTP pattern."""
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
    
    def _extract_http_method(self, call: str) -> Optional[str]:
        """Extract HTTP method from function call."""
        call_lower = call.lower()
        
        for method in self.http_methods:
            if method.lower() in call_lower:
                return method
        
        # Check for generic request calls
        if 'request' in call_lower:
            return 'REQUEST'
        
        return None
    
    def _get_library_type(self, pattern_name: str) -> str:
        """Get library type from pattern name."""
        library_map = {
            'requests_library': 'requests',
            'requests_session': 'requests',
            'urllib_request': 'urllib',
            'httpx_library': 'httpx',
            'aiohttp_client': 'aiohttp',
            'http_client_builtin': 'http.client',
            'tornado_httpclient': 'tornado',
            'grequests_library': 'grequests'
        }
        return library_map.get(pattern_name, 'unknown')
    
    def _enhance_matches(self, matches: List[DetectionMatch], ast_result: ASTParseResult) -> None:
        """Enhance matches with additional context and information."""
        for match in matches:
            # Add module context
            match.module_name = ast_result.module_name
            
            # Enhance evidence based on detection method
            if match.context.get('detection_method') == 'http_import_analysis':
                self._enhance_import_match(match, ast_result)
            elif match.context.get('detection_method') == 'http_call_analysis':
                self._enhance_call_match(match, ast_result)
            elif match.context.get('detection_method') == 'url_literal_analysis':
                self._enhance_url_match(match)
    
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
    
    def _enhance_url_match(self, match: DetectionMatch) -> None:
        """Enhance URL-based matches with URL analysis."""
        url = match.evidence.get('url', '')
        
        if url:
            # Determine if it's an API endpoint
            path = match.evidence.get('path', '')
            if any(api_indicator in path.lower() for api_indicator in ['/api/', '/v1/', '/v2/', '/rest/']):
                match.context['endpoint_type'] = 'api'
            elif path.endswith('.json') or path.endswith('.xml'):
                match.context['endpoint_type'] = 'data'
            else:
                match.context['endpoint_type'] = 'web'
            
            # Check for parameters
            query = match.evidence.get('query', '')
            if query:
                match.context['has_parameters'] = True
                match.evidence['parameter_count'] = len(query.split('&'))
    
    def get_http_statistics(self, matches: List[DetectionMatch]) -> Dict[str, Any]:
        """Get HTTP-specific statistics from matches."""
        stats = {
            'total_http_matches': len(matches),
            'library_distribution': {},
            'method_distribution': {},
            'url_count': 0,
            'async_usage': 0,
            'session_usage': 0
        }
        
        for match in matches:
            # Library distribution
            library = match.context.get('library_type', 'unknown')
            stats['library_distribution'][library] = stats['library_distribution'].get(library, 0) + 1
            
            # Method distribution
            method = match.evidence.get('http_method')
            if method:
                stats['method_distribution'][method] = stats['method_distribution'].get(method, 0) + 1
            
            # URL count
            if 'url' in match.evidence:
                stats['url_count'] += 1
            
            # Async usage
            if match.evidence.get('is_async', False):
                stats['async_usage'] += 1
            
            # Session usage
            if 'session' in match.pattern_name.lower():
                stats['session_usage'] += 1
        
        return stats