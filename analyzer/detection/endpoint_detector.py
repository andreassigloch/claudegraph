#!/usr/bin/env python3
"""
Endpoint Detector for Code Architecture Analyzer

Specialized detector for identifying web endpoint patterns in Python code.
Detects Flask routes, FastAPI endpoints, Django views and other web framework patterns.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass

from .pattern_matcher import BaseDetector, PatternRule, DetectionMatch, ActorType, ConfidenceLevel
from ..core.ast_parser import ASTParseResult, FunctionInfo, ImportInfo, DecoratorInfo

logger = logging.getLogger(__name__)


@dataclass
class EndpointPattern:
    """Endpoint-specific pattern information."""
    framework_name: str
    http_methods: List[str]
    confidence: float
    async_support: bool = False
    decorator_based: bool = True
    class_based: bool = False


class EndpointDetector(BaseDetector):
    """
    Specialized detector for web endpoint patterns.
    
    Identifies usage of web framework decorators and patterns that create
    HTTP endpoints and API routes.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize endpoint detector with configuration."""
        super().__init__(config)
        
        # Endpoint-specific configuration
        endpoint_config = self.config.get('detection', {}).get('endpoint', {})
        self.detect_routes = endpoint_config.get('detect_routes', True)
        self.extract_http_methods = endpoint_config.get('extract_http_methods', True)
        self.min_endpoint_confidence = endpoint_config.get('min_endpoint_confidence', 0.9)
        
        # HTTP method patterns
        self.http_methods = {
            'GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS', 'TRACE'
        }
        
        # Route pattern for extracting routes from decorators
        self.route_pattern = re.compile(
            r'["\']([/\w\-<>:]+)["\']',
            re.IGNORECASE
        )
        
        # Common web framework indicators
        self.framework_indicators = {
            'flask': ['Flask', 'Blueprint', 'app.route', 'bp.route'],
            'fastapi': ['FastAPI', 'APIRouter', 'app.get', 'app.post'],
            'django': ['View', 'APIView', 'ViewSet', 'HttpResponse'],
            'tornado': ['RequestHandler', 'Application'],
            'bottle': ['route', 'bottle.route'],
            'sanic': ['Sanic', 'Blueprint']
        }
        
        logger.debug("Endpoint detector initialized")
    
    def _load_patterns(self) -> List[PatternRule]:
        """Load web endpoint detection patterns."""
        patterns = []
        
        # Get confidence scores from config
        base_confidence = self.config.get('deterministic', {}).get('pattern_confidence', {}).get('web_endpoint', 0.99)
        
        # Flask patterns
        patterns.append(PatternRule(
            name="flask_routes",
            actor_type=ActorType.WEB_ENDPOINT,
            confidence=base_confidence,
            imports=["flask", "flask.*"],
            decorators=[
                "@app.route", "@router.route", "@bp.route", "@blueprint.route",
                "@app.before_request", "@app.after_request", "@app.errorhandler"
            ],
            function_calls=["Flask", "Blueprint", "render_template", "jsonify", "redirect"],
            keywords=["flask", "route", "endpoint", "blueprint"]
        ))
        
        # FastAPI patterns
        patterns.append(PatternRule(
            name="fastapi_routes",
            actor_type=ActorType.WEB_ENDPOINT,
            confidence=base_confidence,
            imports=["fastapi", "fastapi.*"],
            decorators=[
                "@app.get", "@app.post", "@app.put", "@app.delete", "@app.patch",
                "@app.head", "@app.options", "@app.trace",
                "@router.get", "@router.post", "@router.put", "@router.delete",
                "@app.middleware", "@app.exception_handler"
            ],
            function_calls=["FastAPI", "APIRouter", "HTTPException", "Depends"],
            keywords=["fastapi", "router", "endpoint", "api"]
        ))
        
        # Django patterns
        patterns.append(PatternRule(
            name="django_views",
            actor_type=ActorType.WEB_ENDPOINT,
            confidence=base_confidence - 0.05,
            imports=["django.*", "django.http", "django.views", "rest_framework.*"],
            class_names=["View", "TemplateView", "ListView", "DetailView", "APIView", "ViewSet"],
            function_calls=[
                "HttpResponse", "JsonResponse", "HttpResponseRedirect",
                "render", "get_object_or_404", "reverse"
            ],
            keywords=["django", "view", "template", "api"]
        ))
        
        # Django REST Framework patterns
        patterns.append(PatternRule(
            name="django_rest_framework",
            actor_type=ActorType.WEB_ENDPOINT,
            confidence=base_confidence,
            imports=["rest_framework.*", "rest_framework.views", "rest_framework.viewsets"],
            class_names=["APIView", "ViewSet", "ModelViewSet", "GenericAPIView"],
            decorators=["@api_view", "@permission_classes", "@authentication_classes"],
            function_calls=["Response", "status", "serializers"],
            keywords=["rest_framework", "api", "serializer"]
        ))
        
        # Tornado patterns
        patterns.append(PatternRule(
            name="tornado_handlers",
            actor_type=ActorType.WEB_ENDPOINT,
            confidence=base_confidence - 0.1,
            imports=["tornado.*", "tornado.web", "tornado.routing"],
            class_names=["RequestHandler", "Application"],
            function_calls=["tornado.web.RequestHandler", "self.write", "self.render"],
            keywords=["tornado", "handler", "request"]
        ))
        
        # Bottle patterns
        patterns.append(PatternRule(
            name="bottle_routes",
            actor_type=ActorType.WEB_ENDPOINT,
            confidence=base_confidence - 0.1,
            imports=["bottle"],
            decorators=["@route", "@get", "@post", "@put", "@delete"],
            function_calls=["bottle.route", "bottle.run", "bottle.Bottle"],
            keywords=["bottle", "route"]
        ))
        
        # Sanic patterns
        patterns.append(PatternRule(
            name="sanic_routes",
            actor_type=ActorType.WEB_ENDPOINT,
            confidence=base_confidence - 0.05,
            imports=["sanic", "sanic.*"],
            decorators=["@app.route", "@app.get", "@app.post", "@bp.route"],
            function_calls=["Sanic", "Blueprint", "response.json", "response.text"],
            keywords=["sanic", "async", "route"]
        ))
        
        # Starlette patterns
        patterns.append(PatternRule(
            name="starlette_routes",
            actor_type=ActorType.WEB_ENDPOINT,
            confidence=base_confidence - 0.1,
            imports=["starlette.*", "starlette.routing", "starlette.applications"],
            function_calls=["Starlette", "Route", "Mount", "JSONResponse"],
            keywords=["starlette", "asgi", "route"]
        ))
        
        # Quart patterns (async Flask)
        patterns.append(PatternRule(
            name="quart_routes",
            actor_type=ActorType.WEB_ENDPOINT,
            confidence=base_confidence - 0.1,
            imports=["quart", "quart.*"],
            decorators=["@app.route", "@app.before_request", "@app.after_request"],
            function_calls=["Quart", "render_template", "jsonify"],
            keywords=["quart", "async", "route"]
        ))
        
        logger.info(f"Loaded {len(patterns)} endpoint detection patterns")
        return patterns
    
    def _detect_function_calls(self, functions: List[FunctionInfo]) -> List[DetectionMatch]:
        """Detect web endpoints from decorators."""
        return self._detect_endpoint_decorators(functions)
    
    def _detect_content_patterns(self, ast_result: ASTParseResult) -> List[DetectionMatch]:
        """Detect class-based endpoints."""
        return self._detect_class_based_endpoints(ast_result)
    
    
    def _detect_endpoint_decorators(self, functions: List[FunctionInfo]) -> List[DetectionMatch]:
        """Detect endpoint decorators on functions."""
        matches = []
        
        for func_info in functions:
            for decorator in func_info.decorators:
                for pattern in self.patterns:
                    if self._match_endpoint_decorator(decorator, pattern):
                        # Extract route and HTTP method
                        route_path = self._extract_route_path(decorator)
                        http_method = self._extract_http_method_from_decorator(decorator)
                        
                        match = DetectionMatch(
                            actor_type=ActorType.WEB_ENDPOINT,
                            confidence=pattern.confidence,
                            pattern_name=pattern.name,
                            evidence={
                                'decorator_name': decorator.name,
                                'decorator_args': decorator.args,
                                'function_name': func_info.name,
                                'route_path': route_path,
                                'http_method': http_method,
                                'is_async': func_info.is_async
                            },
                            context={
                                'detection_method': 'endpoint_decorator_analysis',
                                'framework_type': self._get_framework_type(pattern.name),
                                'endpoint_type': 'api' if 'api' in func_info.name.lower() else 'web'
                            },
                            line_numbers=[decorator.line_number],
                            function_name=func_info.name
                        )
                        matches.append(match)
        
        return matches
    
    def _detect_class_based_endpoints(self, ast_result: ASTParseResult) -> List[DetectionMatch]:
        """Detect class-based view endpoints."""
        matches = []
        
        for class_info in ast_result.classes:
            for pattern in self.patterns:
                if self._match_endpoint_class(class_info, pattern):
                    match = DetectionMatch(
                        actor_type=ActorType.WEB_ENDPOINT,
                        confidence=pattern.confidence - 0.05,
                        pattern_name=pattern.name,
                        evidence={
                            'class_name': class_info.name,
                            'base_classes': class_info.bases,
                            'method_count': len(class_info.methods),
                            'has_http_methods': self._has_http_methods(class_info)
                        },
                        context={
                            'detection_method': 'endpoint_class_analysis',
                            'framework_type': self._get_framework_type(pattern.name),
                            'endpoint_type': 'class_based'
                        },
                        line_numbers=[class_info.line_number]
                    )
                    matches.append(match)
        
        return matches
    
    
    def _match_endpoint_decorator(self, decorator: DecoratorInfo, pattern: PatternRule) -> bool:
        """Check if decorator matches endpoint pattern."""
        if not pattern.decorators:
            return False
        
        for decorator_pattern in pattern.decorators:
            if decorator.name == decorator_pattern or decorator_pattern in decorator.name:
                return True
        
        return False
    
    def _match_endpoint_class(self, class_info, pattern: PatternRule) -> bool:
        """Check if class matches endpoint pattern."""
        if not pattern.class_names:
            return False
        
        # Check if class inherits from known endpoint base classes
        for base in class_info.bases:
            if any(class_pattern in base for class_pattern in pattern.class_names):
                return True
        
        return False
    
    def _extract_route_path(self, decorator: DecoratorInfo) -> Optional[str]:
        """Extract route path from decorator arguments."""
        if not decorator.args:
            return None
        
        # Look for route path in decorator args
        for arg in decorator.args:
            match = self.route_pattern.search(arg)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_http_method_from_decorator(self, decorator: DecoratorInfo) -> Optional[str]:
        """Extract HTTP method from decorator name."""
        decorator_name = decorator.name.lower()
        
        # Direct method mapping
        if '.get' in decorator_name or decorator_name.endswith('get'):
            return 'GET'
        elif '.post' in decorator_name or decorator_name.endswith('post'):
            return 'POST'
        elif '.put' in decorator_name or decorator_name.endswith('put'):
            return 'PUT'
        elif '.delete' in decorator_name or decorator_name.endswith('delete'):
            return 'DELETE'
        elif '.patch' in decorator_name or decorator_name.endswith('patch'):
            return 'PATCH'
        elif '.head' in decorator_name or decorator_name.endswith('head'):
            return 'HEAD'
        elif '.options' in decorator_name or decorator_name.endswith('options'):
            return 'OPTIONS'
        
        # Check decorator arguments for methods
        if decorator.args:
            for arg in decorator.args:
                arg_upper = arg.upper()
                for method in self.http_methods:
                    if method in arg_upper:
                        return method
        
        # Default for route decorators
        if 'route' in decorator_name:
            return 'GET'  # Default HTTP method
        
        return None
    
    def _has_http_methods(self, class_info) -> bool:
        """Check if class has HTTP method handlers."""
        http_method_names = {method.lower() for method in self.http_methods}
        class_method_names = {method.name.lower() for method in class_info.methods}
        
        return bool(http_method_names.intersection(class_method_names))
    
    def _get_framework_type(self, pattern_name: str) -> str:
        """Get framework type from pattern name."""
        framework_map = {
            'flask_routes': 'flask',
            'fastapi_routes': 'fastapi',
            'django_views': 'django',
            'django_rest_framework': 'django_rest',
            'tornado_handlers': 'tornado',
            'bottle_routes': 'bottle',
            'sanic_routes': 'sanic',
            'starlette_routes': 'starlette',
            'quart_routes': 'quart'
        }
        return framework_map.get(pattern_name, 'unknown')
    
    def _enhance_matches(self, matches: List[DetectionMatch], ast_result: ASTParseResult) -> None:
        """Enhance matches with additional context and information."""
        for match in matches:
            # Add module context
            match.module_name = ast_result.module_name
            
            # Enhance evidence based on detection method
            if match.context.get('detection_method') == 'endpoint_decorator_analysis':
                self._enhance_decorator_match(match, ast_result)
            elif match.context.get('detection_method') == 'endpoint_class_analysis':
                self._enhance_class_match(match, ast_result)
            elif match.context.get('detection_method') == 'endpoint_import_analysis':
                self._enhance_import_match(match, ast_result)
    
    def _enhance_decorator_match(self, match: DetectionMatch, ast_result: ASTParseResult) -> None:
        """Enhance decorator-based matches with function context."""
        function_name = match.function_name
        
        if function_name:
            func_info = next((f for f in ast_result.functions if f.name == function_name), None)
            if func_info:
                match.evidence['function_complexity'] = func_info.complexity
                match.evidence['function_args'] = func_info.args
                match.evidence['decorator_count'] = len(func_info.decorators)
                
                # Analyze function name for endpoint type
                name_lower = function_name.lower()
                if any(api_word in name_lower for api_word in ['api', 'json', 'ajax']):
                    match.context['endpoint_type'] = 'api'
                elif any(view_word in name_lower for view_word in ['view', 'page', 'render']):
                    match.context['endpoint_type'] = 'view'
                
                # Check for authentication/authorization decorators
                auth_decorators = [d for d in func_info.decorators 
                                 if any(auth in d.name.lower() for auth in ['auth', 'login', 'permission'])]
                match.context['has_auth'] = len(auth_decorators) > 0
    
    def _enhance_class_match(self, match: DetectionMatch, ast_result: ASTParseResult) -> None:
        """Enhance class-based matches with class context."""
        class_name = match.evidence.get('class_name', '')
        
        # Find class info
        class_info = next((c for c in ast_result.classes if c.name == class_name), None)
        if class_info:
            # Count HTTP method implementations
            http_methods = []
            for method in class_info.methods:
                if method.name.upper() in self.http_methods:
                    http_methods.append(method.name.upper())
            
            match.evidence['implemented_methods'] = http_methods
            match.context['method_count'] = len(http_methods)
            
            # Check for REST patterns
            if any('rest' in base.lower() or 'api' in base.lower() for base in class_info.bases):
                match.context['endpoint_type'] = 'rest_api'
    
    def _enhance_import_match(self, match: DetectionMatch, ast_result: ASTParseResult) -> None:
        """Enhance import-based matches with usage context."""
        imported_module = match.evidence.get('import_module', '')
        
        # Count decorator usage
        decorator_usage = 0
        for func_info in ast_result.functions:
            for decorator in func_info.decorators:
                if imported_module in decorator.name:
                    decorator_usage += 1
        
        match.evidence['decorator_usage'] = decorator_usage
        match.context['import_usage'] = 'active' if decorator_usage > 0 else 'imported_only'
    
    def get_endpoint_statistics(self, matches: List[DetectionMatch]) -> Dict[str, Any]:
        """Get endpoint-specific statistics from matches."""
        stats = {
            'total_endpoint_matches': len(matches),
            'framework_distribution': {},
            'method_distribution': {},
            'endpoint_type_distribution': {},
            'async_endpoints': 0,
            'auth_protected_endpoints': 0,
            'api_endpoints': 0,
            'routes_detected': 0
        }
        
        for match in matches:
            # Framework distribution
            framework = match.context.get('framework_type', 'unknown')
            stats['framework_distribution'][framework] = stats['framework_distribution'].get(framework, 0) + 1
            
            # Method distribution
            method = match.evidence.get('http_method')
            if method:
                stats['method_distribution'][method] = stats['method_distribution'].get(method, 0) + 1
            
            # Endpoint type distribution
            endpoint_type = match.context.get('endpoint_type', 'unknown')
            stats['endpoint_type_distribution'][endpoint_type] = stats['endpoint_type_distribution'].get(endpoint_type, 0) + 1
            
            # Async endpoints
            if match.evidence.get('is_async', False):
                stats['async_endpoints'] += 1
            
            # Auth protected endpoints
            if match.context.get('has_auth', False):
                stats['auth_protected_endpoints'] += 1
            
            # API endpoints
            if endpoint_type in ['api', 'rest_api']:
                stats['api_endpoints'] += 1
            
            # Routes detected
            if match.evidence.get('route_path'):
                stats['routes_detected'] += 1
        
        return stats