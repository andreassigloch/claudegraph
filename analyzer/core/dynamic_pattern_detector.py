"""
Dynamic Pattern Detector - Detects functions called through dynamic patterns.

This module identifies functions that may be called dynamically through:
- importlib.import_module()
- getattr() calls
- Plugin/extension systems
- Decorator registration
- String-based dispatch
"""

import ast
import re
import logging
from typing import Set, List, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class DynamicPatternDetector:
    """Detect functions that may be called through dynamic patterns."""
    
    # Common dynamic call patterns
    DYNAMIC_PATTERNS = [
        # importlib patterns
        (r'importlib\.import_module\(["\']([^"\']+)["\']', 'importlib'),
        (r'__import__\(["\']([^"\']+)["\']', '__import__'),
        
        # getattr patterns  
        (r'getattr\([^,]+,\s*["\']([^"\']+)["\']', 'getattr'),
        (r'hasattr\([^,]+,\s*["\']([^"\']+)["\']', 'hasattr'),
        
        # Plugin/extension patterns
        (r'load_plugin\(["\']([^"\']+)["\']', 'plugin'),
        (r'register_plugin\(["\']([^"\']+)["\']', 'plugin'),
        (r'load_extension\(["\']([^"\']+)["\']', 'extension'),
        
        # Decorator registration patterns
        (r'@register\(["\']([^"\']+)["\']', 'decorator'),
        (r'@route\(["\']([^"\']+)["\']', 'route_decorator'),
        (r'@endpoint\(["\']([^"\']+)["\']', 'endpoint_decorator'),
        
        # String-based dispatch
        (r'dispatch\(["\']([^"\']+)["\']', 'dispatch'),
        (r'call_method\(["\']([^"\']+)["\']', 'dispatch'),
        
        # Command/action patterns
        (r'run_command\(["\']([^"\']+)["\']', 'command'),
        (r'execute_action\(["\']([^"\']+)["\']', 'action'),
        
        # Entry point patterns
        (r'entry_points\s*=.*["\']([^"\']+)["\']', 'entry_point'),
        (r'console_scripts\s*=.*["\']([^"\']+)["\']', 'console_script'),
    ]
    
    # Common dynamic module/class access patterns
    MODULE_ACCESS_PATTERNS = [
        r'globals\(\)\[["\']([^"\']+)["\']\]',  # globals()['function_name']
        r'locals\(\)\[["\']([^"\']+)["\']\]',   # locals()['function_name']
        r'vars\(\)\[["\']([^"\']+)["\']\]',     # vars()['function_name']
        r'__dict__\[["\']([^"\']+)["\']\]',     # obj.__dict__['method_name']
    ]
    
    # Test/fixture patterns that shouldn't count as dead code
    TEST_PATTERNS = [
        r'pytest\.fixture',
        r'unittest\.TestCase',
        r'test_[a-zA-Z0-9_]+',
        r'Test[A-Z][a-zA-Z0-9_]*',
    ]
    
    def __init__(self):
        self.dynamic_calls = set()
        self.potential_calls = {}  # function -> (pattern_type, confidence)
        self.module_registries = {}  # module -> set of registered names
        
    def detect_dynamic_calls(self, ast_results: List) -> Set[str]:
        """
        Find functions that may be called dynamically.
        
        Args:
            ast_results: List of AST parsing results (ASTParseResult objects)
            
        Returns:
            Set of function names that may be called dynamically
        """
        self.dynamic_calls.clear()
        self.potential_calls.clear()
        
        for ast_result in ast_results:
            self._analyze_ast_result(ast_result)
        
        # Combine all dynamic calls
        all_dynamic = set(self.dynamic_calls)
        all_dynamic.update(self.potential_calls.keys())
        
        logger.info(f"Detected {len(all_dynamic)} potentially dynamic function calls")
        
        return all_dynamic
    
    def _analyze_ast_result(self, ast_result):
        """Analyze an ASTParseResult for dynamic patterns."""
        module_name = ast_result.module_name or Path(ast_result.file_path).stem
        
        # Read the actual file content for pattern analysis
        try:
            with open(ast_result.file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
        except Exception as e:
            logger.warning(f"Could not read file {ast_result.file_path}: {e}")
            file_content = ""
        
        # Analyze each function for dynamic patterns
        for func in ast_result.functions:
            func_name = func.name
            
            # Check if function is accessed dynamically in the file
            if self._is_dynamically_accessed(func_name, file_content):
                self.dynamic_calls.add(func_name)
            
            # Check for dynamic calls within the function (if we have body content)
            # Note: We use file content since individual function bodies aren't available
            dynamic_refs = self._find_dynamic_references(file_content)
            self.dynamic_calls.update(dynamic_refs)
            
            # Check for decorator-based registration
            decorator_names = [d.name for d in func.decorators]
            if self._has_registration_decorator(decorator_names):
                self.potential_calls[func_name] = ('decorator_registration', 0.8)
        
        # Check for class methods
        for class_info in ast_result.classes:
            for method in class_info.methods:
                method_name = method.name
                
                # Skip constructors and private methods
                if method_name.startswith('__'):
                    continue
                
                if self._is_dynamically_accessed(method_name, file_content):
                    self.dynamic_calls.add(method_name)
                
                # Check decorators on methods
                decorator_names = [d.name for d in method.decorators]
                if self._has_registration_decorator(decorator_names):
                    self.potential_calls[method_name] = ('decorator_registration', 0.8)
        
        # Check module-level registrations
        self._check_file_registrations(file_content, module_name)
    
    
    def _check_file_registrations(self, file_content: str, module_name: str):
        """Check for module-level function registrations in file content."""
        # Look for registry patterns like COMMANDS = {'name': function}
        registry_patterns = [
            r'COMMANDS\s*=\s*{[^}]+}',
            r'HANDLERS\s*=\s*{[^}]+}',
            r'ROUTES\s*=\s*{[^}]+}',
            r'PLUGINS\s*=\s*{[^}]+}',
            r'REGISTRY\s*=\s*{[^}]+}',
        ]
        
        for pattern in registry_patterns:
            try:
                match = re.search(pattern, file_content)
                if match:
                    # Extract registered function names
                    registry_text = match.group(0)
                    func_names = re.findall(r'[\'"]([^\'"]+)[\'"]:\s*(\w+)', registry_text)
                    
                    for cmd_name, func_name in func_names:
                        self.potential_calls[func_name] = ('registry', 0.9)
                        if module_name not in self.module_registries:
                            self.module_registries[module_name] = set()
                        self.module_registries[module_name].add(func_name)
            except re.error as e:
                logger.warning(f"Regex error in registry pattern '{pattern}': {e}")
                continue
    
    def _is_dynamically_accessed(self, func_name: str, module_body: str) -> bool:
        """Check if a function is accessed through dynamic patterns."""
        # Check for module access patterns
        for pattern in self.MODULE_ACCESS_PATTERNS:
            try:
                matches = re.findall(pattern, module_body)
            except re.error as e:
                logger.warning(f"Regex error in MODULE_ACCESS_PATTERNS '{pattern}': {e}")
                continue
            if func_name in matches:
                return True
        
        # Check for string references to function
        if f'"{func_name}"' in module_body or f"'{func_name}'" in module_body:
            # More sophisticated check - is it used in a dynamic context?
            # Escape the function name for regex
            escaped_func_name = re.escape(func_name)
            dynamic_contexts = [
                f'getattr\\(.*{escaped_func_name}',
                f'__import__.*{escaped_func_name}',
                f'importlib.*{escaped_func_name}',
                f'eval.*{escaped_func_name}',
                f'exec.*{escaped_func_name}',
            ]
            
            for context in dynamic_contexts:
                try:
                    if re.search(context, module_body):
                        return True
                except re.error as e:
                    logger.warning(f"Regex error in dynamic context '{context}': {e}")
                    continue
        
        return False
    
    def _find_dynamic_references(self, code_body: str) -> Set[str]:
        """Find function names referenced in dynamic patterns."""
        references = set()
        
        # Check each dynamic pattern
        for pattern, pattern_type in self.DYNAMIC_PATTERNS:
            try:
                matches = re.findall(pattern, code_body, re.MULTILINE)
            except re.error as e:
                logger.warning(f"Regex error in pattern '{pattern}': {e}")
                continue
            for match in matches:
                # Extract potential function name
                if '.' in match:
                    # Could be module.function
                    parts = match.split('.')
                    references.add(parts[-1])  # Add function name
                    references.add(match)      # Add full reference
                else:
                    references.add(match)
                
                # Track pattern type for confidence scoring
                if match not in self.potential_calls:
                    self.potential_calls[match] = (pattern_type, 0.7)
        
        return references
    
    def _has_registration_decorator(self, decorators: List[str]) -> bool:
        """Check if function has a registration-style decorator."""
        registration_patterns = [
            'register', 'route', 'endpoint', 'handler',
            'command', 'action', 'task', 'plugin',
            'extension', 'hook', 'listener', 'subscriber'
        ]
        
        for decorator in decorators:
            decorator_lower = decorator.lower()
            if any(pattern in decorator_lower for pattern in registration_patterns):
                return True
        
        return False
    
    def _check_module_registrations(self, module_result: Dict):
        """Check for module-level function registrations."""
        module_name = module_result.get('module_name', '')
        
        # Look for registry patterns like COMMANDS = {'name': function}
        registry_patterns = [
            r'COMMANDS\s*=\s*{[^}]+}',
            r'HANDLERS\s*=\s*{[^}]+}',
            r'ROUTES\s*=\s*{[^}]+}',
            r'PLUGINS\s*=\s*{[^}]+}',
            r'REGISTRY\s*=\s*{[^}]+}',
        ]
        
        module_body = str(module_result)  # Simplified - would need actual source
        
        for pattern in registry_patterns:
            match = re.search(pattern, module_body)
            if match:
                # Extract registered function names
                registry_text = match.group(0)
                func_names = re.findall(r'[\'"]([^"\']+)[\'"]:\s*(\w+)', registry_text)
                
                for cmd_name, func_name in func_names:
                    self.potential_calls[func_name] = ('registry', 0.9)
                    if module_name not in self.module_registries:
                        self.module_registries[module_name] = set()
                    self.module_registries[module_name].add(func_name)
    
    def mark_potentially_live(self, dead_functions: List[Dict], 
                            dynamic_calls: Optional[Set[str]] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter out potentially live functions from dead function list.
        
        Args:
            dead_functions: List of functions marked as dead
            dynamic_calls: Optional set of known dynamic calls
            
        Returns:
            Tuple of (filtered_dead_functions, potentially_live_functions)
        """
        if dynamic_calls is None:
            dynamic_calls = self.dynamic_calls
        
        filtered_dead = []
        potentially_live = []
        
        for func in dead_functions:
            func_name = func.get('Name', '').split('.')[-1]  # Get just function name
            
            # Check if function might be called dynamically
            if func_name in dynamic_calls or func_name in self.potential_calls:
                func['dead_code_confidence'] = 0.3  # Low confidence it's dead
                func['dynamic_pattern'] = self.potential_calls.get(func_name, ('unknown', 0.5))
                potentially_live.append(func)
            elif self._is_test_fixture(func_name):
                func['dead_code_confidence'] = 0.2  # Very low confidence
                func['dynamic_pattern'] = ('test_fixture', 0.8)
                potentially_live.append(func)
            else:
                func['dead_code_confidence'] = 0.9  # High confidence it's dead
                filtered_dead.append(func)
        
        logger.info(f"Filtered {len(potentially_live)} potentially live functions from dead code list")
        
        return filtered_dead, potentially_live
    
    def _is_test_fixture(self, func_name: str) -> bool:
        """Check if function is likely a test fixture."""
        for pattern in self.TEST_PATTERNS:
            if re.match(pattern, func_name):
                return True
        return False
    
    def get_dynamic_call_report(self) -> Dict:
        """Generate a report of detected dynamic patterns."""
        return {
            'total_dynamic_calls': len(self.dynamic_calls),
            'potential_calls': len(self.potential_calls),
            'pattern_breakdown': self._get_pattern_breakdown(),
            'confidence_distribution': self._get_confidence_distribution(),
            'module_registries': {
                module: list(funcs) 
                for module, funcs in self.module_registries.items()
            }
        }
    
    def _get_pattern_breakdown(self) -> Dict[str, int]:
        """Get breakdown of dynamic patterns by type."""
        breakdown = {}
        for func, (pattern_type, _) in self.potential_calls.items():
            breakdown[pattern_type] = breakdown.get(pattern_type, 0) + 1
        return breakdown
    
    def _get_confidence_distribution(self) -> Dict[str, int]:
        """Get distribution of confidence scores."""
        distribution = {
            'high': 0,      # >= 0.8
            'medium': 0,    # >= 0.6
            'low': 0        # < 0.6
        }
        
        for _, (_, confidence) in self.potential_calls.items():
            if confidence >= 0.8:
                distribution['high'] += 1
            elif confidence >= 0.6:
                distribution['medium'] += 1
            else:
                distribution['low'] += 1
        
        return distribution


def enhance_dead_code_detection(dead_functions: List[Dict], 
                              ast_results: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    Enhanced dead code detection that considers dynamic patterns.
    
    Args:
        dead_functions: Initial list of dead functions
        ast_results: AST parsing results
        
    Returns:
        Tuple of (filtered_dead_functions, analysis_report)
    """
    detector = DynamicPatternDetector()
    
    # Detect dynamic calls
    dynamic_calls = detector.detect_dynamic_calls(ast_results)
    
    # Filter dead functions
    filtered_dead, potentially_live = detector.mark_potentially_live(
        dead_functions, 
        dynamic_calls
    )
    
    # Generate report
    report = detector.get_dynamic_call_report()
    report['filtered_count'] = len(filtered_dead)
    report['potentially_live_count'] = len(potentially_live)
    report['accuracy_improvement'] = (
        (len(dead_functions) - len(filtered_dead)) / len(dead_functions) * 100
        if dead_functions else 0
    )
    
    return filtered_dead, report