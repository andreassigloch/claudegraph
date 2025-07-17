#!/usr/bin/env python3
"""
Enhanced Dead Code Detection for Code Architecture Analyzer

Detects unused functions with detailed context and code snippets
for customer decision-making.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import re
from .dynamic_pattern_detector import DynamicPatternDetector

logger = logging.getLogger(__name__)


@dataclass
class DeadCodeFunction:
    """Represents a dead/unused function with detailed context"""
    name: str
    module: str
    full_name: str
    location: str  # file:line_start-line_end
    code_snippet: str
    issue_type: str  # "DUPLICATE", "ORPHANED", "UNREACHABLE"
    similar_functions: List[str] = field(default_factory=list)
    suggestion: str = ""
    reason: str = ""
    file_path: str = ""
    line_start: int = 0
    line_end: int = 0
    function_size: int = 0  # lines of code
    has_docstring: bool = False
    complexity: int = 0


@dataclass
class DeadCodeAnalysis:
    """Complete dead code analysis results"""
    total_functions: int
    dead_functions: List[DeadCodeFunction]
    duplicates: List[DeadCodeFunction]
    orphaned: List[DeadCodeFunction]
    unreachable: List[DeadCodeFunction]
    summary: Dict[str, Any]


class DeadCodeDetector:
    """Enhanced dead code detection with detailed context"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_dead_code(self, ast_results: List[Any], flow_relationships: List[Any],
                         project_structure: Any, enable_dynamic_detection: bool = True) -> DeadCodeAnalysis:
        """
        Perform comprehensive dead code analysis with detailed context
        
        Args:
            ast_results: AST parsing results
            flow_relationships: Flow relationships between functions
            project_structure: Project structure information
            enable_dynamic_detection: Whether to use dynamic pattern detection
            
        Returns:
            DeadCodeAnalysis with detailed function information
        """
        try:
            # Build function inventory
            all_functions = self._build_function_inventory(ast_results)
            
            # Find functions involved in flows
            connected_functions = self._find_connected_functions(flow_relationships)
            
            # Identify dead functions
            dead_functions = self._identify_dead_functions(all_functions, connected_functions)
            
            # Apply dynamic pattern detection to reduce false positives
            if enable_dynamic_detection and dead_functions:
                dynamic_detector = DynamicPatternDetector()
                dynamic_calls = dynamic_detector.detect_dynamic_calls(ast_results)
                
                # Filter out potentially live functions
                filtered_dead = []
                for func_info in dead_functions:
                    func_name = func_info.get('name', '').split('.')[-1]
                    if func_name not in dynamic_calls:
                        filtered_dead.append(func_info)
                    else:
                        self.logger.info(f"Function '{func_name}' may be called dynamically - removing from dead code list")
                
                dead_functions = filtered_dead
                self.logger.info(f"Dynamic pattern detection reduced dead functions from {len(dead_functions) + len(dynamic_calls)} to {len(dead_functions)}")
            
            # Add detailed context to each dead function
            enhanced_dead_functions = []
            for func_info in dead_functions:
                enhanced_func = self._enhance_function_context(func_info, all_functions, project_structure)
                enhanced_dead_functions.append(enhanced_func)
            
            # Categorize dead functions
            duplicates, orphaned, unreachable = self._categorize_dead_functions(enhanced_dead_functions)
            
            # Generate summary
            summary = self._generate_summary(enhanced_dead_functions, duplicates, orphaned, unreachable)
            
            analysis = DeadCodeAnalysis(
                total_functions=len(all_functions),
                dead_functions=enhanced_dead_functions,
                duplicates=duplicates,
                orphaned=orphaned,
                unreachable=unreachable,
                summary=summary
            )
            
            self.logger.info(f"Dead code analysis: {len(enhanced_dead_functions)} dead functions found "
                           f"({len(duplicates)} duplicates, {len(orphaned)} orphaned)")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in dead code analysis: {e}")
            return DeadCodeAnalysis(
                total_functions=0,
                dead_functions=[],
                duplicates=[],
                orphaned=[],
                unreachable=[],
                summary={"error": str(e)}
            )
    
    def _build_function_inventory(self, ast_results: List[Any]) -> Dict[str, Any]:
        """Build complete inventory of all functions"""
        functions = {}
        
        for ast_result in ast_results:
            module_name = ast_result.module_name or Path(ast_result.file_path).stem
            
            # Process standalone functions
            for func in ast_result.functions:
                full_name = f"{module_name}.{func.name}"
                functions[full_name] = {
                    'info': func,
                    'module': module_name,
                    'ast_result': ast_result,
                    'is_method': False
                }
            
            # Process class methods
            for class_info in ast_result.classes:
                for method in class_info.methods:
                    full_name = f"{module_name}.{class_info.name}.{method.name}"
                    functions[full_name] = {
                        'info': method,
                        'module': module_name,
                        'ast_result': ast_result,
                        'is_method': True,
                        'class_name': class_info.name
                    }
        
        return functions
    
    def _find_connected_functions(self, flow_relationships: List[Any]) -> Set[str]:
        """Find all functions that have flow relationships"""
        connected = set()
        
        for rel in flow_relationships:
            if hasattr(rel, 'type') and rel.type == 'flow':
                # Add source function
                if hasattr(rel, 'source_name'):
                    connected.add(rel.source_name)
                
                # Add target function  
                if hasattr(rel, 'target_name'):
                    connected.add(rel.target_name)
            
            # Also check properties for function names
            elif hasattr(rel, 'properties'):
                props = rel.properties if hasattr(rel.properties, 'get') else {}
                if 'source_name' in props:
                    connected.add(props['source_name'])
                if 'target_name' in props:
                    connected.add(props['target_name'])
        
        return connected
    
    def _identify_dead_functions(self, all_functions: Dict[str, Any], 
                               connected_functions: Set[str]) -> List[Dict[str, Any]]:
        """Identify functions that are not connected in flows"""
        dead_functions = []
        
        # Define patterns for functions that should be considered live even if not connected
        entry_point_patterns = [
            'main',           # Main functions
            'cli',            # CLI functions
            'run_',           # Run functions
            'start_',         # Start functions
            'launch_',        # Launch functions
            'execute_',       # Execute functions
            'serve_',         # Server functions
            '__main__',       # Module entry point
            'test_',          # Test functions
            '_test',          # Test functions
            'Test',           # Test classes methods
            'setup',          # Setup functions
            'teardown',       # Teardown functions
        ]
        
        decorator_indicators = [
            'app.route',      # Flask routes
            'router.',        # Router endpoints
            '@click.',        # Click commands
            '@command',       # Command decorators
            '@endpoint',      # Endpoint decorators
            '@api.',          # API decorators
        ]
        
        for full_name, func_data in all_functions.items():
            func_info = func_data['info']
            module_name = func_data['module']
            
            # Skip constructor methods
            if func_info.name == '__init__':
                continue
            
            # Check various name patterns for connections
            name_variants = [
                full_name,  # module.function
                func_info.name,  # function
                func_info.full_name if hasattr(func_info, 'full_name') else func_info.name
            ]
            
            # Check if any variant is connected
            is_connected = any(variant in connected_functions for variant in name_variants)
            
            if not is_connected:
                # Check if it's likely an entry point function
                is_entry_point = False
                
                # Check function name patterns
                for pattern in entry_point_patterns:
                    if pattern in func_info.name.lower():
                        is_entry_point = True
                        break
                
                # Check if it's in a main module
                if module_name in ['main', '__main__', 'cli', 'run_ui_server']:
                    is_entry_point = True
                
                # Check decorators for entry point indicators
                if hasattr(func_info, 'decorators'):
                    decorator_names = [d.name for d in func_info.decorators]
                    for decorator in decorator_names:
                        if any(pattern in decorator for pattern in decorator_indicators):
                            is_entry_point = True
                            break
                
                # Only add to dead functions if it's not an entry point
                if not is_entry_point:
                    dead_functions.append(func_data)
                else:
                    self.logger.info(f"Keeping potential entry point: {full_name}")
        
        return dead_functions
    
    def _enhance_function_context(self, func_data: Dict[str, Any], 
                                all_functions: Dict[str, Any],
                                project_structure: Any) -> DeadCodeFunction:
        """Add detailed context to a dead function"""
        func_info = func_data['info']
        module_name = func_data['module']
        ast_result = func_data['ast_result']
        
        # Build full name
        if func_data.get('is_method'):
            full_name = f"{module_name}.{func_data['class_name']}.{func_info.name}"
        else:
            full_name = f"{module_name}.{func_info.name}"
        
        # Extract code snippet
        code_snippet = self._extract_code_snippet(func_info, ast_result)
        
        # Determine issue type and similar functions
        issue_type, similar_functions, suggestion, reason = self._analyze_function_issue(
            func_info, module_name, all_functions
        )
        
        # Calculate function size
        function_size = getattr(func_info, 'end_line_number', 0) - getattr(func_info, 'line_number', 0) + 1
        
        # Create clean relative path
        if project_structure and hasattr(project_structure, 'root_path'):
            try:
                relative_path = Path(ast_result.file_path).relative_to(project_structure.root_path)
                clean_location = f"{relative_path}:{func_info.line_number}-{getattr(func_info, 'end_line_number', func_info.line_number)}"
                clean_file_path = str(relative_path)
            except ValueError:
                # Fallback if relative path fails
                clean_location = f"{module_name}.py:{func_info.line_number}-{getattr(func_info, 'end_line_number', func_info.line_number)}"
                clean_file_path = f"{module_name}.py"
        else:
            # Fallback when no project structure
            clean_location = f"{module_name}.py:{func_info.line_number}-{getattr(func_info, 'end_line_number', func_info.line_number)}"
            clean_file_path = f"{module_name}.py"
        
        return DeadCodeFunction(
            name=func_info.name,
            module=module_name,
            full_name=full_name,
            location=clean_location,
            code_snippet=code_snippet,
            issue_type=issue_type,
            similar_functions=similar_functions,
            suggestion=suggestion,
            reason=reason,
            file_path=clean_file_path,
            line_start=func_info.line_number,
            line_end=getattr(func_info, 'end_line_number', func_info.line_number),
            function_size=function_size,
            has_docstring=bool(getattr(func_info, 'docstring', None)),
            complexity=getattr(func_info, 'complexity', 0)
        )
    
    def _extract_code_snippet(self, func_info: Any, ast_result: Any) -> str:
        """Extract code snippet for the function"""
        try:
            # Try to read the actual file content
            file_path = Path(ast_result.file_path)
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                start_line = func_info.line_number - 1  # Convert to 0-based
                end_line = getattr(func_info, 'end_line_number', func_info.line_number)
                
                if start_line < len(lines):
                    # Get function signature and first few lines
                    snippet_lines = lines[start_line:min(start_line + 10, end_line)]
                    snippet = ''.join(snippet_lines).strip()
                    
                    # If function is longer than 10 lines, add truncation indicator
                    if end_line - start_line > 10:
                        snippet += f"\n    # ... ({end_line - start_line - 10} more lines) ..."
                    
                    return snippet
            
            # Fallback: create basic snippet from function info
            args_str = ', '.join(getattr(func_info, 'args', []))
            docstring = getattr(func_info, 'docstring', '')
            docstring_line = f'    """{docstring}"""' if docstring else ''
            
            snippet = f"def {func_info.name}({args_str}):"
            if docstring_line:
                snippet += f"\n{docstring_line}"
            snippet += "\n    # Implementation..."
            
            return snippet
            
        except Exception as e:
            self.logger.warning(f"Could not extract code snippet for {func_info.name}: {e}")
            return f"def {func_info.name}(...):\n    # Code snippet unavailable"
    
    def _analyze_function_issue(self, func_info: Any, module_name: str, 
                              all_functions: Dict[str, Any]) -> Tuple[str, List[str], str, str]:
        """Analyze why a function is dead and find similar functions"""
        similar_functions = []
        func_name = func_info.name
        
        # Look for functions with the same name in different modules
        for full_name, other_func_data in all_functions.items():
            other_func = other_func_data['info']
            other_module = other_func_data['module']
            
            if other_func.name == func_name and other_module != module_name:
                similar_functions.append(f"{other_module}.{func_name}")
        
        # Determine issue type and suggestions
        if similar_functions:
            issue_type = "DUPLICATE"
            suggestion = f"Remove or integrate with active version: {similar_functions[0]}"
            reason = f"Duplicate function exists in {len(similar_functions)} other module(s)"
        else:
            # Check if it looks like a utility function
            if func_name.startswith('_') or func_name in ['helper', 'util', 'debug']:
                issue_type = "UNREACHABLE"
                suggestion = "Internal/utility function - verify if needed"
                reason = "Private or utility function with no callers"
            else:
                issue_type = "ORPHANED"
                suggestion = "Legacy code or incomplete feature?"
                reason = "No callers or triggers found"
        
        return issue_type, similar_functions, suggestion, reason
    
    def _categorize_dead_functions(self, dead_functions: List[DeadCodeFunction]) -> Tuple[List[DeadCodeFunction], List[DeadCodeFunction], List[DeadCodeFunction]]:
        """Categorize dead functions by issue type"""
        duplicates = [f for f in dead_functions if f.issue_type == "DUPLICATE"]
        orphaned = [f for f in dead_functions if f.issue_type == "ORPHANED"] 
        unreachable = [f for f in dead_functions if f.issue_type == "UNREACHABLE"]
        
        return duplicates, orphaned, unreachable
    
    def _generate_summary(self, dead_functions: List[DeadCodeFunction],
                         duplicates: List[DeadCodeFunction],
                         orphaned: List[DeadCodeFunction], 
                         unreachable: List[DeadCodeFunction]) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not dead_functions:
            return {
                "total_dead": 0,
                "by_type": {"duplicates": 0, "orphaned": 0, "unreachable": 0},
                "by_module": {},
                "total_loc": 0,
                "largest_function": None
            }
        
        # Group by module
        by_module = {}
        total_loc = 0
        largest_function = None
        max_size = 0
        
        for func in dead_functions:
            if func.module not in by_module:
                by_module[func.module] = []
            by_module[func.module].append(func.name)
            
            total_loc += func.function_size
            
            if func.function_size > max_size:
                max_size = func.function_size
                largest_function = func.full_name
        
        return {
            "total_dead": len(dead_functions),
            "by_type": {
                "duplicates": len(duplicates),
                "orphaned": len(orphaned), 
                "unreachable": len(unreachable)
            },
            "by_module": by_module,
            "total_loc": total_loc,
            "largest_function": largest_function,
            "largest_function_size": max_size
        }