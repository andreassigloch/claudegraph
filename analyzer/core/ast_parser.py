#!/usr/bin/env python3
"""
AST Parser for Code Architecture Analyzer

Extracts functions, classes, imports, and other structural elements
from Python source code using the built-in ast module.
"""

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import re


logger = logging.getLogger(__name__)


@dataclass
class ImportInfo:
    """Information about an import statement."""
    module: str
    name: Optional[str] = None  # For 'from module import name'
    alias: Optional[str] = None  # For 'import module as alias'
    line_number: int = 0
    is_from_import: bool = False
    level: int = 0  # For relative imports (number of dots)


@dataclass
class DecoratorInfo:
    """Information about a function/class decorator."""
    name: str
    args: List[str] = field(default_factory=list)
    kwargs: Dict[str, str] = field(default_factory=dict)
    line_number: int = 0


@dataclass
class FunctionInfo:
    """Information about a function or method."""
    name: str
    full_name: str  # Including class name if method
    line_number: int
    end_line_number: int = 0
    args: List[str] = field(default_factory=list)
    returns: Optional[str] = None
    decorators: List[DecoratorInfo] = field(default_factory=list)
    docstring: Optional[str] = None
    is_method: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False
    is_property: bool = False
    is_async: bool = False
    parent_class: Optional[str] = None
    calls: List[str] = field(default_factory=list)  # Function calls within this function
    complexity: int = 1  # Cyclomatic complexity estimate


@dataclass
class ClassInfo:
    """Information about a class definition."""
    name: str
    line_number: int
    end_line_number: int = 0
    bases: List[str] = field(default_factory=list)
    decorators: List[DecoratorInfo] = field(default_factory=list)
    docstring: Optional[str] = None
    methods: List[FunctionInfo] = field(default_factory=list)
    is_dataclass: bool = False
    is_exception: bool = False


@dataclass
class ASTParseResult:
    """Complete result of AST parsing for a file."""
    file_path: Path
    module_name: str
    imports: List[ImportInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    constants: Dict[str, Any] = field(default_factory=dict)
    global_calls: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    encoding: str = 'utf-8'
    syntax_version: Tuple[int, int] = (3, 8)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    tree: Optional[ast.AST] = None
    source_code: Optional[str] = None


class ASTParser:
    """
    Python AST parser for extracting structural information from source code.
    
    Extracts functions, classes, imports, and other elements needed for
    architecture analysis and ontology mapping.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize AST parser with configuration."""
        self.config = config or {}
        
        # Parser settings
        deterministic_config = self.config.get('deterministic', {})
        self.include_private_functions = deterministic_config.get('include_private_functions', True)
        self.include_test_functions = deterministic_config.get('include_test_functions', False)
        self.min_function_lines = deterministic_config.get('min_function_lines', 1)
        
        # Debugging settings
        dev_config = self.config.get('development', {})
        self.debug_ast_parsing = dev_config.get('debug_ast_parsing', False)
        
        logger.debug("AST Parser initialized")
    
    def parse_file(self, file_path: Union[str, Path], module_name: str = "") -> ASTParseResult:
        """
        Parse a single Python file and extract structural information.
        
        Args:
            file_path: Path to the Python file
            module_name: Module name for the file
            
        Returns:
            ASTParseResult with extracted information
        """
        file_path = Path(file_path)
        result = ASTParseResult(
            file_path=file_path,
            module_name=module_name or self._extract_module_name(file_path)
        )
        
        try:
            # Read file content
            content = self._read_file(file_path)
            if not content:
                result.errors.append(f"Could not read file: {file_path}")
                return result
            
            # Parse AST
            tree = ast.parse(content, filename=str(file_path))
            
            # Store tree and source code in result
            result.tree = tree
            result.source_code = content
            
            # Extract information using visitor
            visitor = ASTVisitor(result, self.config)
            visitor.visit(tree)
            
            # Post-process results
            self._post_process_result(result)
            
            logger.debug(f"Parsed {file_path}: {len(result.functions)} functions, "
                        f"{len(result.classes)} classes, {len(result.imports)} imports")
            
        except SyntaxError as e:
            error_msg = f"Syntax error in {file_path}: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
        except Exception as e:
            error_msg = f"Error parsing {file_path}: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
        
        return result
    
    def parse_files(self, file_paths: List[Union[str, Path]], 
                   module_names: Optional[List[str]] = None) -> List[ASTParseResult]:
        """
        Parse multiple Python files.
        
        Args:
            file_paths: List of file paths to parse
            module_names: Optional list of module names
            
        Returns:
            List of ASTParseResult objects
        """
        results = []
        module_names = module_names or [None] * len(file_paths)
        
        for i, file_path in enumerate(file_paths):
            module_name = module_names[i] if i < len(module_names) else ""
            result = self.parse_file(file_path, module_name)
            results.append(result)
        
        logger.info(f"Parsed {len(file_paths)} files")
        return results
    
    def _read_file(self, file_path: Path) -> Optional[str]:
        """Read file content with encoding detection."""
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Error reading {file_path} with {encoding}: {e}")
                continue
        
        logger.error(f"Could not read {file_path} with any encoding")
        return None
    
    def _extract_module_name(self, file_path: Path) -> str:
        """Extract module name from file path."""
        # Simple extraction - can be enhanced based on project structure
        name = file_path.stem
        if name == '__init__':
            name = file_path.parent.name
        return name
    
    def _post_process_result(self, result: ASTParseResult) -> None:
        """Post-process parsing results."""
        # Filter functions based on configuration
        if not self.include_private_functions:
            result.functions = [f for f in result.functions if not f.name.startswith('_')]
        
        if not self.include_test_functions:
            result.functions = [f for f in result.functions if not self._is_test_function(f)]
        
        # Filter by minimum lines
        if self.min_function_lines > 1:
            result.functions = [f for f in result.functions 
                              if (f.end_line_number - f.line_number) >= self.min_function_lines]
        
        # Update class methods
        for class_info in result.classes:
            class_info.methods = [f for f in result.functions if f.parent_class == class_info.name]
    
    def _is_test_function(self, func: FunctionInfo) -> bool:
        """Check if function is a test function."""
        return (func.name.startswith('test_') or 
                func.name.endswith('_test') or
                any(d.name in ['pytest.mark', 'unittest'] for d in func.decorators))


class ASTVisitor(ast.NodeVisitor):
    """AST visitor for extracting structural information."""
    
    def __init__(self, result: ASTParseResult, config: Dict[str, Any]):
        self.result = result
        self.config = config
        self.current_class = None
        self.current_function = None
        self.scope_stack = []
        self.main_calls = []  # Track calls in __main__ block
        
    def visit_Module(self, node: ast.Module) -> None:
        """Visit module node to extract docstring."""
        if ast.get_docstring(node):
            self.result.docstring = ast.get_docstring(node)
        self.generic_visit(node)
    
    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statement."""
        for alias in node.names:
            import_info = ImportInfo(
                module=alias.name,
                alias=alias.asname,
                line_number=node.lineno,
                is_from_import=False
            )
            self.result.imports.append(import_info)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from-import statement."""
        module = node.module or ""
        level = node.level or 0
        
        for alias in node.names:
            import_info = ImportInfo(
                module=module,
                name=alias.name,
                alias=alias.asname,
                line_number=node.lineno,
                is_from_import=True,
                level=level
            )
            self.result.imports.append(import_info)
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition."""
        self._visit_function(node, is_async=False)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition."""
        self._visit_function(node, is_async=True)
    
    def _visit_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], is_async: bool) -> None:
        """Handle both sync and async function definitions."""
        # Skip inner/nested functions - only process top-level functions and methods
        if self.current_function is not None:
            # This is a nested function, skip it but still visit its children
            previous_function = self.current_function
            self.current_function = node.name
            self.generic_visit(node)
            self.current_function = previous_function
            return
        
        # Determine full name
        if self.current_class:
            full_name = f"{self.current_class}.{node.name}"
            parent_class = self.current_class
            is_method = True
        else:
            full_name = node.name
            parent_class = None
            is_method = False
        
        # Extract decorators
        decorators = [self._extract_decorator(d) for d in node.decorator_list]
        
        # Check decorator types
        is_classmethod = any(d.name == 'classmethod' for d in decorators)
        is_staticmethod = any(d.name == 'staticmethod' for d in decorators)
        is_property = any(d.name == 'property' for d in decorators)
        
        # Extract arguments with type annotations
        args = []
        if node.args.args:
            for arg in node.args.args:
                arg_info = {'name': arg.arg}
                if arg.annotation:
                    if hasattr(ast, 'unparse'):
                        arg_info['annotation'] = ast.unparse(arg.annotation)
                    else:
                        arg_info['annotation'] = 'Any'
                else:
                    arg_info['annotation'] = 'Any'
                args.append(arg_info)
        
        # Extract return annotation
        returns = None
        if hasattr(node, 'returns') and node.returns:
            returns = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)
        
        # Get docstring
        docstring = ast.get_docstring(node)
        
        # Create function info
        func_info = FunctionInfo(
            name=node.name,
            full_name=full_name,
            line_number=node.lineno,
            end_line_number=getattr(node, 'end_lineno', node.lineno),
            args=args,
            returns=returns,
            decorators=decorators,
            docstring=docstring,
            is_method=is_method,
            is_classmethod=is_classmethod,
            is_staticmethod=is_staticmethod,
            is_property=is_property,
            is_async=is_async,
            parent_class=parent_class
        )
        
        # Set current function before processing calls and children
        self.current_function = node.name
        
        # Extract function calls within this function
        call_visitor = CallVisitor()
        call_visitor.visit(node)
        func_info.calls = call_visitor.calls
        func_info.complexity = call_visitor.complexity
        
        self.result.functions.append(func_info)
        
        # Visit child nodes (will skip nested functions due to current_function being set)
        self.generic_visit(node)
        self.current_function = None
    
    def visit_If(self, node: ast.If) -> None:
        """Visit if statement, looking for __main__ pattern."""
        if self._is_main_guard(node):
            # Check if a real main() function already exists
            existing_main = any(f.name == "main" and f.is_method == False 
                              for f in self.result.functions)
            
            # Only create synthetic main() if no real main() function exists
            if not existing_main:
                # Extract calls from the __main__ block
                main_calls = []
                call_visitor = CallVisitor()
                
                # Visit the body of the if statement
                for stmt in node.body:
                    call_visitor.visit(stmt)
                
                # Create synthetic main() function if there are calls
                if call_visitor.calls:
                    main_func = FunctionInfo(
                        name="main",
                        full_name="main",
                        line_number=node.lineno,
                        end_line_number=getattr(node, 'end_lineno', node.lineno),
                        args=[],
                        returns=None,
                        decorators=[],
                        docstring="Synthetic main entry point from __main__ guard",
                        is_method=False,
                        is_classmethod=False,
                        is_staticmethod=False,
                        is_property=False,
                        is_async=False,
                        parent_class=None
                    )
                    main_func.calls = call_visitor.calls
                    main_func.complexity = call_visitor.complexity
                    self.result.functions.append(main_func)
        
        self.generic_visit(node)
    
    def _is_main_guard(self, node: ast.If) -> bool:
        """Check if this is a __name__ == "__main__" guard."""
        if not isinstance(node.test, ast.Compare):
            return False
        
        # Check for __name__ == "__main__" pattern
        test = node.test
        if (isinstance(test.left, ast.Name) and 
            test.left.id == "__name__" and
            len(test.ops) == 1 and
            isinstance(test.ops[0], ast.Eq) and
            len(test.comparators) == 1 and
            isinstance(test.comparators[0], ast.Constant) and
            test.comparators[0].value == "__main__"):
            return True
            
        return False
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        # Extract decorators
        decorators = [self._extract_decorator(d) for d in node.decorator_list]
        
        # Extract base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(ast.unparse(base) if hasattr(ast, 'unparse') else str(base))
        
        # Check if it's a dataclass or exception
        is_dataclass = any(d.name == 'dataclass' for d in decorators)
        is_exception = any('Exception' in base or 'Error' in base for base in bases)
        
        # Get docstring
        docstring = ast.get_docstring(node)
        
        class_info = ClassInfo(
            name=node.name,
            line_number=node.lineno,
            end_line_number=getattr(node, 'end_lineno', node.lineno),
            bases=bases,
            decorators=decorators,
            docstring=docstring,
            is_dataclass=is_dataclass,
            is_exception=is_exception
        )
        
        self.result.classes.append(class_info)
        
        # Visit methods within class
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignment to extract constants."""
        # Extract module-level constants
        if not self.current_class and not self.current_function:
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    try:
                        if isinstance(node.value, (ast.Constant, ast.Str, ast.Num)):
                            value = ast.literal_eval(node.value)
                            self.result.constants[target.id] = value
                    except (ValueError, TypeError):
                        pass
        
        self.generic_visit(node)
    
    def _extract_decorator(self, decorator_node: ast.expr) -> DecoratorInfo:
        """Extract decorator information."""
        if isinstance(decorator_node, ast.Name):
            return DecoratorInfo(
                name=decorator_node.id,
                line_number=decorator_node.lineno
            )
        elif isinstance(decorator_node, ast.Call):
            if isinstance(decorator_node.func, ast.Name):
                name = decorator_node.func.id
            elif isinstance(decorator_node.func, ast.Attribute):
                name = ast.unparse(decorator_node.func) if hasattr(ast, 'unparse') else str(decorator_node.func)
            else:
                name = "unknown"
            
            # Extract arguments
            args = []
            kwargs = {}
            
            for arg in decorator_node.args:
                try:
                    if isinstance(arg, ast.Constant):
                        args.append(str(arg.value))
                    else:
                        args.append(ast.unparse(arg) if hasattr(ast, 'unparse') else str(arg))
                except:
                    args.append("unknown")
            
            for keyword in decorator_node.keywords:
                try:
                    if isinstance(keyword.value, ast.Constant):
                        kwargs[keyword.arg] = str(keyword.value.value)
                    else:
                        kwargs[keyword.arg] = ast.unparse(keyword.value) if hasattr(ast, 'unparse') else str(keyword.value)
                except:
                    kwargs[keyword.arg] = "unknown"
            
            return DecoratorInfo(
                name=name,
                args=args,
                kwargs=kwargs,
                line_number=decorator_node.lineno
            )
        else:
            return DecoratorInfo(
                name=ast.unparse(decorator_node) if hasattr(ast, 'unparse') else str(decorator_node),
                line_number=getattr(decorator_node, 'lineno', 0)
            )


class CallVisitor(ast.NodeVisitor):
    """Enhanced visitor to extract detailed function calls and estimate complexity."""
    
    def __init__(self):
        self.calls = []
        self.complexity = 1  # Base complexity
        self.in_async_context = False
    
    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call and extract detailed information."""
        call_info = {
            'line_number': node.lineno,
            'type': 'async' if self.in_async_context else 'direct',
            'parameters': []
        }
        
        # Extract function name and determine call type
        if isinstance(node.func, ast.Name):
            call_info['name'] = node.func.id
            call_info['type'] = 'direct'
        elif isinstance(node.func, ast.Attribute):
            # Handle method calls with better chaining detection
            call_name = self._extract_attribute_call_name(node.func)
            call_info['name'] = call_name
            call_info['type'] = 'method'
        else:
            # Complex call patterns
            if hasattr(ast, 'unparse'):
                call_info['name'] = ast.unparse(node.func)
            else:
                call_info['name'] = 'complex_call'
            call_info['type'] = 'complex'
        
        # Extract parameter information
        try:
            call_info['parameters'] = self._extract_call_parameters(node)
        except Exception:
            call_info['parameters'] = []
        
        # Store both simple name (for backward compatibility) and detailed info
        if isinstance(call_info['name'], str):
            self.calls.append(call_info['name'])  # Backward compatibility
            self.calls.append(call_info)  # Detailed info for flow analysis
        
        self.generic_visit(node)
    
    def visit_Await(self, node: ast.Await) -> None:
        """Visit await expression."""
        self.in_async_context = True
        self.generic_visit(node)
        self.in_async_context = False
    
    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        """Visit async with statement."""
        self.complexity += 1
        self.in_async_context = True
        self.generic_visit(node)
        self.in_async_context = False
    
    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        """Visit async for loop."""
        self.complexity += 1
        self.in_async_context = True
        self.generic_visit(node)
        self.in_async_context = False
    
    def _extract_attribute_call_name(self, node: ast.Attribute) -> str:
        """Extract method call name with better chaining support."""
        if hasattr(ast, 'unparse'):
            try:
                return ast.unparse(node)
            except Exception:
                pass
        
        # Manual extraction for better control
        parts = []
        current = node
        
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        
        if isinstance(current, ast.Name):
            parts.append(current.id)
        elif hasattr(current, 'id'):
            parts.append(current.id)
        else:
            parts.append('unknown')
        
        # Reverse to get correct order
        parts.reverse()
        return '.'.join(parts)
    
    def _extract_call_parameters(self, call_node: ast.Call) -> List[str]:
        """Extract parameter information from function call."""
        parameters = []
        
        # Extract positional arguments
        for arg in call_node.args:
            if hasattr(ast, 'unparse'):
                try:
                    param_str = ast.unparse(arg)
                    parameters.append(param_str)
                except Exception:
                    parameters.append('unknown')
            else:
                # Fallback for older Python versions
                if isinstance(arg, ast.Name):
                    parameters.append(arg.id)
                elif isinstance(arg, ast.Constant):
                    parameters.append(str(arg.value))
                else:
                    parameters.append('complex_arg')
        
        # Extract keyword arguments
        for keyword in call_node.keywords:
            if hasattr(ast, 'unparse'):
                try:
                    param_str = f"{keyword.arg}={ast.unparse(keyword.value)}"
                    parameters.append(param_str)
                except Exception:
                    parameters.append(f"{keyword.arg}=unknown")
            else:
                # Fallback for older Python versions
                if isinstance(keyword.value, ast.Name):
                    parameters.append(f"{keyword.arg}={keyword.value.id}")
                elif isinstance(keyword.value, ast.Constant):
                    parameters.append(f"{keyword.arg}={keyword.value.value}")
                else:
                    parameters.append(f"{keyword.arg}=complex_value")
        
        return parameters
    
    def visit_If(self, node: ast.If) -> None:
        """Visit if statement to increase complexity."""
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_For(self, node: ast.For) -> None:
        """Visit for loop to increase complexity."""
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_While(self, node: ast.While) -> None:
        """Visit while loop to increase complexity."""
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_Try(self, node: ast.Try) -> None:
        """Visit try statement to increase complexity."""
        self.complexity += len(node.handlers)
        self.generic_visit(node)
    
    def visit_With(self, node: ast.With) -> None:
        """Visit with statement to capture context manager calls."""
        self.complexity += 1
        
        # Extract calls from context managers
        for item in node.items:
            if isinstance(item.context_expr, ast.Call):
                # This is a call within a context manager
                call_visitor = CallVisitor()
                call_visitor.visit(item.context_expr)
                self.calls.extend(call_visitor.calls)
        
        self.generic_visit(node)
    
    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        """Visit async with statement."""
        self.complexity += 1
        self.in_async_context = True
        
        # Extract calls from async context managers
        for item in node.items:
            if isinstance(item.context_expr, ast.Call):
                call_visitor = CallVisitor()
                call_visitor.in_async_context = True
                call_visitor.visit(item.context_expr)
                self.calls.extend(call_visitor.calls)
        
        self.generic_visit(node)
        self.in_async_context = False
    
    def visit_Return(self, node: ast.Return) -> None:
        """Visit return statement to capture return calls."""
        if node.value and isinstance(node.value, ast.Call):
            # This is a function call in a return statement
            call_visitor = CallVisitor()
            call_visitor.visit(node.value)
            self.calls.extend(call_visitor.calls)
        
        self.generic_visit(node)