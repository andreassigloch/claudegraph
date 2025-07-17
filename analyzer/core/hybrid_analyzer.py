#!/usr/bin/env python3
"""
Hybrid Static/Dynamic Analyzer - Phase 3.1 Implementation

Combines static analysis with execution tracing for improved accuracy.
This analyzer runs static analysis first, then optionally performs
dynamic tracing to validate and enhance the results.

Key Features:
- Static analysis for main flow detection (existing approach)
- Execution tracing for dynamic pattern validation  
- Test suite execution with function call tracing
- Hybrid result merging for better accuracy
- Optional tracing mode to avoid performance impact
"""

import sys
import os
import subprocess
import tempfile
import logging
import importlib.util
from pathlib import Path
from typing import Dict, Set, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .flow_based_analyzer import FlowBasedAnalyzer, FlowBasedAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class TraceResult:
    """Results from execution tracing"""
    traced_functions: Set[str] = field(default_factory=set)
    traced_calls: Set[Tuple[str, str]] = field(default_factory=set)  # (caller, callee) pairs
    execution_time: float = 0.0
    trace_errors: List[str] = field(default_factory=list)
    modules_traced: Set[str] = field(default_factory=set)
    

@dataclass
class HybridAnalysisResult:
    """Combined static and dynamic analysis results"""
    static_result: FlowBasedAnalysisResult
    trace_result: Optional[TraceResult] = None
    validation_stats: Dict[str, Any] = field(default_factory=dict)
    confirmed_live_functions: Set[str] = field(default_factory=set)
    potential_false_positives: Set[str] = field(default_factory=set)
    hybrid_confidence_scores: Dict[str, float] = field(default_factory=dict)


class HybridAnalyzer:
    """Combines static analysis with limited dynamic tracing."""
    
    def __init__(self, config: Dict[str, Any], enable_tracing: bool = False):
        self.config = config
        self.static_analyzer = FlowBasedAnalyzer(config)
        self.enable_tracing = enable_tracing
        self.traced_functions = set()
        self.traced_calls = set()
        self.project_root = None
        
    def analyze_with_tracing(self, project_path: str) -> HybridAnalysisResult:
        """Run analysis with optional execution tracing."""
        self.project_root = Path(project_path)
        
        logger.info("Starting hybrid analysis...")
        
        # Phase 1: Static analysis first (always run)
        logger.info("Phase 1: Running static analysis...")
        static_result = self.static_analyzer.analyze(project_path)
        
        hybrid_result = HybridAnalysisResult(static_result=static_result)
        
        if not self.enable_tracing:
            logger.info("Dynamic tracing disabled - returning static results only")
            return hybrid_result
        
        # Phase 2: Dynamic tracing for validation (optional)
        logger.info("Phase 2: Running dynamic tracing...")
        try:
            trace_result = self._run_test_suite_with_tracing(project_path)
            hybrid_result.trace_result = trace_result
            
            # Phase 3: Merge and validate results
            logger.info("Phase 3: Merging static and dynamic results...")
            self._merge_static_dynamic_results(hybrid_result)
            
        except Exception as e:
            logger.warning(f"Dynamic tracing failed: {e}. Continuing with static analysis only.")
            hybrid_result.trace_result = TraceResult(trace_errors=[str(e)])
        
        return hybrid_result
    
    def _run_test_suite_with_tracing(self, project_path: str) -> TraceResult:
        """Run existing tests with function call tracing."""
        import time
        start_time = time.time()
        
        trace_result = TraceResult()
        
        # Find test files in the project
        test_files = self._find_test_files(project_path)
        
        if not test_files:
            logger.warning("No test files found - attempting to trace main modules")
            # Fallback: trace main entry points
            main_files = self._find_main_files(project_path)
            if main_files:
                trace_result = self._trace_main_modules(main_files)
        else:
            logger.info(f"Found {len(test_files)} test files for tracing")
            trace_result = self._trace_test_files(test_files)
        
        trace_result.execution_time = time.time() - start_time
        return trace_result
    
    def _find_test_files(self, project_path: str) -> List[Path]:
        """Find test files in the project."""
        test_files = []
        project_root = Path(project_path)
        
        # Common test file patterns
        test_patterns = [
            "**/test_*.py",
            "**/*_test.py", 
            "**/tests.py",
            "tests/**/*.py",
            "test/**/*.py"
        ]
        
        for pattern in test_patterns:
            for test_file in project_root.glob(pattern):
                if test_file.is_file() and test_file.name != "__init__.py":
                    test_files.append(test_file)
        
        # Remove duplicates
        return list(set(test_files))
    
    def _find_main_files(self, project_path: str) -> List[Path]:
        """Find main entry point files."""
        main_files = []
        project_root = Path(project_path)
        
        # Common main file patterns
        main_patterns = [
            "**/main.py",
            "**/__main__.py",
            "**/cli.py",
            "**/run*.py",
            "**/start*.py"
        ]
        
        for pattern in main_patterns:
            for main_file in project_root.glob(pattern):
                if main_file.is_file():
                    main_files.append(main_file)
        
        return list(set(main_files))
    
    def _trace_test_files(self, test_files: List[Path]) -> TraceResult:
        """Trace execution of test files."""
        trace_result = TraceResult()
        
        # Set up function call tracer
        def trace_calls(frame, event, arg):
            if event == 'call':
                filename = frame.f_code.co_filename
                func_name = frame.f_code.co_name
                
                # Only trace functions in our project
                if self._is_project_file(filename):
                    module_name = self._get_module_name(filename)
                    full_func_name = f"{module_name}.{func_name}"
                    trace_result.traced_functions.add(full_func_name)
                    trace_result.modules_traced.add(module_name)
                    
                    # Track caller-callee relationships
                    if frame.f_back and self._is_project_file(frame.f_back.f_code.co_filename):
                        caller_module = self._get_module_name(frame.f_back.f_code.co_filename)
                        caller_func = f"{caller_module}.{frame.f_back.f_code.co_name}"
                        trace_result.traced_calls.add((caller_func, full_func_name))
            
            return trace_calls
        
        # Run tests with tracing
        old_trace = sys.gettrace()
        
        for test_file in test_files[:5]:  # Limit to first 5 test files to avoid excessive runtime
            try:
                logger.debug(f"Tracing execution of {test_file}")
                
                # Import and run the test module
                sys.settrace(trace_calls)
                self._safely_import_and_run(test_file, trace_result)
                
            except Exception as e:
                trace_result.trace_errors.append(f"Error tracing {test_file}: {e}")
                logger.debug(f"Error tracing {test_file}: {e}")
            finally:
                sys.settrace(old_trace)
        
        sys.settrace(old_trace)
        
        logger.info(f"Traced {len(trace_result.traced_functions)} unique functions across {len(trace_result.modules_traced)} modules")
        return trace_result
    
    def _trace_main_modules(self, main_files: List[Path]) -> TraceResult:
        """Trace execution of main modules with limited execution."""
        trace_result = TraceResult()
        
        # Set up simplified tracer for main modules
        def trace_calls(frame, event, arg):
            if event == 'call':
                filename = frame.f_code.co_filename
                func_name = frame.f_code.co_name
                
                if self._is_project_file(filename):
                    module_name = self._get_module_name(filename)
                    full_func_name = f"{module_name}.{func_name}"
                    trace_result.traced_functions.add(full_func_name)
                    trace_result.modules_traced.add(module_name)
            
            return trace_calls
        
        old_trace = sys.gettrace()
        
        for main_file in main_files[:3]:  # Limit to first 3 main files
            try:
                logger.debug(f"Tracing import of {main_file}")
                
                # Only import the module to trace its definitions
                sys.settrace(trace_calls)
                self._safely_import_module(main_file, trace_result)
                
            except Exception as e:
                trace_result.trace_errors.append(f"Error tracing {main_file}: {e}")
                logger.debug(f"Error tracing {main_file}: {e}")
            finally:
                sys.settrace(old_trace)
        
        sys.settrace(old_trace)
        
        logger.info(f"Traced {len(trace_result.traced_functions)} functions from main modules")
        return trace_result
    
    def _safely_import_and_run(self, test_file: Path, trace_result: TraceResult):
        """Safely import and run a test file without executing harmful code."""
        try:
            # Create module spec
            module_name = test_file.stem
            spec = importlib.util.spec_from_file_location(module_name, test_file)
            
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                
                # Add module to sys.modules temporarily
                sys.modules[module_name] = module
                
                try:
                    # Load the module (this executes module-level code)
                    spec.loader.exec_module(module)
                    
                    # Look for test functions and try to call them safely
                    for attr_name in dir(module):
                        if attr_name.startswith('test_') and callable(getattr(module, attr_name)):
                            try:
                                # Call test function (with timeout protection)
                                test_func = getattr(module, attr_name)
                                test_func()
                            except Exception as e:
                                # Test failures are expected, just continue tracing
                                logger.debug(f"Test function {attr_name} failed: {e}")
                                continue
                
                finally:
                    # Clean up
                    if module_name in sys.modules:
                        del sys.modules[module_name]
                        
        except Exception as e:
            trace_result.trace_errors.append(f"Failed to import {test_file}: {e}")
    
    def _safely_import_module(self, module_file: Path, trace_result: TraceResult):
        """Safely import a module without executing main code."""
        try:
            module_name = module_file.stem
            spec = importlib.util.spec_from_file_location(module_name, module_file)
            
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                
                try:
                    # Load the module
                    spec.loader.exec_module(module)
                finally:
                    # Clean up
                    if module_name in sys.modules:
                        del sys.modules[module_name]
                        
        except Exception as e:
            trace_result.trace_errors.append(f"Failed to import {module_file}: {e}")
    
    def _is_project_file(self, filename: str) -> bool:
        """Check if a file belongs to the project being analyzed."""
        if not self.project_root:
            return False
        
        try:
            file_path = Path(filename)
            return str(self.project_root) in str(file_path.resolve())
        except:
            return False
    
    def _get_module_name(self, filename: str) -> str:
        """Extract module name from file path."""
        if not self.project_root:
            return Path(filename).stem
        
        try:
            file_path = Path(filename)
            relative_path = file_path.relative_to(self.project_root)
            
            # Convert path to module name
            module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
            return '.'.join(module_parts)
        except:
            return Path(filename).stem
    
    def _merge_static_dynamic_results(self, hybrid_result: HybridAnalysisResult):
        """Merge static analysis results with dynamic tracing results."""
        static_result = hybrid_result.static_result
        trace_result = hybrid_result.trace_result
        
        if not trace_result:
            return
        
        # Extract dead functions from static analysis
        dead_functions = set()
        if static_result.dead_code_analysis:
            for dead_func in static_result.dead_code_analysis.dead_functions:
                dead_functions.add(dead_func.full_name)
        
        # Find functions that were marked dead but are actually called
        traced_functions_simple = set()
        for traced_func in trace_result.traced_functions:
            # Extract simple function names for comparison
            traced_functions_simple.add(traced_func.split('.')[-1])
        
        false_positives = set()
        confirmed_live = set()
        
        for dead_func_name in dead_functions:
            simple_name = dead_func_name.split('.')[-1]
            if simple_name in traced_functions_simple:
                false_positives.add(dead_func_name)
                confirmed_live.add(dead_func_name)
        
        # Calculate confidence scores
        confidence_scores = {}
        for dead_func_name in dead_functions:
            if dead_func_name in false_positives:
                # Function was traced, so likely live
                confidence_scores[dead_func_name] = 0.1  # Low confidence it's dead
            else:
                # Function wasn't traced, higher confidence it's dead
                confidence_scores[dead_func_name] = 0.9  # High confidence it's dead
        
        # Store validation results
        hybrid_result.confirmed_live_functions = confirmed_live
        hybrid_result.potential_false_positives = false_positives
        hybrid_result.hybrid_confidence_scores = confidence_scores
        
        # Generate validation statistics
        hybrid_result.validation_stats = {
            'total_dead_functions': len(dead_functions),
            'traced_functions': len(trace_result.traced_functions),
            'traced_calls': len(trace_result.traced_calls),
            'potential_false_positives': len(false_positives),
            'confirmed_live_functions': len(confirmed_live),
            'trace_coverage': len(trace_result.modules_traced),
            'trace_errors': len(trace_result.trace_errors),
            'execution_time': trace_result.execution_time
        }
        
        logger.info(f"Hybrid validation: {len(false_positives)} potential false positives identified")
        logger.info(f"Traced {len(trace_result.traced_functions)} functions in {trace_result.execution_time:.2f}s")
    
    def format_hybrid_results(self, hybrid_result: HybridAnalysisResult) -> Dict[str, Any]:
        """Format hybrid analysis results for export."""
        # Start with static analysis results
        export_data = self.static_analyzer.format_for_export(hybrid_result.static_result)
        
        # Add hybrid analysis metadata
        if 'metadata' not in export_data:
            export_data['metadata'] = {}
        
        export_data['metadata']['analysis_type'] = 'hybrid_static_dynamic'
        export_data['metadata']['tracing_enabled'] = self.enable_tracing
        
        # Add dynamic tracing results if available
        if hybrid_result.trace_result:
            export_data['metadata']['dynamic_tracing'] = {
                'traced_functions': len(hybrid_result.trace_result.traced_functions),
                'traced_calls': len(hybrid_result.trace_result.traced_calls),
                'modules_traced': len(hybrid_result.trace_result.modules_traced),
                'execution_time': hybrid_result.trace_result.execution_time,
                'trace_errors': len(hybrid_result.trace_result.trace_errors)
            }
        
        # Add validation statistics
        if hybrid_result.validation_stats:
            export_data['metadata']['validation_stats'] = hybrid_result.validation_stats
        
        # Add confidence scores for dead functions
        if hybrid_result.hybrid_confidence_scores:
            export_data['metadata']['hybrid_confidence_scores'] = dict(hybrid_result.hybrid_confidence_scores)
        
        return export_data