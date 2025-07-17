#!/usr/bin/env python3
"""
PyCG Integration Module for Code Architecture Analyzer

Provides integration with PyCG (Python Call Graph) tool for static call graph analysis.
Handles subprocess execution, JSON parsing, and error recovery.
"""

import os
import json
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CallGraphNode:
    """Represents a node in the call graph."""
    name: str
    module: str
    file_path: str
    line_number: Optional[int] = None
    is_builtin: bool = False
    is_external: bool = False


@dataclass
class CallGraphEdge:
    """Represents an edge (call relationship) in the call graph."""
    caller: str
    callee: str
    call_type: str = "direct"  # direct, indirect, dynamic
    line_number: Optional[int] = None
    confidence: float = 1.0


@dataclass
class CallGraphResult:
    """Complete call graph analysis result."""
    nodes: Dict[str, CallGraphNode] = field(default_factory=dict)
    edges: List[CallGraphEdge] = field(default_factory=list)
    entry_points: List[str] = field(default_factory=list)
    external_calls: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error_message: Optional[str] = None
    pycg_output: Optional[Dict] = None


class PyCGIntegration:
    """
    Integration wrapper for PyCG call graph analysis.
    
    Handles subprocess execution, output parsing, and provides
    fallback mechanisms when PyCG fails or is unavailable.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize PyCG integration with configuration."""
        self.config = config or {}
        
        # PyCG configuration
        pycg_config = self.config.get('pycg', {})
        self.enabled = pycg_config.get('enabled', True)
        self.timeout = pycg_config.get('timeout', 60)
        self.max_depth = pycg_config.get('max_depth', 10)
        self.include_builtin = pycg_config.get('include_builtin', False)
        self.include_external = pycg_config.get('include_external', True)
        
        # Check if PyCG is available
        self.pycg_available = self._check_pycg_availability()
        
        if self.enabled and not self.pycg_available:
            logger.warning("PyCG is enabled but not available. Call graph analysis will be limited.")
    
    def _check_pycg_availability(self) -> bool:
        """Check if PyCG is installed and available."""
        try:
            result = subprocess.run(
                ['pycg', '--help'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.debug(f"PyCG not available: {e}")
            return False
    
    def analyze_project(self, project_path: str) -> CallGraphResult:
        """
        Analyze project call graph using PyCG.
        
        Args:
            project_path: Path to the Python project root
            
        Returns:
            CallGraphResult with nodes, edges, and statistics
        """
        if not self.enabled:
            logger.info("PyCG integration disabled")
            return CallGraphResult(success=False, error_message="PyCG integration disabled")
        
        if not self.pycg_available:
            logger.warning("PyCG not available, generating basic call graph")
            return self._generate_basic_call_graph(project_path)
        
        try:
            return self._run_pycg_analysis(project_path)
        except Exception as e:
            logger.error(f"PyCG analysis failed: {e}")
            return CallGraphResult(success=False, error_message=str(e))
    
    def _run_pycg_analysis(self, project_path: str) -> CallGraphResult:
        """Run PyCG analysis on the project."""
        project_path = Path(project_path).absolute()
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_output = temp_file.name
        
        try:
            # Build PyCG command
            cmd = [
                'pycg',
                '--package', str(project_path),
                '--output', temp_output,
                '--format', 'json'
            ]
            
            # Add optional parameters
            if self.max_depth:
                cmd.extend(['--max-iter', str(self.max_depth)])
            
            if not self.include_builtin:
                cmd.append('--no-builtin')
            
            logger.info(f"Running PyCG analysis: {' '.join(cmd)}")
            
            # Execute PyCG
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=project_path.parent
            )
            
            if result.returncode != 0:
                error_msg = f"PyCG failed with code {result.returncode}: {result.stderr}"
                logger.error(error_msg)
                return CallGraphResult(success=False, error_message=error_msg)
            
            # Parse PyCG output
            return self._parse_pycg_output(temp_output, project_path)
            
        except subprocess.TimeoutExpired:
            error_msg = f"PyCG analysis timed out after {self.timeout} seconds"
            logger.error(error_msg)
            return CallGraphResult(success=False, error_message=error_msg)
        
        except Exception as e:
            error_msg = f"PyCG execution error: {e}"
            logger.error(error_msg)
            return CallGraphResult(success=False, error_message=error_msg)
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_output)
            except OSError:
                pass
    
    def _parse_pycg_output(self, output_file: str, project_path: Path) -> CallGraphResult:
        """Parse PyCG JSON output into CallGraphResult."""
        try:
            with open(output_file, 'r') as f:
                pycg_data = json.load(f)
            
            result = CallGraphResult(pycg_output=pycg_data)
            
            # Parse nodes (functions/methods)
            if isinstance(pycg_data, dict):
                for caller, callees in pycg_data.items():
                    # Add caller node
                    if caller not in result.nodes:
                        result.nodes[caller] = self._create_node_from_name(caller, project_path)
                    
                    # Add callee nodes and edges
                    if isinstance(callees, list):
                        for callee in callees:
                            if callee not in result.nodes:
                                result.nodes[callee] = self._create_node_from_name(callee, project_path)
                            
                            # Create edge
                            edge = CallGraphEdge(
                                caller=caller,
                                callee=callee,
                                call_type="direct",
                                confidence=0.95
                            )
                            result.edges.append(edge)
            
            # Identify entry points (functions not called by others)
            called_functions = {edge.callee for edge in result.edges}
            all_functions = set(result.nodes.keys())
            result.entry_points = list(all_functions - called_functions)
            
            # Identify external calls
            result.external_calls = [
                name for name, node in result.nodes.items()
                if node.is_external or node.is_builtin
            ]
            
            # Generate statistics
            result.statistics = {
                'total_nodes': len(result.nodes),
                'total_edges': len(result.edges),
                'entry_points': len(result.entry_points),
                'external_calls': len(result.external_calls),
                'internal_functions': len([n for n in result.nodes.values() if not n.is_external and not n.is_builtin]),
                'builtin_functions': len([n for n in result.nodes.values() if n.is_builtin]),
                'external_functions': len([n for n in result.nodes.values() if n.is_external])
            }
            
            result.success = True
            logger.info(f"PyCG analysis successful: {result.statistics['total_nodes']} nodes, {result.statistics['total_edges']} edges")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to parse PyCG output: {e}"
            logger.error(error_msg)
            return CallGraphResult(success=False, error_message=error_msg)
    
    def _create_node_from_name(self, function_name: str, project_path: Path) -> CallGraphNode:
        """Create CallGraphNode from function name string."""
        # Parse function name format: module.function or class.method
        parts = function_name.split('.')
        
        # Determine if this is builtin/external
        is_builtin = any(part in function_name for part in ['__builtin__', '<built-in>', '<builtin>'])
        is_external = not str(project_path) in function_name and not is_builtin
        
        # Extract module name
        if len(parts) >= 2 and not is_builtin and not is_external:
            module = parts[0]
            file_path = str(project_path / f"{module}.py")
        else:
            module = parts[0] if parts else "unknown"
            file_path = "unknown"
        
        return CallGraphNode(
            name=function_name,
            module=module,
            file_path=file_path,
            is_builtin=is_builtin,
            is_external=is_external
        )
    
    def _generate_basic_call_graph(self, project_path: str) -> CallGraphResult:
        """
        Generate basic call graph when PyCG is not available.
        
        This provides a fallback with limited functionality.
        """
        logger.info("Generating basic call graph (PyCG fallback)")
        
        result = CallGraphResult()
        
        try:
            # Simple analysis based on file discovery
            project_path = Path(project_path)
            python_files = list(project_path.rglob("*.py"))
            
            # Create basic nodes for discovered files
            for py_file in python_files:
                relative_path = py_file.relative_to(project_path)
                module_name = str(relative_path).replace('/', '.').replace('.py', '')
                
                # Create a basic node for the module
                node = CallGraphNode(
                    name=f"{module_name}.__main__",
                    module=module_name,
                    file_path=str(py_file),
                    is_builtin=False,
                    is_external=False
                )
                result.nodes[node.name] = node
            
            # Basic statistics
            result.statistics = {
                'total_nodes': len(result.nodes),
                'total_edges': 0,
                'entry_points': len(result.nodes),
                'external_calls': 0,
                'internal_functions': len(result.nodes),
                'builtin_functions': 0,
                'external_functions': 0,
                'fallback_mode': True
            }
            
            result.success = True
            logger.info(f"Basic call graph generated: {len(result.nodes)} nodes")
            
        except Exception as e:
            error_msg = f"Basic call graph generation failed: {e}"
            logger.error(error_msg)
            result.error_message = error_msg
            result.success = False
        
        return result
    
    def get_function_calls(self, result: CallGraphResult, function_name: str) -> List[str]:
        """Get all functions called by a specific function."""
        if not result.success:
            return []
        
        return [edge.callee for edge in result.edges if edge.caller == function_name]
    
    def get_function_callers(self, result: CallGraphResult, function_name: str) -> List[str]:
        """Get all functions that call a specific function."""
        if not result.success:
            return []
        
        return [edge.caller for edge in result.edges if edge.callee == function_name]
    
    def is_entry_point(self, result: CallGraphResult, function_name: str) -> bool:
        """Check if a function is an entry point (not called by others)."""
        return function_name in result.entry_points
    
    def get_call_chain(self, result: CallGraphResult, start_function: str, max_depth: int = 5) -> List[List[str]]:
        """Get call chains starting from a function."""
        if not result.success:
            return []
        
        chains = []
        visited = set()
        
        def dfs(current: str, chain: List[str], depth: int):
            if depth >= max_depth or current in visited:
                return
            
            visited.add(current)
            chain.append(current)
            
            callees = self.get_function_calls(result, current)
            if not callees:
                # End of chain
                chains.append(chain.copy())
            else:
                for callee in callees:
                    dfs(callee, chain.copy(), depth + 1)
        
        dfs(start_function, [], 0)
        return chains