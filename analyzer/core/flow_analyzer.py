"""
Flow Analysis Engine for Code Architecture Analyzer

This module provides deterministic flow relationship detection and analysis
between functions without requiring LLM calls. It analyzes AST data to identify
function calls, method calls, and other flow patterns.

Key Features:
- Deterministic flow detection from AST data
- Clean FlowDescr and FlowDef generation
- Support for various call patterns (direct calls, method calls, async/await)
- Optional LLM enhancement integration
- FCHAIN node generation for functional sequences
"""

import ast
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

try:
    from ..llm.client import LLMRequest
except ImportError:
    LLMRequest = None

logger = logging.getLogger(__name__)


@dataclass
class FlowRelationship:
    """Represents a flow relationship between two functions."""
    source_uuid: str
    target_uuid: str
    source_name: str
    target_name: str
    flow_descr: str  # Human-readable description
    flow_def: str    # Technical definition
    context: Dict[str, Any]
    confidence: float = 1.0  # Always 1.0 for deterministic flows


@dataclass  
class FunctionCall:
    """Represents a function call detected in the AST."""
    caller_name: str
    callee_name: str
    call_type: str  # 'direct', 'method', 'attribute', 'async'
    parameters: List[str]
    line_number: int
    context: Dict[str, Any]


class FlowDetector:
    """Detects flow relationships between functions from AST data."""
    
    def __init__(self):
        self.function_calls: List[FunctionCall] = []
        self.function_signatures: Dict[str, Dict[str, Any]] = {}
        self.purpose_patterns = {
            # Common function name patterns and their purposes
            'setup': 'initialization',
            'init': 'initialization', 
            'configure': 'configuration',
            'validate': 'data validation',
            'check': 'validation',
            'verify': 'verification',
            'transform': 'data transformation',
            'process': 'data processing',
            'parse': 'data parsing',
            'convert': 'data conversion',
            'save': 'data persistence',
            'store': 'data storage',
            'write': 'data writing',
            'persist': 'data persistence',
            'fetch': 'data retrieval',
            'get': 'data retrieval',
            'load': 'data loading',
            'read': 'data reading',
            'retrieve': 'data retrieval',
            'connect': 'connection establishment',
            'disconnect': 'connection termination',
            'close': 'resource cleanup',
            'cleanup': 'resource cleanup',
            'destroy': 'resource cleanup',
            'send': 'data transmission',
            'receive': 'data reception',
            'handle': 'event handling',
            'execute': 'execution',
            'run': 'execution',
            'start': 'initialization',
            'stop': 'termination',
            'log': 'logging',
            'debug': 'debugging',
            'error': 'error handling',
            'exception': 'error handling'
        }
    
    def analyze_flows(self, ast_data: Dict[str, Any]) -> List[FlowRelationship]:
        """
        Analyze flow relationships from AST data.
        
        Args:
            ast_data: Dictionary containing AST parsing results
            
        Returns:
            List of FlowRelationship objects
        """
        logger.info("Starting deterministic flow analysis")
        
        # Extract function calls from AST data
        self._extract_function_calls(ast_data)
        
        # Extract function signatures for better FlowDef generation
        self._extract_function_signatures(ast_data)
        
        # Generate flow relationships
        relationships = self._generate_flow_relationships()
        
        logger.info(f"Generated {len(relationships)} flow relationships")
        return relationships
    
    def _extract_function_calls(self, ast_data: Dict[str, Any]) -> None:
        """Extract function calls from AST data."""
        self.function_calls = []
        
        for file_path, file_data in ast_data.items():
            if not isinstance(file_data, dict) or 'functions' not in file_data:
                continue
                
            module_name = file_data.get('module_name', Path(file_path).stem)
            
            for func_info in file_data['functions']:
                caller_name = func_info.get('name', '')
                calls = func_info.get('calls', [])
                
                # Process direct function calls
                for call in calls:
                    if isinstance(call, str):
                        # Simple call name
                        self.function_calls.append(FunctionCall(
                            caller_name=f"{module_name}.{caller_name}",
                            callee_name=call,
                            call_type='direct',
                            parameters=[],
                            line_number=func_info.get('line_number', 0),
                            context={'module': module_name, 'file': file_path}
                        ))
                    elif isinstance(call, dict):
                        # Detailed call information
                        callee = call.get('name', call.get('function', ''))
                        call_type = call.get('type', 'direct')
                        
                        # Handle method calls and attribute access
                        if '.' in callee:
                            call_type = 'method'
                        elif call_type == 'async':
                            call_type = 'async'
                        
                        self.function_calls.append(FunctionCall(
                            caller_name=f"{module_name}.{caller_name}",
                            callee_name=callee,
                            call_type=call_type,
                            parameters=call.get('parameters', []),
                            line_number=call.get('line_number', func_info.get('line_number', 0)),
                            context={'module': module_name, 'file': file_path}
                        ))
                        
                        # Check for callback patterns - function names passed as parameters
                        parameters = call.get('parameters', [])
                        for param in parameters:
                            if isinstance(param, str) and self._is_function_reference(param):
                                # This parameter is likely a function reference (callback)
                                self.function_calls.append(FunctionCall(
                                    caller_name=f"{module_name}.{caller_name}",
                                    callee_name=param,
                                    call_type='callback',
                                    parameters=[],
                                    line_number=call.get('line_number', func_info.get('line_number', 0)),
                                    context={
                                        'module': module_name, 
                                        'file': file_path,
                                        'callback_via': callee,
                                        'callback_pattern': True
                                    }
                                ))

    
    def _is_function_reference(self, param: str) -> bool:
        """Check if a parameter string is likely a function reference."""
        # Remove common non-function patterns
        if not param or len(param) < 2:
            return False
        
        # Skip obvious non-function patterns
        non_function_patterns = [
            "'", '"',  # String literals
            '[', ']',  # Lists
            '{', '}',  # Dicts
            '(', ')',  # Tuples/expressions
            '=',       # Keyword arguments
            '+', '-', '*', '/', '%',  # Math operations
            'True', 'False', 'None',  # Literals
        ]
        
        for pattern in non_function_patterns:
            if pattern in param:
                return False
        
        # Check if it looks like a function name (alphanumeric + underscore, no spaces)
        import re
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', param.strip()):
            # Additional heuristics: function names often start with lowercase or underscore
            return param[0].islower() or param[0] == '_'
        
        return False
    
    def _extract_function_signatures(self, ast_data: Dict[str, Any]) -> None:
        """Extract function signatures for FlowDef generation."""
        self.function_signatures = {}
        
        for file_path, file_data in ast_data.items():
            if not isinstance(file_data, dict) or 'functions' not in file_data:
                continue
                
            module_name = file_data.get('module_name', Path(file_path).stem)
            
            for func_info in file_data['functions']:
                func_name = func_info.get('name', '')
                full_name = f"{module_name}.{func_name}"
                
                # Extract parameters - handle both list and dict formats
                parameters = func_info.get('parameters', [])
                if not parameters:
                    # Fallback to 'args' field from AST parser
                    parameters = func_info.get('args', [])
                
                self.function_signatures[full_name] = {
                    'parameters': parameters,
                    'return_annotation': func_info.get('return_annotation'),
                    'decorators': func_info.get('decorators', []),
                    'is_async': func_info.get('is_async', False),
                    'docstring': func_info.get('docstring', ''),
                    'full_name': full_name
                }
                
                # Also store with simple name for lookup
                self.function_signatures[func_name] = self.function_signatures[full_name]
    
    def _generate_flow_relationships(self) -> List[FlowRelationship]:
        """Generate flow relationships from detected function calls."""
        relationships = []
        
        for call in self.function_calls:
            # Generate deterministic description
            flow_descr = self.generate_deterministic_description(
                call.caller_name, call.callee_name, call.context
            )
            
            # Generate clean flow definition
            flow_def = self.extract_flow_definition(call)
            
            # Create relationship
            relationship = FlowRelationship(
                source_uuid=self._generate_function_uuid(call.caller_name),
                target_uuid=self._generate_function_uuid(call.callee_name),
                source_name=call.caller_name,
                target_name=call.callee_name,
                flow_descr=flow_descr,
                flow_def=flow_def,
                context=call.context,
                confidence=1.0
            )
            
            relationships.append(relationship)
        
        return relationships
    
    def generate_deterministic_description(self, source_func: str, target_func: str, 
                                         context: Dict[str, Any]) -> str:
        """
        Generate deterministic flow description without LLM.
        
        Args:
            source_func: Name of the calling function
            target_func: Name of the called function  
            context: Additional context information
            
        Returns:
            Human-readable flow description
        """
        # Extract simple names for readability
        source_simple = source_func.split('.')[-1]
        target_simple = target_func.split('.')[-1]
        
        # Infer purpose from function names
        purpose = self._infer_purpose(source_simple, target_simple, context)
        
        # Generate description
        return f"{source_simple} calls {target_simple} for {purpose}"
    
    def _infer_purpose(self, source: str, target: str, context: Dict[str, Any]) -> str:
        """Infer the purpose of a function call based on names and context."""
        target_lower = target.lower()
        
        # Check for exact matches first
        for pattern, purpose in self.purpose_patterns.items():
            if pattern in target_lower:
                return purpose
        
        # Check for common patterns in combination
        if any(p in target_lower for p in ['setup', 'init', 'configure']):
            return 'initialization'
        elif any(p in target_lower for p in ['validate', 'check', 'verify']):
            return 'validation'
        elif any(p in target_lower for p in ['save', 'store', 'write', 'persist']):
            return 'data persistence'
        elif any(p in target_lower for p in ['fetch', 'get', 'load', 'read', 'retrieve']):
            return 'data retrieval'
        elif any(p in target_lower for p in ['transform', 'process', 'parse', 'convert']):
            return 'data processing'
        elif any(p in target_lower for p in ['connect', 'disconnect', 'close']):
            return 'connection management'
        elif any(p in target_lower for p in ['send', 'receive', 'handle']):
            return 'communication'
        elif any(p in target_lower for p in ['log', 'debug', 'error']):
            return 'logging and debugging'
        
        # Default purpose
        return 'processing'
    
    def extract_flow_definition(self, call: FunctionCall) -> str:
        """
        Generate clean FlowDef without redundant labels.
        
        Args:
            call: FunctionCall object with call details
            
        Returns:
            Clean technical flow definition
        """
        # Try multiple lookup strategies for method calls
        callee_signature = self.function_signatures.get(call.callee_name, {})
        
        if not callee_signature and '.' in call.callee_name:
            # For instance method calls like "config.load_config", try to find class method
            instance_name, method_name = call.callee_name.split('.', 1)
            
            # Try common class name patterns
            potential_class_names = []
            
            # Convert instance name to potential class names
            if instance_name == 'config':
                potential_class_names.extend(['ConfigManager', 'Configuration', 'Config'])
            elif instance_name == 'client':
                potential_class_names.extend(['HttpClient', 'Client', 'APIClient'])
            elif instance_name == 'db':
                potential_class_names.extend(['Database', 'DatabaseManager', 'DB'])
            elif instance_name == 'logger':
                potential_class_names.extend(['Logger', 'LogManager'])
            
            # Try to find the method signature under class names
            for class_name in potential_class_names:
                class_method_name = f"{class_name}.{method_name}"
                callee_signature = self.function_signatures.get(class_method_name, {})
                if callee_signature:
                    break
            
            # If still not found, try looking for just the method name
            if not callee_signature:
                callee_signature = self.function_signatures.get(method_name, {})
        
        # Extract parameter information
        params_info = self._extract_parameters_info(call, callee_signature)
        
        # Extract return information
        return_info = self._extract_return_info(callee_signature)
        
        # Generate clean definition
        if params_info and return_info:
            return f"{params_info} → {return_info}"
        elif params_info:
            return f"{params_info} → void"
        elif return_info:
            return f"void → {return_info}"
        else:
            return "function call"
    
    def _extract_parameters_info(self, call: FunctionCall, signature: Dict[str, Any]) -> str:
        """Extract parameter information for FlowDef."""
        params = signature.get('parameters', [])
        
        if not params:
            # Try to infer from call parameters if available
            call_params = call.parameters
            if call_params:
                return f"{len(call_params)} parameters"
            return "void"
        
        # Format parameters as type annotations when available
        param_strs = []
        for param in params:
            if isinstance(param, dict):
                name = param.get('name', '')
                type_hint = param.get('annotation', 'Any')
                if name and name != 'self':  # Skip 'self' parameter
                    param_strs.append(f"{name}: {type_hint}")
            elif isinstance(param, str) and param != 'self':
                # Handle string format parameters
                param_strs.append(f"{param}: Any")
        
        if not param_strs:
            return "void"
        
        return ", ".join(param_strs)
    
    def _extract_return_info(self, signature: Dict[str, Any]) -> str:
        """Extract return type information for FlowDef."""
        return_annotation = signature.get('return_annotation')
        
        if return_annotation and return_annotation != 'None':
            # Try to infer description from function purpose
            docstring = signature.get('docstring', '')
            description = self._infer_return_description(docstring, signature)
            
            if description:
                return f"{return_annotation} ({description})"
            else:
                return str(return_annotation)
        elif return_annotation == 'None':
            return "None"
        
        # Try to infer return type from function name
        func_name = signature.get('full_name', '').lower()
        if any(pattern in func_name for pattern in ['get', 'fetch', 'load', 'read']):
            return "Any (retrieved data)"
        elif any(pattern in func_name for pattern in ['validate', 'check', 'verify']):
            return "bool (validation result)"
        elif any(pattern in func_name for pattern in ['save', 'write', 'store', 'create']):
            return "None"
        else:
            return "Any"
    
    def _infer_return_description(self, docstring: str, signature: Dict[str, Any]) -> str:
        """Infer return value description from docstring and function info."""
        if not docstring:
            return ""
        
        # Look for common return patterns in docstring
        docstring_lower = docstring.lower()
        
        if 'return' in docstring_lower:
            # Try to extract return description
            lines = docstring.split('\n')
            for line in lines:
                line_lower = line.lower().strip()
                if line_lower.startswith('return'):
                    # Extract description after 'return'
                    desc = line.split(':', 1)[-1].strip()
                    if desc:
                        return desc[:50]  # Limit length
        
        return ""
    
    def _generate_function_uuid(self, func_name: str) -> str:
        """Generate UUID for function based on its name."""
        # Simple UUID generation - in real implementation, this should 
        # integrate with the existing UUID generation system
        clean_name = func_name.replace('.', '_').replace(':', '_')
        return f"func_{clean_name}"
    
    def detect_fchains(self, relationships: List[FlowRelationship]) -> List[Dict[str, Any]]:
        """
        Detect functional chains (FCHAIN nodes) following TRIGGER→FUNCTION→RECEIVER pattern.
        
        Args:
            relationships: List of flow relationships including Actor→Function and Function→Actor
            
        Returns:
            List of FCHAIN node data
        """
        logger.info("Detecting TRIGGER→FUNCTION→RECEIVER chains")
        
        # Separate relationships by type
        trigger_flows = []  # Actor → Function
        processing_flows = []  # Function → Function
        receiver_flows = []  # Function → Actor
        
        for rel in relationships:
            source_type = self._infer_node_type(rel.source_name)
            target_type = self._infer_node_type(rel.target_name)
            
            if source_type == 'ACTOR' and target_type == 'FUNC':
                trigger_flows.append(rel)
            elif source_type == 'FUNC' and target_type == 'FUNC':
                processing_flows.append(rel)
            elif source_type == 'FUNC' and target_type == 'ACTOR':
                receiver_flows.append(rel)
        
        # Detect complete TRIGGER→FUNCTION→RECEIVER chains
        chains = self._detect_trigger_receiver_chains(trigger_flows, processing_flows, receiver_flows)
        
        # Generate FCHAIN nodes
        fchain_nodes = []
        for i, chain in enumerate(chains):
            if len(chain['functions']) >= 1:  # At least one function between trigger and receiver
                fchain_node = {
                    'uuid': f"fchain_{i}",
                    'type': 'FCHAIN',
                    'name': self._generate_trigger_chain_name(chain),
                    'description': self._generate_trigger_chain_description(chain),
                    'trigger': chain['trigger'],
                    'functions': chain['functions'],
                    'receiver': chain['receiver'],
                    'length': len(chain['functions']) + 2,  # Include trigger and receiver
                    'pattern': 'TRIGGER→FUNCTION→RECEIVER'
                }
                fchain_nodes.append(fchain_node)
        
        logger.info(f"Detected {len(fchain_nodes)} TRIGGER→FUNCTION→RECEIVER chains")
        return fchain_nodes
    
    def _detect_trigger_receiver_chains(self, trigger_flows: List[FlowRelationship], 
                                       processing_flows: List[FlowRelationship], 
                                       receiver_flows: List[FlowRelationship]) -> List[Dict[str, Any]]:
        """Detect complete TRIGGER→FUNCTION→RECEIVER chains."""
        chains = []
        
        # Build lookup maps
        triggers_by_target = {}  # function -> triggering actor
        for rel in trigger_flows:
            triggers_by_target[rel.target_name] = rel.source_name
        
        receivers_by_source = {}  # function -> receiving actor
        for rel in receiver_flows:
            if rel.source_name not in receivers_by_source:
                receivers_by_source[rel.source_name] = []
            receivers_by_source[rel.source_name].append(rel.target_name)
        
        # Build function call graph
        func_call_graph = {}
        for rel in processing_flows:
            if rel.source_name not in func_call_graph:
                func_call_graph[rel.source_name] = []
            func_call_graph[rel.source_name].append(rel.target_name)
        
        # Find chains starting from triggered functions
        for triggered_func, trigger_actor in triggers_by_target.items():
            # Trace function chain from this entry point
            function_chain = self._trace_function_chain(triggered_func, func_call_graph, set())
            
            # Find receivers for functions in the chain
            receivers = []
            for func in function_chain:
                if func in receivers_by_source:
                    receivers.extend(receivers_by_source[func])
            
            if receivers:  # Only create chain if it has both trigger and receiver
                chain = {
                    'trigger': trigger_actor,
                    'functions': function_chain,
                    'receiver': receivers[0] if len(receivers) == 1 else receivers  # Multiple receivers possible
                }
                chains.append(chain)
        
        return chains
    
    def _trace_function_chain(self, start_func: str, call_graph: Dict[str, List[str]], 
                            visited: Set[str]) -> List[str]:
        """Trace a function call chain starting from a function."""
        chain = [start_func]
        visited.add(start_func)
        current = start_func
        
        while current in call_graph and current not in visited:
            targets = call_graph[current]
            # Follow single path chains
            if len(targets) == 1 and targets[0] not in visited:
                current = targets[0]
                chain.append(current)
                visited.add(current)
            else:
                # Handle multiple targets by taking the first non-visited
                next_target = None
                for target in targets:
                    if target not in visited:
                        next_target = target
                        break
                if next_target:
                    current = next_target
                    chain.append(current)
                    visited.add(current)
                else:
                    break
        
        return chain
    
    def _infer_node_type(self, node_name: str) -> str:
        """Infer if a node is ACTOR or FUNC based on naming patterns."""
        # Actor naming patterns
        actor_patterns = [
            'endpoint', 'actor', 'client', 'service', 'manager', 'handler',
            'api', 'database', 'file', 'http', 'web', 'server'
        ]
        
        node_lower = node_name.lower()
        for pattern in actor_patterns:
            if pattern in node_lower:
                return 'ACTOR'
        
        # Function naming patterns (typically have module.function format)
        if '.' in node_name or 'func' in node_lower:
            return 'FUNC'
        
        # Default assumption
        return 'FUNC'
    
    def _generate_trigger_chain_name(self, chain: Dict[str, Any]) -> str:
        """Generate a name for a TRIGGER→FUNCTION→RECEIVER chain."""
        trigger = chain['trigger'].split('.')[-1] if isinstance(chain['trigger'], str) else 'Trigger'
        receiver = chain['receiver']
        if isinstance(receiver, list):
            receiver = receiver[0] if receiver else 'Receiver'
        receiver = receiver.split('.')[-1]
        
        # Get main function in the chain
        main_func = chain['functions'][0].split('.')[-1] if chain['functions'] else 'Process'
        
        return f"{trigger}To{receiver}Via{main_func}"
    
    def _generate_trigger_chain_description(self, chain: Dict[str, Any]) -> str:
        """Generate a description for a TRIGGER→FUNCTION→RECEIVER chain."""
        trigger = chain['trigger'].split('.')[-1] if isinstance(chain['trigger'], str) else 'external trigger'
        receiver = chain['receiver']
        if isinstance(receiver, list):
            receiver_str = f"{len(receiver)} receivers" if len(receiver) > 1 else receiver[0].split('.')[-1]
        else:
            receiver_str = receiver.split('.')[-1]
        
        func_count = len(chain['functions'])
        if func_count == 1:
            return f"Complete flow from {trigger} through {chain['functions'][0].split('.')[-1]} to {receiver_str}"
        else:
            return f"Complete flow from {trigger} through {func_count} functions to {receiver_str}"


class FlowEnhancer:
    """Optional LLM enhancement for flow descriptions."""
    
    def __init__(self, llm_client=None, enabled: bool = True):
        self.llm_client = llm_client
        self.enabled = enabled and llm_client is not None
        logger.info(f"FlowEnhancer initialized, enabled: {self.enabled}")
    
    def enhance_flow_descriptions(self, relationships: List[FlowRelationship]) -> List[FlowRelationship]:
        """
        Enhance flow descriptions using LLM when available.
        
        Args:
            relationships: List of flow relationships with deterministic descriptions
            
        Returns:
            List of relationships with enhanced descriptions (or original if LLM unavailable)
        """
        if not self.enabled:
            logger.debug("LLM enhancement disabled, returning original descriptions")
            return relationships
        
        logger.info(f"Enhancing {len(relationships)} flow descriptions with LLM")
        
        enhanced_relationships = []
        for rel in relationships:
            try:
                enhanced_descr = self._enhance_single_description(rel)
                
                # Create new relationship with enhanced description
                enhanced_rel = FlowRelationship(
                    source_uuid=rel.source_uuid,
                    target_uuid=rel.target_uuid,
                    source_name=rel.source_name,
                    target_name=rel.target_name,
                    flow_descr=enhanced_descr,
                    flow_def=rel.flow_def,  # Keep original FlowDef
                    context=rel.context,
                    confidence=rel.confidence
                )
                enhanced_relationships.append(enhanced_rel)
                
            except Exception as e:
                logger.warning(f"Failed to enhance description for {rel.source_name} -> {rel.target_name}: {e}")
                # Fallback to original
                enhanced_relationships.append(rel)
        
        return enhanced_relationships
    
    def _enhance_single_description(self, relationship: FlowRelationship) -> str:
        """Enhance a single flow description using LLM."""
        if not self.enabled:
            return relationship.flow_descr
        
        # Prepare prompt for LLM
        prompt = f"""
        Enhance this function call description to be more descriptive and meaningful:
        
        Original: {relationship.flow_descr}
        Source function: {relationship.source_name}
        Target function: {relationship.target_name}
        Technical definition: {relationship.flow_def}
        Context: {relationship.context}
        
        Provide a single, clear sentence describing what this function call accomplishes in the system.
        Focus on the business purpose and data flow, not just the technical action.
        """
        
        try:
            # Create LLM request
            if LLMRequest is None:
                logger.warning("LLMRequest not available, using deterministic description")
                return relationship.flow_descr
            
            # Get model from LLM client
            model = getattr(self.llm_client, 'model', 'local-model')
            
            request = LLMRequest(
                prompt=prompt,
                model=model,
                max_tokens=100,
                temperature=0.1
            )
            
            # Call LLM using correct interface
            response = self.llm_client.call(request)
            
            if response and hasattr(response, 'content') and response.content and len(response.content.strip()) > 0:
                return response.content.strip()
            else:
                return relationship.flow_descr
                
        except Exception as e:
            logger.warning(f"LLM enhancement failed: {e}")
            return relationship.flow_descr