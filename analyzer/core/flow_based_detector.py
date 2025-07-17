#!/usr/bin/env python3
"""
Flow-Based Actor Detection for Code Architecture Analyzer

Detects actors from flow analysis: entry points become TRIGGER actors,
exit points become RECEIVER actors. Proper flow direction: trigger → func → receiver.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class FlowDirection(Enum):
    """Direction of data flow between function and actor"""
    INBOUND = "inbound"     # Actor → Function (external sends to system)
    OUTBOUND = "outbound"   # Function → Actor (system sends to external)


@dataclass
class ActorFlow:
    """Represents a directional flow between a function and an actor"""
    actor_name: str
    actor_type: str  # web_request, cli_user, database, external_api, filesystem
    function_name: str  # Connected function
    direction: FlowDirection  # INBOUND or OUTBOUND
    operation: str = ""  # describe the operation (e.g., "query", "response", "trigger")
    code_context: str = ""
    confidence: float = 1.0
    needs_llm_enhancement: bool = False


@dataclass
class FlowChain:
    """Complete flow chain with bidirectional actor flows"""
    function_chain: List[str]  # Function names in order
    actor_flows: List[ActorFlow]  # All directional flows
    flow_id: str


class FlowBasedActorDetector:
    """Detects actors from flow entry/exit points"""
    
    def __init__(self):
        self.detected_flows = []
        self.flow_chains = []
    
    def detect_actors(self, ast_result, function_flows: List) -> Tuple[List[ActorFlow], List[FlowChain]]:
        """Main detection method: analyze flows to find directional actor interactions"""
        
        # Step 1: Find all directional flows (inbound and outbound)
        actor_flows = self._detect_actor_flows(ast_result)
        
        # Step 2: Build complete flow chains with bidirectional actors
        chains = self._build_flow_chains(actor_flows, function_flows)
        
        logger.info(f"Flow-based detection: {len(actor_flows)} directional flows, {len(chains)} chains")
        
        return actor_flows, chains
    
    def _detect_actor_flows(self, ast_result) -> List[ActorFlow]:
        """Detect all directional flows between functions and actors"""
        actor_flows = []
        
        # 1. INBOUND: External triggers → Functions (entry points)
        main_functions = self._find_main_functions(ast_result)
        for main_func_name in main_functions:
            actor_flows.append(ActorFlow(
                actor_name="CLIUser",
                actor_type="cli_user",
                function_name=main_func_name,  # Use qualified name
                direction=FlowDirection.INBOUND,
                operation="trigger",
                code_context="if __name__ == '__main__':"
            ))
        
        # 2. Web endpoints = INBOUND flows from HTTP requests
        for func in ast_result.functions:
            if self._is_web_endpoint(func):
                route = self._extract_route_info(func)
                # Generate optimized name: WebApi-{FunctionName} 
                function_title = func.name.replace('_', '').title()
                actor_name = f"WebApi-{function_title}"
                path = route.get('path', '/')
                
                actor_flows.append(ActorFlow(
                    actor_name=actor_name,
                    actor_type="web_request", 
                    function_name=func.name,  # Use simple function name as stored in graph
                    direction=FlowDirection.INBOUND,
                    operation="request",
                    code_context=path,  # Use path as description
                    needs_llm_enhancement=False  # No LLM needed with clear naming
                ))
        
        # 3. OUTBOUND: Functions → External systems (boundary calls)
        all_function_calls = self._extract_all_function_calls(ast_result)
        for call in all_function_calls:
            
            # HTTP outbound calls
            if self._is_http_outbound(call):
                api_name = self._deduce_api_name(call)
                actor_flows.append(ActorFlow(
                    actor_name=api_name,
                    actor_type="external_api",
                    function_name=call.caller_name,
                    direction=FlowDirection.OUTBOUND,
                    operation="request",
                    code_context=call.code,
                    needs_llm_enhancement=(api_name == "ExternalAPI")
                ))
                logger.debug(f"Detected HTTP outbound: {call.caller_name} -> {api_name} ({call.code})")
            
            # Database operations - could be bidirectional
            elif self._is_database_operation(call):
                db_name = self._deduce_database_name(call)
                operation = self._get_database_operation_type(call)
                
                if operation in ["query", "select", "fetch"]:
                    # Query: Function → Database (request), Database → Function (response)
                    actor_flows.extend([
                        ActorFlow(
                            actor_name=db_name,
                            actor_type="database",
                            function_name=call.caller_name,
                            direction=FlowDirection.OUTBOUND,
                            operation="query",
                            code_context=call.code,
                            needs_llm_enhancement=(db_name == "Database")
                        ),
                        ActorFlow(
                            actor_name=db_name,
                            actor_type="database", 
                            function_name=call.caller_name,
                            direction=FlowDirection.INBOUND,
                            operation="response",
                            code_context=call.code,
                            needs_llm_enhancement=(db_name == "Database")
                        )
                    ])
                else:
                    # Insert/Update: Function → Database (data)
                    actor_flows.append(ActorFlow(
                        actor_name=db_name,
                        actor_type="database",
                        function_name=call.caller_name,
                        direction=FlowDirection.OUTBOUND,
                        operation=operation,
                        code_context=call.code,
                        needs_llm_enhancement=(db_name == "Database")
                    ))
            
            # File operations - typically bidirectional
            elif self._is_file_operation(call):
                operation = self._get_file_operation_type(call)
                
                if operation == "read":
                    # Read: File → Function
                    actor_flows.append(ActorFlow(
                        actor_name="FileSystem",
                        actor_type="filesystem",
                        function_name=call.caller_name,
                        direction=FlowDirection.INBOUND,
                        operation="read",
                        code_context=call.code
                    ))
                elif operation == "write":
                    # Write: Function → File
                    actor_flows.append(ActorFlow(
                        actor_name="FileSystem",
                        actor_type="filesystem",
                        function_name=call.caller_name,
                        direction=FlowDirection.OUTBOUND,
                        operation="write",
                        code_context=call.code
                    ))
        
        return actor_flows
    
    def _build_flow_chains(self, actor_flows: List[ActorFlow], 
                          function_flows: List) -> List[FlowChain]:
        """Build complete flow chains with bidirectional actor flows"""
        chains = []
        
        # Group flows by function name
        function_to_flows = {}
        for flow in actor_flows:
            if flow.function_name not in function_to_flows:
                function_to_flows[flow.function_name] = []
            function_to_flows[flow.function_name].append(flow)
        
        # Start from entry points and build comprehensive chains
        processed_functions = set()
        for flow in actor_flows:
            if flow.direction == FlowDirection.INBOUND:  # Start from entry points
                if flow.function_name not in processed_functions:
                    # Trace the function chain from this entry point
                    function_chain = self._trace_function_chain(flow.function_name, function_flows)
                    
                    # Collect unique actor flows for this chain
                    chain_flows_set = set()
                    
                    # Add the entry flow
                    chain_flows_set.add((flow.actor_name, flow.direction.value, flow.function_name, flow.operation))
                    
                    # Collect flows for all functions in the chain
                    for func_name in function_chain:
                        logger.debug(f"Checking function {func_name} in chain")
                        # Direct match
                        if func_name in function_to_flows:
                            logger.debug(f"Direct match found for {func_name}")
                            for f in function_to_flows[func_name]:
                                chain_flows_set.add((f.actor_name, f.direction.value, f.function_name, f.operation))
                        
                        # Check for pattern matches (e.g., client.get_data -> HttpClient.get_data)
                        for flow_func_name, flows in function_to_flows.items():
                            if self._functions_match(func_name, flow_func_name):
                                logger.debug(f"Pattern match: {func_name} <-> {flow_func_name}")
                                for f in flows:
                                    chain_flows_set.add((f.actor_name, f.direction.value, f.function_name, f.operation))
                    
                    # Convert back to ActorFlow objects
                    chain_flows = []
                    for flow_tuple in chain_flows_set:
                        # Find the original flow object
                        for f in actor_flows:
                            if (f.actor_name, f.direction.value, f.function_name, f.operation) == flow_tuple:
                                chain_flows.append(f)
                                break
                    
                    if chain_flows:
                        chains.append(FlowChain(
                            function_chain=function_chain,
                            actor_flows=chain_flows,
                            flow_id=f"flow_{flow.function_name}_{len(chains)}"
                        ))
                        processed_functions.add(flow.function_name)
        
        return chains
    
    def _functions_match(self, flow_func_name: str, actor_func_name: str) -> bool:
        """Check if function names match allowing for different naming conventions"""
        if flow_func_name == actor_func_name:
            return True
        
        # Extract method name from both (e.g., client.get_data -> get_data, HttpClient.get_data -> get_data)
        flow_method = flow_func_name.split('.')[-1] if '.' in flow_func_name else flow_func_name
        actor_method = actor_func_name.split('.')[-1] if '.' in actor_func_name else actor_func_name
        
        return flow_method == actor_method
    
    def _get_database_operation_type(self, call) -> str:
        """Determine database operation type"""
        code = call.code.lower()
        if any(pattern in code for pattern in ['select', 'query', 'fetch', '.get(']):
            return "query"
        elif any(pattern in code for pattern in ['insert', 'create', 'add']):
            return "insert"
        elif any(pattern in code for pattern in ['update', 'modify', 'set']):
            return "update"
        elif any(pattern in code for pattern in ['delete', 'remove', 'drop']):
            return "delete"
        else:
            return "query"  # Default assumption
    
    def _get_file_operation_type(self, call) -> str:
        """Determine file operation type"""
        code = call.code.lower()
        if any(pattern in code for pattern in ['.read(', '.readlines(', '.load(']):
            return "read"
        elif any(pattern in code for pattern in ['.write(', '.writelines(', '.save(']):
            return "write"
        else:
            return "read"  # Default assumption
    
    def _find_main_functions(self, ast_result) -> List[str]:
        """Find all main functions with proper function names"""
        main_functions = []
        
        for func in ast_result.functions:
            if func.name == "main":
                # Use the actual function name as stored in graph nodes
                # This should match how NodeFactory creates function nodes
                main_functions.append(func.name)
        
        return main_functions
    
    
    def _is_web_endpoint(self, func) -> bool:
        """Check if function is a web endpoint"""
        if not func.decorators:
            return False
        
        decorators_str = str(func.decorators).lower()
        web_patterns = ['route', 'get', 'post', 'put', 'delete', 'api', 'endpoint']
        return any(pattern in decorators_str for pattern in web_patterns)
    
    
    def _is_http_outbound(self, call) -> bool:
        """Check if call is outbound HTTP request"""
        code = call.code.lower()
        http_patterns = ['requests.', 'httpx.', 'urllib.', 'aiohttp', '.get(', '.post(', '.put(', '.delete(']
        return any(pattern in code for pattern in http_patterns)
    
    def _is_database_operation(self, call) -> bool:
        """Check if call is database operation"""
        code = call.code.lower()
        db_patterns = ['.execute(', '.query(', '.save(', '.delete(', '.insert(', '.update(',
                      'cursor.', 'session.', 'connection.', 'sqlite3.']
        return any(pattern in code for pattern in db_patterns)
    
    def _is_file_operation(self, call) -> bool:
        """Check if call is file system operation"""
        code = call.code.lower()
        file_patterns = ['open(', 'file.', '.read(', '.write(', '.close(', 'pathlib.',
                        'os.path.', 'shutil.', '.json.', '.csv.']
        return any(pattern in code for pattern in file_patterns)
    
    def _deduce_api_name(self, call) -> str:
        """Deduce API name from call context"""
        code = call.code.lower()
        
        # Look for URL patterns
        if 'stripe' in code: return "StripeAPI"
        if 'github' in code: return "GitHubAPI"
        if 'google' in code: return "GoogleAPI"
        if 'aws' in code: return "AWSAPI"
        if 'slack' in code: return "SlackAPI"
        if 'twitter' in code: return "TwitterAPI"
        
        # Look for variable names
        if 'payment' in code: return "PaymentAPI"
        if 'auth' in code: return "AuthAPI"
        if 'notification' in code: return "NotificationAPI"
        if 'email' in code: return "EmailAPI"
        
        return "ExternalAPI"  # Generic - needs LLM enhancement
    
    def _deduce_database_name(self, call) -> str:
        """Deduce database name from call context"""
        code = call.code.lower()
        
        if 'sqlite' in code: return "SQLiteDB"
        if 'postgres' in code or 'psycopg' in code: return "PostgresDB"
        if 'mysql' in code: return "MySQLDB"
        if 'mongo' in code: return "MongoDB"
        if 'redis' in code: return "RedisDB"
        
        # Look for variable context
        if 'user' in code: return "UserDatabase"
        if 'product' in code: return "ProductDatabase"
        if 'order' in code: return "OrderDatabase"
        
        return "Database"  # Generic - needs LLM enhancement
    
    def _extract_route_info(self, func) -> Dict[str, str]:
        """Extract route information from function decorators"""
        route_info = {"method": "HTTP", "path": "/"}
        
        decorators_str = str(func.decorators)
        
        # Extract HTTP method
        methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
        for method in methods:
            if method.lower() in decorators_str.lower():
                route_info["method"] = method
                break
        
        # Extract path - handle both quoted formats
        if "'" in decorators_str:
            # Single quotes: @app.route('/health')
            parts = decorators_str.split("'")
            for part in parts:
                if part.startswith('/'):
                    route_info["path"] = part
                    break
        elif '"' in decorators_str:
            # Double quotes: @app.route("/health")
            parts = decorators_str.split('"')
            for part in parts:
                if part.startswith('/'):
                    route_info["path"] = part
                    break
        else:
            # Fallback: derive from function name
            func_name = func.name.lower()
            if 'health' in func_name:
                route_info["path"] = "/health"
            elif 'model' in func_name:
                route_info["path"] = "/models" 
            elif 'chat' in func_name or 'message' in func_name:
                route_info["path"] = "/v1/messages"
        
        return route_info
    
    def _trace_function_chain(self, start_function: str, function_flows: List) -> List[str]:
        """Trace function call chain from starting function"""
        chain = [start_function]
        current = start_function
        visited = {start_function}
        
        # Follow the flow chain (simplified - could be more sophisticated)
        for flow in function_flows:
            if hasattr(flow, 'source_name') and flow.source_name == current:
                if flow.target_name not in visited:
                    chain.append(flow.target_name)
                    current = flow.target_name
                    visited.add(flow.target_name)
        
        return chain
    
    def _extract_all_function_calls(self, ast_result):
        """Extract all function calls from AST result"""
        class FunctionCallInfo:
            def __init__(self, function_name, caller_name, code=None):
                self.function_name = function_name
                self.caller_name = caller_name
                self.code = code or function_name
        
        all_calls = []
        for func in ast_result.functions:
            # Use simple function name to match graph node naming
            caller_name = func.name
            
            for call in func.calls:
                if isinstance(call, str):
                    # Simple string call
                    all_calls.append(FunctionCallInfo(call, caller_name, call))
                elif isinstance(call, dict):
                    # Detailed call info
                    call_name = call.get('name', str(call))
                    all_calls.append(FunctionCallInfo(call_name, caller_name, call_name))
        
        return all_calls