#!/usr/bin/env python3
"""
Relationship Builder for Code Architecture Analyzer

Handles creation of ontology-compliant relationships between nodes.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field

from .node_factory import OntologyNode
from ..core.ast_parser import ASTParseResult, FunctionInfo

logger = logging.getLogger(__name__)


@dataclass
class OntologyRelationship:
    """Represents a relationship in the ontology graph."""
    uuid: str
    type: str
    source_uuid: str
    target_uuid: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary format."""
        result = {
            "uuid": self.uuid,
            "type": self.type,
            "source": self.source_uuid,
            "target": self.target_uuid
        }
        result.update(self.properties)
        return result


class RelationshipBuilder:
    """Builder for creating different types of ontology relationships."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize relationship builder with configuration."""
        self.config = config or {}
        self._relationship_counter = 0
    
    def create_compose_relationships(self, system_node: OntologyNode, 
                                   module_nodes: List[OntologyNode]) -> List[OntologyRelationship]:
        """Create compose relationships between system and modules."""
        relationships = []
        
        for module_node in module_nodes:
            rel = self._create_relationship(
                "compose", 
                system_node.uuid, 
                module_node.uuid,
                {
                    "RelType": "system_module_composition",
                    "Description": f"System {system_node.name} contains module {module_node.name}"
                }
            )
            relationships.append(rel)
        
        return relationships
    
    def create_allocate_relationships(self, module_nodes: List[OntologyNode],
                                    function_nodes: List[OntologyNode]) -> List[OntologyRelationship]:
        """Create allocate relationships between modules and functions."""
        relationships = []
        
        # Group functions by module
        module_function_map = {}
        for func_node in function_nodes:
            module_name = func_node.properties.get("ModuleName", "")
            if module_name not in module_function_map:
                module_function_map[module_name] = []
            module_function_map[module_name].append(func_node)
        
        # Create relationships
        for module_node in module_nodes:
            module_name = module_node.properties.get("ModuleName", module_node.name)
            functions = module_function_map.get(module_name, [])
            
            for func_node in functions:
                rel = self._create_relationship(
                    "allocate",
                    module_node.uuid,
                    func_node.uuid,
                    {
                        "RelType": "module_function_allocation",
                        "Description": f"Module {module_node.name} allocates function {func_node.name}"
                    }
                )
                relationships.append(rel)
        
        return relationships
    
    def create_flow_relationships(self, actor_results: List[Any], 
                                func_nodes: List[OntologyNode],
                                actor_nodes: List[OntologyNode]) -> List[OntologyRelationship]:
        """Generate flow relationships between actors and functions."""
        relationships = []
        
        if not actor_results or not func_nodes:
            logger.warning("No actor results or function nodes provided for flow relationship generation")
            return relationships
        
        # Create actor name to node mapping
        actor_map = {node.properties.get("ActorName", node.name): node for node in actor_nodes}
        func_map = {node.properties.get("FunctionName", node.name): node for node in func_nodes}
        
        for actor_result in actor_results:
            try:
                # Extract flows from actor result
                flows = self._extract_flows_from_actor_result(actor_result)
                
                for flow in flows:
                    flow_rels = self._create_flow_relationship_from_flow(flow, actor_map, func_map)
                    relationships.extend(flow_rels)
                    
            except Exception as e:
                logger.error(f"Error processing actor flows: {e}")
                continue
        
        logger.info(f"Generated {len(relationships)} flow relationships")
        return relationships
    
    def create_function_flow_relationships(self, ast_results: List[ASTParseResult],
                                         function_nodes: List[OntologyNode]) -> List[OntologyRelationship]:
        """Generate function-to-function flow relationships from AST results."""
        relationships = []
        
        # Build function mapping
        func_map = {}
        for func_node in function_nodes:
            func_name = func_node.properties.get("FunctionName", func_node.name)
            module_name = func_node.properties.get("ModuleName", "")
            
            # Store both simple name and full qualified name
            func_map[func_name] = func_node
            if module_name:
                qualified_name = f"{module_name}.{func_name}"
                func_map[qualified_name] = func_node
        
        # Process each AST result
        for ast_result in ast_results:
            for func_info in ast_result.functions:
                source_func = func_map.get(func_info.name)
                if not source_func:
                    continue
                
                # Process function calls
                for call in func_info.calls:
                    # Handle both string calls and complex call objects
                    call_name = call if isinstance(call, str) else getattr(call, 'name', str(call))
                    target_func = self._resolve_function_call(call_name, func_map, ast_result.module_name)
                    if target_func and target_func != source_func:
                        rel = self._create_relationship(
                            "flow",
                            source_func.uuid,
                            target_func.uuid,
                            {
                                "FlowType": "function_call",
                                "CallName": call_name,
                                "Direction": "outbound",
                                "FlowDescr": f"Function {func_info.name} calls {call_name}",
                                "SourceModule": ast_result.module_name,
                                "TargetModule": target_func.properties.get("ModuleName", "")
                            }
                        )
                        relationships.append(rel)
        
        logger.info(f"Generated {len(relationships)} function flow relationships")
        return relationships
    
    def create_fchain_nodes_from_relationships(self, flow_relationships: List[OntologyRelationship]) -> List[OntologyNode]:
        """Generate FCHAIN nodes from flow relationships."""
        # Group relationships by source to identify flow chains
        source_flows = {}
        for rel in flow_relationships:
            if rel.type == "flow":
                source = rel.source_uuid
                if source not in source_flows:
                    source_flows[source] = []
                source_flows[source].append(rel)
        
        # Create FCHAIN nodes for functions with multiple outbound flows
        fchain_nodes = []
        for source_uuid, flows in source_flows.items():
            if len(flows) > 1:  # Only create FCHAIN for functions with multiple flows
                fchain_node = OntologyNode(
                    uuid=f"FCHAIN_{source_uuid}",
                    type="FCHAIN",
                    name=f"FlowChain_{len(fchain_nodes)}",
                    descr=f"Flow chain with {len(flows)} outbound flows",
                    properties={
                        "SourceFunction": source_uuid,
                        "FlowCount": len(flows),
                        "TargetFunctions": [rel.target_uuid for rel in flows]
                    }
                )
                fchain_nodes.append(fchain_node)
        
        logger.info(f"Generated {len(fchain_nodes)} FCHAIN nodes")
        return fchain_nodes
    
    def _create_relationship(self, rel_type: str, source_uuid: str, target_uuid: str,
                           properties: Optional[Dict[str, Any]] = None) -> OntologyRelationship:
        """Create a generic relationship."""
        self._relationship_counter += 1
        
        return OntologyRelationship(
            uuid=f"REL_{self._relationship_counter:06d}",
            type=rel_type,
            source_uuid=source_uuid,
            target_uuid=target_uuid,
            properties=properties or {}
        )
    
    def _extract_flows_from_actor_result(self, actor_result: Any) -> List[Dict[str, Any]]:
        """Extract flow information from actor result."""
        flows = []
        
        try:
            # Handle different actor result formats
            if hasattr(actor_result, 'flows'):
                flows = actor_result.flows
            elif hasattr(actor_result, 'detected_actors'):
                # Extract flows from detected actors
                for actor in actor_result.detected_actors:
                    if hasattr(actor, 'flows'):
                        flows.extend(actor.flows)
            elif isinstance(actor_result, dict):
                flows = actor_result.get('flows', [])
        except Exception as e:
            logger.warning(f"Could not extract flows from actor result: {e}")
        
        return flows
    
    def _create_flow_relationship_from_flow(self, flow: Dict[str, Any], 
                                          actor_map: Dict[str, OntologyNode],
                                          func_map: Dict[str, OntologyNode]) -> List[OntologyRelationship]:
        """Create flow relationship from a single flow."""
        relationships = []
        
        try:
            flow_type = flow.get('type', 'unknown')
            direction = flow.get('direction', 'unknown')
            actor_name = flow.get('actor', '')
            function_name = flow.get('function', '')
            
            actor_node = actor_map.get(actor_name)
            func_node = func_map.get(function_name)
            
            if not actor_node or not func_node:
                return relationships
            
            if direction == 'inbound':
                # Actor -> Function
                rel = self._create_relationship(
                    "flow",
                    actor_node.uuid,
                    func_node.uuid,
                    {
                        "FlowType": flow_type,
                        "Direction": direction,
                        "FlowDescr": f"Actor {actor_name} triggers function {function_name}",
                        "ActorType": actor_node.properties.get("ActorType", ""),
                        "Operation": flow.get('operation', 'trigger')
                    }
                )
                relationships.append(rel)
            elif direction == 'outbound':
                # Function -> Actor
                rel = self._create_relationship(
                    "flow",
                    func_node.uuid,
                    actor_node.uuid,
                    {
                        "FlowType": flow_type,
                        "Direction": direction,
                        "FlowDescr": f"Function {function_name} calls actor {actor_name}",
                        "ActorType": actor_node.properties.get("ActorType", ""),
                        "Operation": flow.get('operation', 'request')
                    }
                )
                relationships.append(rel)
        except Exception as e:
            logger.error(f"Error creating flow relationship from flow: {e}")
        
        return relationships
    
    def _resolve_function_call(self, call: str, func_map: Dict[str, OntologyNode], 
                              current_module: str) -> Optional[OntologyNode]:
        """Resolve a function call to a function node."""
        # Try direct name match first
        if call in func_map:
            return func_map[call]
        
        # Try with current module prefix
        qualified_call = f"{current_module}.{call}"
        if qualified_call in func_map:
            return func_map[qualified_call]
        
        # Try pattern matching for method calls (obj.method)
        if '.' in call:
            method_name = call.split('.')[-1]
            if method_name in func_map:
                return func_map[method_name]
        
        # No match found
        return None