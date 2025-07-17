#!/usr/bin/env python3
"""
Graph Builder Module for Code Architecture Analyzer

Builds ontology-compliant graph structures from analysis results and exports
to Neo4j-compatible JSON format with proper UUID generation and metadata.
"""

import uuid
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field

from ..core.ast_parser import ASTParseResult
from ..detection.pattern_matcher import ActorDetectionResult
from ..core.pycg_integration import CallGraphResult

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Represents a node in the architecture graph."""
    uuid: str
    node_type: str  # SYS, MOD, FUNC, ACTOR, SCHEMA, FCHAIN
    name: str
    description: str
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphRelationship:
    """Represents a relationship between nodes."""
    uuid: str
    relationship_type: str  # compose, allocate, flow, relation
    source_uuid: str
    target_uuid: str
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphBuildResult:
    """Complete graph build result."""
    nodes: List[GraphNode] = field(default_factory=list)
    relationships: List[GraphRelationship] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error_message: Optional[str] = None


class OntologyGraphBuilder:
    """
    Builds ontology-compliant graph structures from analysis results.
    
    Combines deterministic analysis, actor detection, and call graph results
    into a unified graph structure suitable for Neo4j import.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize graph builder with configuration."""
        self.config = config or {}
        
        # Graph generation settings
        graph_config = self.config.get('graph', {})
        self.include_metadata = graph_config.get('include_metadata', True)
        self.generate_stats = graph_config.get('generate_statistics', True)
        self.validate_ontology = graph_config.get('validate_ontology', True)
        
        # Node name limits from ontology
        self.max_name_length = 25
        
        # Track generated UUIDs to ensure uniqueness
        self.generated_uuids: Set[str] = set()
        
        # Node and relationship registries
        self.nodes_registry: Dict[str, GraphNode] = {}
        self.relationships_registry: List[GraphRelationship] = []
    
    def build_graph(
        self,
        project_name: str,
        ast_results: Dict[str, ASTParseResult],
        actor_results: Dict[str, ActorDetectionResult],
        call_graph_result: Optional[CallGraphResult] = None,
        project_metadata: Optional[Dict[str, Any]] = None
    ) -> GraphBuildResult:
        """
        Build complete ontology-compliant graph from analysis results.
        
        Args:
            project_name: Name of the analyzed project
            ast_results: AST parsing results by module
            actor_results: Actor detection results by module
            call_graph_result: Optional call graph analysis result
            project_metadata: Optional project-level metadata
            
        Returns:
            GraphBuildResult with nodes, relationships, and metadata
        """
        try:
            logger.info("Building ontology-compliant graph structure")
            
            # Reset registries
            self.nodes_registry.clear()
            self.relationships_registry.clear()
            self.generated_uuids.clear()
            
            # Build system node (root)
            system_node = self._create_system_node(project_name, project_metadata or {})
            self._register_node(system_node)
            
            # Build module nodes
            module_nodes = {}
            for module_name, ast_result in ast_results.items():
                module_node = self._create_module_node(module_name, ast_result)
                module_nodes[module_name] = module_node
                self._register_node(module_node)
                
                # Create compose relationship: SYS -> MOD
                self._create_compose_relationship(system_node.uuid, module_node.uuid)
            
            # Build function nodes
            function_nodes = {}
            for module_name, ast_result in ast_results.items():
                module_node = module_nodes[module_name]
                
                for func_info in ast_result.functions:
                    func_node = self._create_function_node(func_info, module_name)
                    function_key = f"{module_name}.{func_info.name}"
                    function_nodes[function_key] = func_node
                    self._register_node(func_node)
                    
                    # Create compose relationship: MOD -> FUNC
                    self._create_compose_relationship(module_node.uuid, func_node.uuid)
                    
                    # Create allocate relationship: MOD -> FUNC (resource allocation)
                    self._create_allocate_relationship(module_node.uuid, func_node.uuid)
            
            # Build actor nodes and relationships
            actor_nodes = {}
            for module_name, actor_result in actor_results.items():
                for detection_match in actor_result.detected_actors:
                    actor_key = self._generate_actor_key(detection_match)
                    
                    if actor_key not in actor_nodes:
                        actor_node = self._create_actor_node(detection_match)
                        actor_nodes[actor_key] = actor_node
                        self._register_node(actor_node)
                    
                    # Create flow relationships between functions and actors
                    if detection_match.function_name:
                        func_key = f"{module_name}.{detection_match.function_name}"
                        if func_key in function_nodes:
                            func_node = function_nodes[func_key]
                            actor_node = actor_nodes[actor_key]
                            
                            # Determine flow direction based on actor type
                            if self._is_input_actor(detection_match.actor_type):
                                # ACTOR -> FUNC (data flows from actor to function)
                                self._create_flow_relationship(
                                    actor_node.uuid, func_node.uuid, detection_match
                                )
                            else:
                                # FUNC -> ACTOR (data flows from function to actor)
                                self._create_flow_relationship(
                                    func_node.uuid, actor_node.uuid, detection_match
                                )
            
            # Build call graph relationships if available
            if call_graph_result and call_graph_result.success:
                self._build_call_relationships(call_graph_result, function_nodes)
            
            # Generate final result
            result = GraphBuildResult(
                nodes=list(self.nodes_registry.values()),
                relationships=self.relationships_registry,
                success=True
            )
            
            # Add metadata and statistics
            if self.include_metadata:
                result.metadata = self._generate_metadata(project_name, project_metadata)
            
            if self.generate_stats:
                result.statistics = self._generate_statistics(result)
            
            logger.info(f"Graph built successfully: {len(result.nodes)} nodes, {len(result.relationships)} relationships")
            return result
            
        except Exception as e:
            logger.error(f"Graph building failed: {e}")
            return GraphBuildResult(success=False, error_message=str(e))
    
    def _create_system_node(self, project_name: str, metadata: Dict[str, Any]) -> GraphNode:
        """Create the root system node."""
        name = self._sanitize_name(project_name, "System")
        
        description = f"Python project with {metadata.get('file_count', 0)} modules"
        if 'total_lines' in metadata:
            description += f" and {metadata['total_lines']} lines of code"
        
        return GraphNode(
            uuid=self._generate_uuid(f"SYS_{project_name}"),
            node_type="SYS",
            name=name,
            description=description,
            properties={
                "project_name": project_name,
                "node_type": "SYS"
            },
            metadata=metadata
        )
    
    def _create_module_node(self, module_name: str, ast_result: ASTParseResult) -> GraphNode:
        """Create a module node from AST result."""
        name = self._sanitize_name(module_name, "Module")
        
        description = f"Python module with {len(ast_result.functions)} functions"
        if len(ast_result.classes) > 0:
            description += f" and {len(ast_result.classes)} classes"
        
        return GraphNode(
            uuid=self._generate_uuid(f"MOD_{module_name}"),
            node_type="MOD",
            name=name,
            description=description,
            properties={
                "module_name": module_name,
                "file_path": ast_result.file_path,
                "function_count": len(ast_result.functions),
                "class_count": len(ast_result.classes),
                "import_count": len(ast_result.imports),
                "node_type": "MOD"
            },
            metadata={
                "parse_success": ast_result.success,
                "parse_errors": ast_result.error_count,
                "lines_of_code": getattr(ast_result, 'lines_of_code', 0)
            }
        )
    
    def _create_function_node(self, func_info, module_name: str) -> GraphNode:
        """Create a function node from function info."""
        name = self._sanitize_name(func_info.name, "Func")
        
        description = f"Function {func_info.name}"
        if hasattr(func_info, 'docstring') and func_info.docstring:
            # Truncate docstring for description
            doc_preview = func_info.docstring[:50].strip()
            if len(func_info.docstring) > 50:
                doc_preview += "..."
            description += f": {doc_preview}"
        
        properties = {
            "function_name": func_info.name,
            "module_name": module_name,
            "line_number": getattr(func_info, 'line_number', 0),
            "is_method": getattr(func_info, 'is_method', False),
            "is_async": getattr(func_info, 'is_async', False),
            "parameter_count": len(getattr(func_info, 'parameters', [])),
            "node_type": "FUNC"
        }
        
        # Add class information if it's a method
        if hasattr(func_info, 'class_name') and func_info.class_name:
            properties["class_name"] = func_info.class_name
            properties["is_method"] = True
        
        return GraphNode(
            uuid=self._generate_uuid(f"FUNC_{module_name}_{func_info.name}"),
            node_type="FUNC",
            name=name,
            description=description,
            properties=properties,
            metadata={
                "complexity": getattr(func_info, 'complexity', 1),
                "has_docstring": bool(getattr(func_info, 'docstring', False))
            }
        )
    
    def _create_actor_node(self, detection_match) -> GraphNode:
        """Create an actor node from detection match."""
        actor_type = detection_match.actor_type.value
        
        # Generate readable name
        base_name = self._get_actor_base_name(detection_match)
        name = self._sanitize_name(f"{actor_type}{base_name}", "Actor")
        
        # Generate description
        evidence = detection_match.evidence
        technology = evidence.get('technology', 'unknown')
        description = f"{actor_type} using {base_name}"
        if technology != 'unknown':
            description += f" ({technology})"
        
        properties = {
            "actor_type": actor_type,
            "technology": technology,
            "confidence": detection_match.confidence,
            "pattern_name": detection_match.pattern_name,
            "node_type": "ACTOR"
        }
        
        # Add specific properties based on actor type
        if detection_match.actor_type.value == "HttpClient":
            properties.update({
                "endpoints": evidence.get('endpoints', []),
                "methods": evidence.get('methods', [])
            })
        elif detection_match.actor_type.value == "Database":
            properties.update({
                "database_type": evidence.get('database_type', 'unknown'),
                "connection_info": evidence.get('connection_info', {})
            })
        elif detection_match.actor_type.value == "FileSystem":
            properties.update({
                "file_operations": evidence.get('operations', []),
                "file_paths": evidence.get('paths', [])
            })
        
        return GraphNode(
            uuid=self._generate_uuid(f"ACTOR_{actor_type}_{base_name}_{technology}"),
            node_type="ACTOR",
            name=name,
            description=description,
            properties=properties,
            metadata={
                "detection_confidence": detection_match.confidence,
                "evidence_count": len(evidence),
                "line_numbers": detection_match.line_numbers
            }
        )
    
    def _create_compose_relationship(self, source_uuid: str, target_uuid: str) -> None:
        """Create a compose relationship (hierarchical)."""
        relationship = GraphRelationship(
            uuid=self._generate_uuid(f"compose_{source_uuid}_{target_uuid}"),
            relationship_type="compose",
            source_uuid=source_uuid,
            target_uuid=target_uuid,
            properties={
                "relationship_type": "compose",
                "direction": "hierarchical"
            }
        )
        self.relationships_registry.append(relationship)
    
    def _create_allocate_relationship(self, source_uuid: str, target_uuid: str) -> None:
        """Create an allocate relationship (resource allocation)."""
        relationship = GraphRelationship(
            uuid=self._generate_uuid(f"allocate_{source_uuid}_{target_uuid}"),
            relationship_type="allocate",
            source_uuid=source_uuid,
            target_uuid=target_uuid,
            properties={
                "relationship_type": "allocate",
                "direction": "resource"
            }
        )
        self.relationships_registry.append(relationship)
    
    def _create_flow_relationship(self, source_uuid: str, target_uuid: str, detection_match) -> None:
        """Create a flow relationship (data/control flow)."""
        properties = {
            "relationship_type": "flow",
            "flow_type": "data",
            "confidence": detection_match.confidence,
            "pattern_name": detection_match.pattern_name
        }
        
        # Add flow-specific metadata
        if hasattr(detection_match, 'evidence'):
            properties["evidence_type"] = detection_match.evidence.get('type', 'unknown')
        
        relationship = GraphRelationship(
            uuid=self._generate_uuid(f"flow_{source_uuid}_{target_uuid}"),
            relationship_type="flow",
            source_uuid=source_uuid,
            target_uuid=target_uuid,
            properties=properties,
            metadata={
                "actor_type": detection_match.actor_type.value,
                "module_name": detection_match.module_name,
                "function_name": detection_match.function_name
            }
        )
        self.relationships_registry.append(relationship)
    
    def _build_call_relationships(self, call_graph: CallGraphResult, function_nodes: Dict[str, GraphNode]) -> None:
        """Build flow relationships from call graph."""
        if not call_graph.success:
            return
        
        for edge in call_graph.edges:
            # Find corresponding function nodes
            caller_node = None
            callee_node = None
            
            for func_key, func_node in function_nodes.items():
                if edge.caller in func_key or func_node.properties.get('function_name') in edge.caller:
                    caller_node = func_node
                if edge.callee in func_key or func_node.properties.get('function_name') in edge.callee:
                    callee_node = func_node
            
            # Create flow relationship if both nodes found
            if caller_node and callee_node and caller_node.uuid != callee_node.uuid:
                relationship = GraphRelationship(
                    uuid=self._generate_uuid(f"call_flow_{caller_node.uuid}_{callee_node.uuid}"),
                    relationship_type="flow",
                    source_uuid=caller_node.uuid,
                    target_uuid=callee_node.uuid,
                    properties={
                        "relationship_type": "flow",
                        "flow_type": "control",
                        "call_type": edge.call_type,
                        "confidence": edge.confidence
                    },
                    metadata={
                        "from_call_graph": True,
                        "line_number": edge.line_number
                    }
                )
                self.relationships_registry.append(relationship)
    
    def _register_node(self, node: GraphNode) -> None:
        """Register a node in the registry."""
        self.nodes_registry[node.uuid] = node
    
    def _generate_uuid(self, seed: str) -> str:
        """Generate deterministic UUID from seed string."""
        # Use UUID5 for deterministic generation
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # DNS namespace
        generated_uuid = str(uuid.uuid5(namespace, seed))
        
        # Ensure uniqueness within this build
        counter = 0
        original_uuid = generated_uuid
        while generated_uuid in self.generated_uuids:
            counter += 1
            generated_uuid = str(uuid.uuid5(namespace, f"{seed}_{counter}"))
        
        self.generated_uuids.add(generated_uuid)
        return generated_uuid
    
    def _sanitize_name(self, name: str, prefix: str = "") -> str:
        """Sanitize node names according to ontology rules."""
        # Remove invalid characters and convert to PascalCase
        clean_name = ''.join(c for c in name if c.isalnum() or c in '_-')
        
        # Convert to PascalCase
        parts = clean_name.replace('-', '_').split('_')
        pascal_name = ''.join(word.capitalize() for word in parts if word)
        
        # Add prefix if provided
        if prefix and not pascal_name.startswith(prefix):
            pascal_name = prefix + pascal_name
        
        # Truncate to max length
        if len(pascal_name) > self.max_name_length:
            pascal_name = pascal_name[:self.max_name_length]
        
        # Ensure it starts with a letter
        if pascal_name and not pascal_name[0].isalpha():
            pascal_name = 'Node' + pascal_name
        
        return pascal_name or 'UnknownNode'
    
    def _generate_actor_key(self, detection_match) -> str:
        """Generate unique key for actor deduplication."""
        evidence = detection_match.evidence
        technology = evidence.get('technology', 'unknown')
        actor_type = detection_match.actor_type.value
        
        # Include specific identifiers based on actor type
        if actor_type == "HttpClient":
            endpoints = evidence.get('endpoints', [])
            key_part = '_'.join(endpoints[:2]) if endpoints else 'unknown'
        elif actor_type == "Database":
            db_type = evidence.get('database_type', 'unknown')
            key_part = db_type
        else:
            key_part = 'unknown'
        
        return f"{actor_type}_{technology}_{key_part}"
    
    def _get_actor_base_name(self, detection_match) -> str:
        """Get base name for actor node."""
        evidence = detection_match.evidence
        
        if detection_match.actor_type.value == "HttpClient":
            endpoints = evidence.get('endpoints', [])
            if endpoints:
                # Extract domain or path for name
                endpoint = endpoints[0]
                if '://' in endpoint:
                    return endpoint.split('://')[1].split('/')[0].replace('.', '')[:10]
                return 'unknown'
            return 'unknown'
        
        elif detection_match.actor_type.value == "Database":
            db_type = evidence.get('database_type', 'unknown')
            return db_type if db_type != 'unknown' else 'unknown'
        
        elif detection_match.actor_type.value == "FileSystem":
            operations = evidence.get('operations', [])
            return operations[0] if operations else 'unknown'
        
        return 'unknown'
    
    def _is_input_actor(self, actor_type) -> bool:
        """Determine if actor is typically an input source."""
        input_actors = ["Database", "FileSystem", "ConfigManager", "Cache"]
        return actor_type.value in input_actors
    
    def _generate_metadata(self, project_name: str, project_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata for the graph."""
        return {
            "analysis_version": "1.0.0",
            "analyzer_type": "deterministic",
            "timestamp": datetime.now().isoformat(),
            "project_name": project_name,
            "configuration": self.config,
            **project_metadata
        }
    
    def _generate_statistics(self, result: GraphBuildResult) -> Dict[str, Any]:
        """Generate statistics for the graph."""
        node_types = {}
        relationship_types = {}
        
        for node in result.nodes:
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
        
        for rel in result.relationships:
            relationship_types[rel.relationship_type] = relationship_types.get(rel.relationship_type, 0) + 1
        
        return {
            "total_nodes": len(result.nodes),
            "total_relationships": len(result.relationships),
            "node_types": node_types,
            "relationship_types": relationship_types,
            "unique_uuids": len(self.generated_uuids)
        }
    
    def export_to_json(self, result: GraphBuildResult, output_path: str) -> bool:
        """Export graph result to Neo4j-compatible JSON format."""
        try:
            # Build Neo4j-compatible structure
            export_data = {
                "metadata": result.metadata,
                "nodes": [],
                "relationships": []
            }
            
            # Add statistics to metadata
            if result.statistics:
                export_data["metadata"]["statistics"] = result.statistics
            
            # Convert nodes to flat property format
            for node in result.nodes:
                node_data = {
                    "uuid": node.uuid,
                    "type": node.node_type,
                    "Name": node.name,
                    "Descr": node.description
                }
                
                # Add all properties as flat fields
                for key, value in node.properties.items():
                    if key not in ["Name", "Descr", "uuid", "type"]:
                        node_data[key] = value
                
                export_data["nodes"].append(node_data)
            
            # Convert relationships to flat format
            for rel in result.relationships:
                rel_data = {
                    "uuid": rel.uuid,
                    "type": rel.relationship_type,
                    "source": rel.source_uuid,
                    "target": rel.target_uuid
                }
                
                # Add all properties as flat fields
                for key, value in rel.properties.items():
                    if key not in ["uuid", "type", "source", "target"]:
                        rel_data[key] = value
                
                export_data["relationships"].append(rel_data)
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Graph exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export graph: {e}")
            return False