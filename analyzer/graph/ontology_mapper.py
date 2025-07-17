#!/usr/bin/env python3
"""
Ontology Mapper for Code Architecture Analyzer

Ensures ontology compliance and handles mapping between different data formats.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from .node_factory import OntologyNode
from .relationship_builder import OntologyRelationship

logger = logging.getLogger(__name__)


@dataclass
class GraphData:
    """Complete graph data structure."""
    nodes: List[OntologyNode] = field(default_factory=list)
    relationships: List[OntologyRelationship] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class OntologyMapper:
    """Maps graph data to ontology-compliant formats."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ontology mapper with configuration."""
        self.config = config or {}
        self.ontology_version = self.config.get('ontology', {}).get('version', '1.0')
        
        # Define valid node types and their properties
        self.valid_node_types = {
            'SYS': ['ProjectPath', 'TotalFiles', 'PythonFiles', 'TotalLines', 'ProjectType'],
            'MOD': ['FilePath', 'RelativePath', 'ModuleName', 'LinesOfCode', 'FunctionCount', 'ClassCount', 'ImportCount'],
            'FUNC': ['FunctionName', 'ModuleName', 'IsAsync', 'IsMethod', 'IsPrivate', 'IsProperty', 
                    'LineNumber', 'EndLineNumber', 'ParameterCount', 'Parameters', 'ReturnType', 
                    'Decorators', 'CallCount', 'FunctionCalls', 'HasDocstring', 'Complexity'],
            'ACTOR': ['ActorType', 'ActorName', 'Confidence', 'Evidence', 'Operations'],
            'FCHAIN': ['SourceFunction', 'FlowCount', 'TargetFunctions']
        }
        
        # Define valid relationship types
        self.valid_relationship_types = {
            'compose': ['system-module composition'],
            'allocate': ['module-function allocation'],
            'flow': ['function-function flow', 'actor-function flow', 'function-actor flow'],
            'call': ['function call relationship'],
            'use': ['dependency usage relationship']
        }
    
    def map_to_ontology_format(self, graph_data: GraphData) -> Dict[str, Any]:
        """Map graph data to ontology-compliant format."""
        logger.info("Mapping graph data to ontology format")
        
        # Validate graph data
        validation_result = self._validate_graph_structure(graph_data)
        if not validation_result['is_valid']:
            logger.warning(f"Graph validation issues: {validation_result['warnings']}")
        
        # Convert to ontology format
        ontology_graph = {
            "metadata": self._create_ontology_metadata(graph_data),
            "nodes": self._map_nodes_to_ontology(graph_data.nodes),
            "relationships": self._map_relationships_to_ontology(graph_data.relationships),
            "statistics": self._generate_graph_statistics(graph_data),
            "validation": validation_result
        }
        
        logger.info(f"Mapped {len(graph_data.nodes)} nodes and {len(graph_data.relationships)} relationships")
        return ontology_graph
    
    def map_to_neo4j_format(self, graph_data: GraphData) -> Dict[str, Any]:
        """Map graph data to Neo4j-compatible format."""
        logger.info("Mapping graph data to Neo4j format")
        
        neo4j_nodes = []
        neo4j_relationships = []
        
        # Convert nodes to Neo4j format
        for node in graph_data.nodes:
            neo4j_node = {
                "id": node.uuid,
                "labels": [node.type],
                "properties": self._flatten_properties(node.to_dict())
            }
            neo4j_nodes.append(neo4j_node)
        
        # Convert relationships to Neo4j format
        for rel in graph_data.relationships:
            neo4j_rel = {
                "id": rel.uuid,
                "type": rel.type.upper(),
                "startNode": rel.source_uuid,
                "endNode": rel.target_uuid,
                "properties": self._flatten_properties(rel.properties)
            }
            neo4j_relationships.append(neo4j_rel)
        
        return {
            "nodes": neo4j_nodes,
            "relationships": neo4j_relationships,
            "metadata": {
                "format": "neo4j",
                "created_at": datetime.utcnow().isoformat(),
                "node_count": len(neo4j_nodes),
                "relationship_count": len(neo4j_relationships)
            }
        }
    
    def map_to_analysis_result(self, graph_data: GraphData, 
                             project_structure: Any = None) -> Dict[str, Any]:
        """Map graph data to analysis result format."""
        
        # Count nodes by type
        node_counts = {}
        for node in graph_data.nodes:
            node_type = node.type
            node_counts[node_type] = node_counts.get(node_type, 0) + 1
        
        # Count relationships by type
        rel_counts = {}
        for rel in graph_data.relationships:
            rel_type = rel.type
            rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1
        
        # Extract trigger and receiver actors
        trigger_actors = []
        receiver_actors = []
        
        for rel in graph_data.relationships:
            if rel.type == "flow":
                direction = rel.properties.get("Direction", "")
                if direction == "inbound":
                    # Actor triggering function
                    trigger_actors.append(rel.source_uuid)
                elif direction == "outbound":
                    # Function calling actor
                    receiver_actors.append(rel.target_uuid)
        
        return {
            "graph": {
                "nodes": [node.to_dict() for node in graph_data.nodes],
                "relationships": [rel.to_dict() for rel in graph_data.relationships]
            },
            "metadata": {
                "analysis_type": "ontology_based",
                "ontology_version": self.ontology_version,
                "created_at": datetime.utcnow().isoformat(),
                **graph_data.metadata
            },
            "statistics": {
                "nodes": {
                    "total": len(graph_data.nodes),
                    "by_type": node_counts
                },
                "relationships": {
                    "total": len(graph_data.relationships),
                    "by_type": rel_counts
                },
                "actors": {
                    "trigger_count": len(set(trigger_actors)),
                    "receiver_count": len(set(receiver_actors)),
                    "unique_triggers": list(set(trigger_actors)),
                    "unique_receivers": list(set(receiver_actors))
                },
                "flow_chains": node_counts.get("FCHAIN", 0)
            }
        }
    
    def _create_ontology_metadata(self, graph_data: GraphData) -> Dict[str, Any]:
        """Create ontology metadata."""
        return {
            "version": self.ontology_version,
            "created_at": datetime.utcnow().isoformat(),
            "node_types": list(self.valid_node_types.keys()),
            "relationship_types": list(self.valid_relationship_types.keys()),
            "compliance_level": "full",
            **graph_data.metadata
        }
    
    def _map_nodes_to_ontology(self, nodes: List[OntologyNode]) -> List[Dict[str, Any]]:
        """Map nodes to ontology format."""
        ontology_nodes = []
        
        for node in nodes:
            ontology_node = node.to_dict()
            
            # Ensure ontology compliance
            if node.type in self.valid_node_types:
                expected_props = self.valid_node_types[node.type]
                ontology_node = self._ensure_required_properties(ontology_node, expected_props)
            
            ontology_nodes.append(ontology_node)
        
        return ontology_nodes
    
    def _map_relationships_to_ontology(self, relationships: List[OntologyRelationship]) -> List[Dict[str, Any]]:
        """Map relationships to ontology format."""
        ontology_rels = []
        
        for rel in relationships:
            ontology_rel = rel.to_dict()
            
            # Ensure relationship type is valid
            if rel.type not in self.valid_relationship_types:
                logger.warning(f"Invalid relationship type: {rel.type}")
                ontology_rel["type"] = "flow"  # Default to flow
            
            ontology_rels.append(ontology_rel)
        
        return ontology_rels
    
    def _validate_graph_structure(self, graph_data: GraphData) -> Dict[str, Any]:
        """Validate graph structure for ontology compliance."""
        warnings = []
        errors = []
        
        # Check for orphaned nodes
        node_uuids = {node.uuid for node in graph_data.nodes}
        for rel in graph_data.relationships:
            if rel.source_uuid not in node_uuids:
                warnings.append(f"Relationship {rel.uuid} references missing source node {rel.source_uuid}")
            if rel.target_uuid not in node_uuids:
                warnings.append(f"Relationship {rel.uuid} references missing target node {rel.target_uuid}")
        
        # Check node type validity
        for node in graph_data.nodes:
            if node.type not in self.valid_node_types:
                warnings.append(f"Node {node.uuid} has invalid type: {node.type}")
        
        # Check relationship type validity
        for rel in graph_data.relationships:
            if rel.type not in self.valid_relationship_types:
                warnings.append(f"Relationship {rel.uuid} has invalid type: {rel.type}")
        
        return {
            "is_valid": len(errors) == 0,
            "warnings": warnings,
            "errors": errors,
            "node_count": len(graph_data.nodes),
            "relationship_count": len(graph_data.relationships)
        }
    
    def _generate_graph_statistics(self, graph_data: GraphData) -> Dict[str, Any]:
        """Generate comprehensive graph statistics."""
        stats = {
            "node_count": len(graph_data.nodes),
            "relationship_count": len(graph_data.relationships),
            "node_types": {},
            "relationship_types": {},
            "connectivity": {}
        }
        
        # Count nodes by type
        for node in graph_data.nodes:
            node_type = node.type
            stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1
        
        # Count relationships by type
        for rel in graph_data.relationships:
            rel_type = rel.type
            stats["relationship_types"][rel_type] = stats["relationship_types"].get(rel_type, 0) + 1
        
        # Calculate connectivity metrics
        node_degrees = {}
        for rel in graph_data.relationships:
            node_degrees[rel.source_uuid] = node_degrees.get(rel.source_uuid, 0) + 1
            node_degrees[rel.target_uuid] = node_degrees.get(rel.target_uuid, 0) + 1
        
        if node_degrees:
            stats["connectivity"] = {
                "max_degree": max(node_degrees.values()),
                "avg_degree": sum(node_degrees.values()) / len(node_degrees),
                "isolated_nodes": len(graph_data.nodes) - len(node_degrees)
            }
        
        return stats
    
    def _ensure_required_properties(self, node_dict: Dict[str, Any], 
                                  expected_props: List[str]) -> Dict[str, Any]:
        """Ensure node has all required properties."""
        for prop in expected_props:
            if prop not in node_dict:
                # Set default value based on property type
                if prop.endswith('Count'):
                    node_dict[prop] = 0
                elif prop.startswith('Is'):
                    node_dict[prop] = False
                elif prop.endswith('Number'):
                    node_dict[prop] = 0
                else:
                    node_dict[prop] = ""
        
        return node_dict
    
    def _flatten_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested properties for Neo4j compatibility."""
        flattened = {}
        
        for key, value in properties.items():
            if isinstance(value, (dict, list)):
                # Convert complex types to strings
                flattened[key] = str(value)
            elif value is None:
                flattened[key] = ""
            else:
                flattened[key] = value
        
        return flattened