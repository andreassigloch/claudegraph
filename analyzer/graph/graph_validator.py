#!/usr/bin/env python3
"""
Graph Validator for Code Architecture Analyzer

Validates graph structure, consistency, and quality metrics.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field

from .node_factory import OntologyNode
from .relationship_builder import OntologyRelationship
from .ontology_mapper import GraphData

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of graph validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class GraphValidator:
    """Validates graph structure and quality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize graph validator with configuration."""
        self.config = config or {}
        
        # Validation thresholds
        self.max_orphaned_nodes = self.config.get('validation', {}).get('max_orphaned_nodes', 5)
        self.max_node_degree = self.config.get('validation', {}).get('max_node_degree', 50)
        self.min_function_coverage = self.config.get('validation', {}).get('min_function_coverage', 0.8)
        
        # Required node types
        self.required_node_types = {'SYS', 'MOD', 'FUNC'}
        
        # Valid relationship types
        self.valid_relationship_types = {'compose', 'allocate', 'flow', 'call', 'use'}
    
    def validate_graph(self, graph_data: GraphData) -> ValidationResult:
        """Perform comprehensive graph validation."""
        logger.info("Starting comprehensive graph validation")
        
        result = ValidationResult(is_valid=True)
        
        # Run all validation checks
        self._validate_basic_structure(graph_data, result)
        self._validate_node_consistency(graph_data, result)
        self._validate_relationship_consistency(graph_data, result)
        self._validate_ontology_compliance(graph_data, result)
        self._validate_graph_connectivity(graph_data, result)
        self._validate_semantic_correctness(graph_data, result)
        self._calculate_quality_metrics(graph_data, result)
        
        # Determine overall validity
        result.is_valid = len(result.errors) == 0
        
        logger.info(f"Validation complete: {'PASSED' if result.is_valid else 'FAILED'} "
                   f"({len(result.errors)} errors, {len(result.warnings)} warnings)")
        
        return result
    
    def validate_node_structure(self, node: OntologyNode) -> ValidationResult:
        """Validate individual node structure."""
        result = ValidationResult(is_valid=True)
        
        # Check required fields
        if not node.uuid:
            result.errors.append("Node missing UUID")
        if not node.type:
            result.errors.append("Node missing type")
        if not node.name:
            result.errors.append("Node missing name")
        
        # Check type-specific requirements
        if node.type == 'SYS':
            self._validate_system_node(node, result)
        elif node.type == 'MOD':
            self._validate_module_node(node, result)
        elif node.type == 'FUNC':
            self._validate_function_node(node, result)
        elif node.type == 'ACTOR':
            self._validate_actor_node(node, result)
        
        result.is_valid = len(result.errors) == 0
        return result
    
    def validate_relationship_structure(self, relationship: OntologyRelationship) -> ValidationResult:
        """Validate individual relationship structure."""
        result = ValidationResult(is_valid=True)
        
        # Check required fields
        if not relationship.uuid:
            result.errors.append("Relationship missing UUID")
        if not relationship.type:
            result.errors.append("Relationship missing type")
        if not relationship.source_uuid:
            result.errors.append("Relationship missing source UUID")
        if not relationship.target_uuid:
            result.errors.append("Relationship missing target UUID")
        
        # Check relationship type validity
        if relationship.type not in self.valid_relationship_types:
            result.warnings.append(f"Unknown relationship type: {relationship.type}")
        
        # Check for self-referencing relationships
        if relationship.source_uuid == relationship.target_uuid:
            result.warnings.append("Self-referencing relationship detected")
        
        result.is_valid = len(result.errors) == 0
        return result
    
    def _validate_basic_structure(self, graph_data: GraphData, result: ValidationResult):
        """Validate basic graph structure."""
        if not graph_data.nodes:
            result.errors.append("Graph contains no nodes")
            return
        
        if not graph_data.relationships:
            result.warnings.append("Graph contains no relationships")
        
        # Check for required node types
        node_types = {node.type for node in graph_data.nodes}
        missing_types = self.required_node_types - node_types
        if missing_types:
            result.warnings.append(f"Missing required node types: {missing_types}")
        
        # Check for duplicate UUIDs
        node_uuids = [node.uuid for node in graph_data.nodes]
        if len(node_uuids) != len(set(node_uuids)):
            result.errors.append("Duplicate node UUIDs detected")
        
        rel_uuids = [rel.uuid for rel in graph_data.relationships]
        if len(rel_uuids) != len(set(rel_uuids)):
            result.errors.append("Duplicate relationship UUIDs detected")
    
    def _validate_node_consistency(self, graph_data: GraphData, result: ValidationResult):
        """Validate node consistency."""
        for node in graph_data.nodes:
            node_result = self.validate_node_structure(node)
            result.errors.extend(node_result.errors)
            result.warnings.extend(node_result.warnings)
    
    def _validate_relationship_consistency(self, graph_data: GraphData, result: ValidationResult):
        """Validate relationship consistency."""
        node_uuids = {node.uuid for node in graph_data.nodes}
        
        for rel in graph_data.relationships:
            rel_result = self.validate_relationship_structure(rel)
            result.errors.extend(rel_result.errors)
            result.warnings.extend(rel_result.warnings)
            
            # Check for dangling references
            if rel.source_uuid not in node_uuids:
                result.errors.append(f"Relationship {rel.uuid} references non-existent source node {rel.source_uuid}")
            if rel.target_uuid not in node_uuids:
                result.errors.append(f"Relationship {rel.uuid} references non-existent target node {rel.target_uuid}")
    
    def _validate_ontology_compliance(self, graph_data: GraphData, result: ValidationResult):
        """Validate ontology compliance."""
        # Check node type distribution
        node_types = {}
        for node in graph_data.nodes:
            node_types[node.type] = node_types.get(node.type, 0) + 1
        
        # Should have at least one system node
        if node_types.get('SYS', 0) == 0:
            result.warnings.append("No system nodes found")
        elif node_types.get('SYS', 0) > 1:
            result.warnings.append("Multiple system nodes found")
        
        # Should have module nodes if there are function nodes
        if node_types.get('FUNC', 0) > 0 and node_types.get('MOD', 0) == 0:
            result.warnings.append("Function nodes without module nodes")
        
        # Check relationship type distribution
        rel_types = {}
        for rel in graph_data.relationships:
            rel_types[rel.type] = rel_types.get(rel.type, 0) + 1
        
        # Should have compose relationships if there are system and module nodes
        if node_types.get('SYS', 0) > 0 and node_types.get('MOD', 0) > 0 and rel_types.get('compose', 0) == 0:
            result.warnings.append("Missing compose relationships between system and modules")
        
        # Should have allocate relationships if there are module and function nodes
        if node_types.get('MOD', 0) > 0 and node_types.get('FUNC', 0) > 0 and rel_types.get('allocate', 0) == 0:
            result.warnings.append("Missing allocate relationships between modules and functions")
    
    def _validate_graph_connectivity(self, graph_data: GraphData, result: ValidationResult):
        """Validate graph connectivity."""
        # Build adjacency information
        node_connections = {}
        for node in graph_data.nodes:
            node_connections[node.uuid] = {'in': 0, 'out': 0, 'total': 0}
        
        for rel in graph_data.relationships:
            if rel.source_uuid in node_connections:
                node_connections[rel.source_uuid]['out'] += 1
                node_connections[rel.source_uuid]['total'] += 1
            if rel.target_uuid in node_connections:
                node_connections[rel.target_uuid]['in'] += 1
                node_connections[rel.target_uuid]['total'] += 1
        
        # Check for orphaned nodes
        orphaned_nodes = [uuid for uuid, conn in node_connections.items() if conn['total'] == 0]
        if len(orphaned_nodes) > self.max_orphaned_nodes:
            result.warnings.append(f"High number of orphaned nodes: {len(orphaned_nodes)}")
        
        # Check for highly connected nodes (potential god objects)
        highly_connected = [uuid for uuid, conn in node_connections.items() 
                          if conn['total'] > self.max_node_degree]
        if highly_connected:
            result.warnings.append(f"Highly connected nodes detected: {len(highly_connected)}")
        
        # Store connectivity metrics
        if node_connections:
            total_connections = [conn['total'] for conn in node_connections.values()]
            result.metrics['connectivity'] = {
                'orphaned_nodes': len(orphaned_nodes),
                'highly_connected_nodes': len(highly_connected),
                'avg_node_degree': sum(total_connections) / len(total_connections),
                'max_node_degree': max(total_connections)
            }
    
    def _validate_semantic_correctness(self, graph_data: GraphData, result: ValidationResult):
        """Validate semantic correctness of the graph."""
        # Check for logical inconsistencies
        node_by_uuid = {node.uuid: node for node in graph_data.nodes}
        
        for rel in graph_data.relationships:
            source_node = node_by_uuid.get(rel.source_uuid)
            target_node = node_by_uuid.get(rel.target_uuid)
            
            if not source_node or not target_node:
                continue  # Already caught in relationship consistency check
            
            # Validate relationship semantics
            if rel.type == 'compose':
                if not (source_node.type == 'SYS' and target_node.type == 'MOD'):
                    result.warnings.append(f"Invalid compose relationship: {source_node.type} -> {target_node.type}")
            
            elif rel.type == 'allocate':
                if not (source_node.type == 'MOD' and target_node.type == 'FUNC'):
                    result.warnings.append(f"Invalid allocate relationship: {source_node.type} -> {target_node.type}")
            
            elif rel.type == 'flow':
                valid_flow_types = {
                    ('FUNC', 'FUNC'), ('ACTOR', 'FUNC'), ('FUNC', 'ACTOR')
                }
                if (source_node.type, target_node.type) not in valid_flow_types:
                    result.warnings.append(f"Invalid flow relationship: {source_node.type} -> {target_node.type}")
    
    def _calculate_quality_metrics(self, graph_data: GraphData, result: ValidationResult):
        """Calculate graph quality metrics."""
        metrics = result.metrics
        
        # Basic counts
        metrics['node_count'] = len(graph_data.nodes)
        metrics['relationship_count'] = len(graph_data.relationships)
        
        # Node type distribution
        node_types = {}
        for node in graph_data.nodes:
            node_types[node.type] = node_types.get(node.type, 0) + 1
        metrics['node_types'] = node_types
        
        # Relationship type distribution
        rel_types = {}
        for rel in graph_data.relationships:
            rel_types[rel.type] = rel_types.get(rel.type, 0) + 1
        metrics['relationship_types'] = rel_types
        
        # Calculate ratios
        if node_types.get('MOD', 0) > 0:
            metrics['functions_per_module'] = node_types.get('FUNC', 0) / node_types.get('MOD', 1)
        
        if len(graph_data.nodes) > 0:
            metrics['relationships_per_node'] = len(graph_data.relationships) / len(graph_data.nodes)
        
        # Quality scores (0-1)
        metrics['completeness_score'] = self._calculate_completeness_score(graph_data)
        metrics['consistency_score'] = self._calculate_consistency_score(result)
        metrics['connectivity_score'] = self._calculate_connectivity_score(graph_data)
    
    def _validate_system_node(self, node: OntologyNode, result: ValidationResult):
        """Validate system node specific requirements."""
        required_props = ['ProjectPath', 'TotalFiles', 'PythonFiles']
        for prop in required_props:
            if prop not in node.properties:
                result.warnings.append(f"System node missing property: {prop}")
    
    def _validate_module_node(self, node: OntologyNode, result: ValidationResult):
        """Validate module node specific requirements."""
        required_props = ['FilePath', 'ModuleName']
        for prop in required_props:
            if prop not in node.properties:
                result.warnings.append(f"Module node missing property: {prop}")
    
    def _validate_function_node(self, node: OntologyNode, result: ValidationResult):
        """Validate function node specific requirements."""
        required_props = ['FunctionName', 'ModuleName']
        for prop in required_props:
            if prop not in node.properties:
                result.warnings.append(f"Function node missing property: {prop}")
    
    def _validate_actor_node(self, node: OntologyNode, result: ValidationResult):
        """Validate actor node specific requirements."""
        required_props = ['ActorType', 'ActorName']
        for prop in required_props:
            if prop not in node.properties:
                result.warnings.append(f"Actor node missing property: {prop}")
    
    def _calculate_completeness_score(self, graph_data: GraphData) -> float:
        """Calculate how complete the graph structure is."""
        score = 0.0
        total_checks = 4.0
        
        # Check if required node types are present
        node_types = {node.type for node in graph_data.nodes}
        if 'SYS' in node_types:
            score += 0.25
        if 'MOD' in node_types:
            score += 0.25
        if 'FUNC' in node_types:
            score += 0.25
        
        # Check if relationships exist
        if graph_data.relationships:
            score += 0.25
        
        return score
    
    def _calculate_consistency_score(self, result: ValidationResult) -> float:
        """Calculate consistency score based on validation results."""
        total_issues = len(result.errors) + len(result.warnings)
        if total_issues == 0:
            return 1.0
        
        # Errors are weighted more heavily than warnings
        error_weight = 2
        weighted_issues = len(result.errors) * error_weight + len(result.warnings)
        
        # Score decreases with more issues (minimum 0.0)
        score = max(0.0, 1.0 - (weighted_issues / 100.0))
        return score
    
    def _calculate_connectivity_score(self, graph_data: GraphData) -> float:
        """Calculate connectivity quality score."""
        if not graph_data.nodes or not graph_data.relationships:
            return 0.0
        
        # Calculate the ratio of connected nodes
        connected_nodes = set()
        for rel in graph_data.relationships:
            connected_nodes.add(rel.source_uuid)
            connected_nodes.add(rel.target_uuid)
        
        connectivity_ratio = len(connected_nodes) / len(graph_data.nodes)
        return connectivity_ratio