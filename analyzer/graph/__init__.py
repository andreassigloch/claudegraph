#!/usr/bin/env python3
"""
Graph Module for Code Architecture Analyzer

This module provides specialized components for graph generation:
- NodeFactory: Creates ontology-compliant nodes
- RelationshipBuilder: Creates relationships between nodes  
- OntologyMapper: Maps graphs to different formats
- GraphValidator: Validates graph structure and quality

Legacy components:
- OntologyGraphBuilder: Original graph builder (preserved for compatibility)
"""

# New modular components
from .node_factory import NodeFactory, OntologyNode
from .relationship_builder import RelationshipBuilder, OntologyRelationship
from .ontology_mapper import OntologyMapper, GraphData
from .graph_validator import GraphValidator, ValidationResult

# Legacy components (preserved for compatibility)
from .builder import (
    OntologyGraphBuilder,
    GraphBuildResult,
    GraphNode,
    GraphRelationship
)

__all__ = [
    # New modular components
    'NodeFactory',
    'OntologyNode', 
    'RelationshipBuilder',
    'OntologyRelationship',
    'OntologyMapper',
    'GraphData',
    'GraphValidator',
    'ValidationResult',
    # Legacy components
    'OntologyGraphBuilder',
    'GraphBuildResult', 
    'GraphNode',
    'GraphRelationship'
]