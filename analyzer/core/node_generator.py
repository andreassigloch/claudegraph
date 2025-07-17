#!/usr/bin/env python3
"""
Node Generator for Code Architecture Analyzer (Refactored)

Orchestrates the creation of ontology-compliant graphs using specialized components.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .ast_parser import ASTParseResult
from .project_discoverer import ProjectStructure
from ..graph.node_factory import NodeFactory, OntologyNode
from ..graph.relationship_builder import RelationshipBuilder, OntologyRelationship
from ..graph.ontology_mapper import OntologyMapper, GraphData
from ..graph.graph_validator import GraphValidator

logger = logging.getLogger(__name__)


class NodeGenerator:
    """
    Orchestrates the generation of ontology-compliant graphs.
    
    This refactored version delegates specific responsibilities to specialized components:
    - NodeFactory: Creates individual nodes
    - RelationshipBuilder: Creates relationships between nodes
    - OntologyMapper: Maps to different formats
    - GraphValidator: Validates graph structure and quality
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_client=None):
        """Initialize the node generator with specialized components."""
        self.config = config or {}
        self.llm_client = llm_client
        
        # Initialize specialized components
        self.node_factory = NodeFactory(config)
        self.relationship_builder = RelationshipBuilder(config)
        self.ontology_mapper = OntologyMapper(config)
        self.graph_validator = GraphValidator(config)
        
        # Configuration options
        graph_config = self.config.get('graph', {})
        self.enable_flow_analysis = graph_config.get('enable_flow_analysis', True)
        self.enable_actor_flows = graph_config.get('enable_actor_flows', True)
        self.enable_function_flows = graph_config.get('enable_function_flows', True)
        self.enable_fchain_generation = graph_config.get('enable_fchain_generation', True)
        self.validate_output = graph_config.get('validate_output', True)
        
        logger.info("NodeGenerator initialized with specialized components")
    
    def generate_graph(self, project_structure: ProjectStructure, 
                      ast_results: List[ASTParseResult],
                      actor_results: Optional[List[Any]] = None,
                      flow_results: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Generate complete ontology graph from analysis results.
        
        This is the main entry point that orchestrates the entire graph generation process.
        """
        logger.info("Starting ontology graph generation")
        
        try:
            # Step 1: Create core nodes
            nodes = self._create_core_nodes(project_structure, ast_results)
            
            # Step 2: Create actor nodes if available
            if actor_results and self.enable_actor_flows:
                actor_nodes = self.node_factory.create_actor_nodes(actor_results, nodes)
                nodes.extend(actor_nodes)
            
            # Step 3: Create relationships
            relationships = self._create_relationships(
                project_structure, ast_results, nodes, actor_results, flow_results
            )
            
            # Step 4: Create FCHAIN nodes if enabled
            if self.enable_fchain_generation:
                fchain_nodes = self.relationship_builder.create_fchain_nodes_from_relationships(relationships)
                nodes.extend(fchain_nodes)
            
            # Step 5: Create graph data structure
            graph_data = GraphData(
                nodes=nodes,
                relationships=relationships,
                metadata=self._generate_metadata(project_structure, ast_results, actor_results)
            )
            
            # Step 6: Validate graph if enabled
            if self.validate_output:
                validation_result = self.graph_validator.validate_graph(graph_data)
                graph_data.metadata['validation'] = validation_result.__dict__
                
                if not validation_result.is_valid:
                    logger.warning(f"Graph validation failed with {len(validation_result.errors)} errors")
            
            # Step 7: Map to analysis result format
            result = self.ontology_mapper.map_to_analysis_result(graph_data, project_structure)
            
            logger.info(f"Graph generation completed: {len(nodes)} nodes, {len(relationships)} relationships")
            return result
            
        except Exception as e:
            logger.error(f"Graph generation failed: {e}", exc_info=True)
            raise
    
    def generate_neo4j_export(self, project_structure: ProjectStructure,
                             ast_results: List[ASTParseResult],
                             actor_results: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Generate graph data in Neo4j-compatible format."""
        logger.info("Generating Neo4j export format")
        
        # Generate standard graph
        result = self.generate_graph(project_structure, ast_results, actor_results)
        
        # Convert to Neo4j format
        graph_data = GraphData(
            nodes=[OntologyNode(**node) for node in result['graph']['nodes']],
            relationships=[OntologyRelationship(**rel) for rel in result['graph']['relationships']],
            metadata=result['metadata']
        )
        
        neo4j_result = self.ontology_mapper.map_to_neo4j_format(graph_data)
        return neo4j_result
    
    def _create_core_nodes(self, project_structure: ProjectStructure, 
                          ast_results: List[ASTParseResult]) -> List[OntologyNode]:
        """Create core nodes (system, modules, functions)."""
        nodes = []
        
        # Create system node
        system_node = self.node_factory.create_system_node(project_structure)
        nodes.append(system_node)
        
        # Create module and function nodes
        module_nodes = []
        function_nodes = []
        
        for ast_result in ast_results:
            # Create module node
            module_node = self.node_factory.create_module_node(ast_result, project_structure)
            module_nodes.append(module_node)
            nodes.append(module_node)
            
            # Create function nodes for this module
            for func_info in ast_result.functions:
                func_node = self.node_factory.create_function_node(func_info, ast_result, module_node)
                if func_node:
                    function_nodes.append(func_node)
                    nodes.append(func_node)
        
        logger.info(f"Created {len(nodes)} core nodes: 1 system, {len(module_nodes)} modules, {len(function_nodes)} functions")
        return nodes
    
    def _create_relationships(self, project_structure: ProjectStructure,
                            ast_results: List[ASTParseResult],
                            nodes: List[OntologyNode],
                            actor_results: Optional[List[Any]] = None,
                            flow_results: Optional[List[Any]] = None) -> List[OntologyRelationship]:
        """Create all relationships between nodes."""
        relationships = []
        
        # Separate nodes by type
        system_nodes = [n for n in nodes if n.type == 'SYS']
        module_nodes = [n for n in nodes if n.type == 'MOD']
        function_nodes = [n for n in nodes if n.type == 'FUNC']
        actor_nodes = [n for n in nodes if n.type == 'ACTOR']
        
        # Create compose relationships (system -> modules)
        if system_nodes and module_nodes:
            compose_rels = self.relationship_builder.create_compose_relationships(
                system_nodes[0], module_nodes
            )
            relationships.extend(compose_rels)
        
        # Create allocate relationships (modules -> functions)
        if module_nodes and function_nodes:
            allocate_rels = self.relationship_builder.create_allocate_relationships(
                module_nodes, function_nodes
            )
            relationships.extend(allocate_rels)
        
        # Create function flow relationships
        if self.enable_function_flows and function_nodes:
            func_flow_rels = self.relationship_builder.create_function_flow_relationships(
                ast_results, function_nodes
            )
            relationships.extend(func_flow_rels)
        
        # Create actor flow relationships
        if self.enable_actor_flows and actor_results and actor_nodes:
            actor_flow_rels = self.relationship_builder.create_flow_relationships(
                actor_results, function_nodes, actor_nodes
            )
            relationships.extend(actor_flow_rels)
        
        logger.info(f"Created {len(relationships)} relationships")
        return relationships
    
    def _generate_metadata(self, project_structure: ProjectStructure,
                          ast_results: List[ASTParseResult],
                          actor_results: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Generate metadata for the graph."""
        metadata = {
            "generator": "NodeGenerator",
            "version": "2.0",
            "project_path": str(project_structure.root_path),
            "files_analyzed": len(ast_results),
            "total_functions": sum(len(ast.functions) for ast in ast_results),
            "total_classes": sum(len(ast.classes) for ast in ast_results),
            "total_imports": sum(len(ast.imports) for ast in ast_results),
            "actor_detection_enabled": actor_results is not None,
            "flow_analysis_enabled": self.enable_flow_analysis,
            "components_used": {
                "node_factory": True,
                "relationship_builder": True,
                "ontology_mapper": True,
                "graph_validator": self.validate_output
            }
        }
        
        if actor_results:
            metadata["actor_results_count"] = len(actor_results)
        
        return metadata