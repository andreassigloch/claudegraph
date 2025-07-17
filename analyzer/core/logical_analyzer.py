#!/usr/bin/env python3
"""
Logical Analyzer for Code Architecture Analyzer

Redesigned analyzer that generates abstract logical dependency and information flow networks
instead of physical networks, while maintaining the same output data format.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .project_discoverer import ProjectDiscoverer, ProjectStructure
from .ast_parser import ASTParser, ASTParseResult
from .node_generator import NodeGenerator, GraphData, OntologyNode, OntologyRelationship
from .flow_analyzer import FlowDetector, FlowRelationship
from .logical_flow_detector import LogicalFlowDetector, LogicalActor, LogicalFlow
from .dead_code_detector import DeadCodeDetector
from ..llm.simple_bundler import SimpleEnhancementService, EnhancementRequest
from ..llm.client import LLMManager

logger = logging.getLogger(__name__)


@dataclass
class LogicalAnalysisStats:
    """Statistics from logical analysis"""
    files_discovered: int = 0
    files_parsed: int = 0
    functions_found: int = 0
    classes_found: int = 0
    
    # Logical analysis stats
    logical_actors: int = 0
    business_domains: int = 0
    logical_flows: int = 0
    data_transformations: int = 0
    
    # Business insights
    domain_services: int = 0
    business_entities: int = 0
    integration_points: int = 0
    workflow_orchestrators: int = 0
    
    # LLM usage stats
    actors_enhanced: int = 0
    llm_calls_made: int = 0
    enhancement_batches: int = 0
    
    # Node generation stats
    sys_nodes: int = 0
    mod_nodes: int = 0 
    func_nodes: int = 0
    actor_nodes: int = 0
    relationships: int = 0
    
    # Code quality stats
    dead_functions: int = 0
    isolated_functions: int = 0
    
    parse_errors: int = 0
    warnings: int = 0
    analysis_time_seconds: float = 0.0


@dataclass
class LogicalAnalysisResult:
    """Complete result from logical analysis"""
    project_structure: ProjectStructure
    ast_results: List[ASTParseResult] = field(default_factory=list)
    logical_actors: List[LogicalActor] = field(default_factory=list)
    logical_flows: List[LogicalFlow] = field(default_factory=list)
    business_domains: List[str] = field(default_factory=list)
    function_flows: List[FlowRelationship] = field(default_factory=list)
    graph_data: Optional[GraphData] = None
    dead_code_analysis: Optional[Any] = None
    stats: LogicalAnalysisStats = field(default_factory=LogicalAnalysisStats)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_successful(self) -> bool:
        """Check if analysis was successful"""
        return (
            self.project_structure.total_files > 0 and
            len(self.ast_results) > 0 and
            (len(self.logical_actors) > 0 or 
             (self.graph_data and len(self.graph_data.nodes) > 0))
        )


class LogicalAnalyzer:
    """Logical analyzer that generates abstract logical dependency networks"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_enabled = config.get('llm_provider', 'none') != 'none'
        
        # Initialize components
        self.project_discoverer = ProjectDiscoverer(config)
        self.ast_parser = ASTParser(config)
        self.flow_detector_engine = FlowDetector()
        self.logical_flow_detector = LogicalFlowDetector(config)
        self.dead_code_detector = DeadCodeDetector()
        
        # Force real UUIDs in NodeGenerator
        node_config = config.copy()
        node_config['node_generation'] = node_config.get('node_generation', {})
        node_config['node_generation']['deterministic_uuids'] = False
        self.node_generator = NodeGenerator(node_config)
        
        # Initialize LLM if enabled
        self.llm_manager = None
        self.enhancement_service = None
        if self.llm_enabled:
            try:
                self.llm_manager = LLMManager(config)
                self.enhancement_service = SimpleEnhancementService(self.llm_manager)
            except Exception as e:
                logger.warning(f"LLM initialization failed: {e}. Continuing without LLM.")
                self.llm_enabled = False
    
    def validate_project(self, project_path: str) -> Tuple[bool, List[str]]:
        """Validate project structure"""
        try:
            return self.project_discoverer.validate_project_path(project_path)
        except Exception as e:
            return False, [f"Project validation failed: {e}"]
    
    def analyze(self, project_path: str) -> LogicalAnalysisResult:
        """Main analysis method using logical approach"""
        
        start_time = time.time()
        result = LogicalAnalysisResult(
            project_structure=ProjectStructure(root_path=Path(project_path)),
            metadata={
                'analyzer_version': 'logical-1.0',
                'analysis_type': 'logical_dependencies',
                'analysis_timestamp': datetime.now().isoformat(),
                'project_path': str(project_path),
                'llm_enabled': self.llm_enabled
            }
        )
        
        try:
            # Phase 1: Project Discovery
            logger.info("Phase 1: Discovering project structure...")
            result.project_structure = self._discover_project(project_path, result)
            
            # Phase 2: AST Parsing
            logger.info("Phase 2: Parsing AST...")
            result.ast_results = self._parse_ast(result.project_structure, result)
            
            # Phase 3: Function Flow Detection (for logical flow mapping)
            logger.info("Phase 3: Detecting function flows...")
            result.function_flows = self._detect_function_flows(result.ast_results, result)
            
            # Phase 4: Logical Dependency Analysis
            logger.info("Phase 4: Analyzing logical dependencies...")
            result.logical_actors, result.logical_flows = self._analyze_logical_dependencies(
                result.ast_results, result.function_flows, result
            )
            
            # Phase 5: LLM Enhancement (optional)
            if self.llm_enabled:
                logger.info("Phase 5: Enhancing logical actors with LLM...")
                self._enhance_logical_actors(result.logical_actors, result)
            
            # Phase 6: Graph Generation
            logger.info("Phase 6: Generating logical dependency graph...")
            result.graph_data = self._generate_logical_graph(result)
            
            # Phase 7: Dead Code Detection
            logger.info("Phase 7: Detecting dead code...")
            self._detect_dead_code(result)
            
            # Phase 8: Business Domain Analysis
            logger.info("Phase 8: Analyzing business domains...")
            result.business_domains = self._analyze_business_domains(result)
            
            # Finalize
            result.stats.analysis_time_seconds = time.time() - start_time
            logger.info(f"Logical analysis completed in {result.stats.analysis_time_seconds:.2f}s")
            
        except Exception as e:
            logger.error(f"Logical analysis failed: {e}")
            result.errors.append(f"Logical analysis failed: {e}")
            result.stats.analysis_time_seconds = time.time() - start_time
        
        return result
    
    def _discover_project(self, project_path: str, result: LogicalAnalysisResult) -> ProjectStructure:
        """Discover project structure"""
        try:
            structure = self.project_discoverer.discover_project(project_path)
            result.stats.files_discovered = structure.total_files
            return structure
        except Exception as e:
            result.errors.append(f"Project discovery failed: {e}")
            return ProjectStructure(root_path=Path(project_path))
    
    def _parse_ast(self, project_structure: ProjectStructure, result: LogicalAnalysisResult) -> List[ASTParseResult]:
        """Parse AST for all Python files"""
        ast_results = []
        
        for file_info in project_structure.python_files:
            try:
                ast_result = self.ast_parser.parse_file(file_info.path)
                if ast_result:
                    ast_results.append(ast_result)
                    result.stats.functions_found += len(ast_result.functions)
                    result.stats.classes_found += len(ast_result.classes)
            except Exception as e:
                result.errors.append(f"AST parsing failed for {file_info.path}: {e}")
                result.stats.parse_errors += 1
        
        result.stats.files_parsed = len(ast_results)
        return ast_results
    
    def _detect_function_flows(self, ast_results: List[ASTParseResult], 
                             result: LogicalAnalysisResult) -> List[FlowRelationship]:
        """Detect function-to-function flows (for logical flow mapping)"""
        ast_data = {}
        
        for ast_result in ast_results:
            try:
                file_path = str(ast_result.file_path)
                
                ast_data[file_path] = {
                    'module_name': ast_result.module_name,
                    'functions': []
                }
                
                for func in ast_result.functions:
                    func_data = {
                        'name': func.name,
                        'full_name': func.full_name,
                        'line_number': getattr(func, 'line_number', 0),
                        'signature': getattr(func, 'signature', f"{func.name}()"),
                        'calls': func.calls,
                        'args': func.args,
                        'parameters': func.args,
                        'returns': func.returns,
                        'return_annotation': func.returns,
                        'decorators': [d.name for d in func.decorators],
                        'is_async': func.is_async,
                        'docstring': func.docstring
                    }
                    ast_data[file_path]['functions'].append(func_data)
                
            except Exception as e:
                result.warnings.append(f"Function flow detection failed for {ast_result.file_path}: {e}")
        
        try:
            flows = self.flow_detector_engine.analyze_flows(ast_data)
            result.stats.logical_flows = len(flows)
            return flows
        except Exception as e:
            result.warnings.append(f"Flow detection failed: {e}")
            return []
    
    def _analyze_logical_dependencies(self, ast_results: List[ASTParseResult], 
                                    function_flows: List[FlowRelationship],
                                    result: LogicalAnalysisResult) -> Tuple[List[LogicalActor], List[LogicalFlow]]:
        """Analyze logical dependencies using the logical flow detector"""
        all_logical_actors = []
        all_logical_flows = []
        
        for ast_result in ast_results:
            try:
                logical_actors, logical_flows = self.logical_flow_detector.analyze_logical_dependencies(
                    ast_result, function_flows
                )
                all_logical_actors.extend(logical_actors)
                all_logical_flows.extend(logical_flows)
            except Exception as e:
                result.warnings.append(f"Logical dependency analysis failed for {ast_result.file_path}: {e}")
        
        # Update statistics
        result.stats.logical_actors = len(all_logical_actors)
        result.stats.logical_flows = len(all_logical_flows)
        
        # Count by type
        for actor in all_logical_actors:
            if actor.logical_type.value == "domain_service":
                result.stats.domain_services += 1
            elif actor.logical_type.value == "business_entity":
                result.stats.business_entities += 1
            elif actor.logical_type.value == "integration_point":
                result.stats.integration_points += 1
            elif actor.logical_type.value == "workflow_orchestrator":
                result.stats.workflow_orchestrators += 1
        
        # Count data transformations
        result.stats.data_transformations = len([f for f in all_logical_flows if f.data_transformation != "data_transfer"])
        
        return all_logical_actors, all_logical_flows
    
    def _enhance_logical_actors(self, logical_actors: List[LogicalActor], result: LogicalAnalysisResult):
        """Enhance logical actors using LLM"""
        if not self.enhancement_service:
            return
        
        # Filter actors that need enhancement (low confidence or generic names)
        actors_to_enhance = [
            actor for actor in logical_actors 
            if actor.confidence < 0.8 or actor.name.endswith("Service")
        ]
        
        if not actors_to_enhance:
            logger.info("No logical actors need LLM enhancement")
            return
        
        try:
            logger.info(f"Enhancing {len(actors_to_enhance)} logical actors with LLM...")
            
            # Create enhancement requests for logical actors
            enhancement_requests = []
            for actor in actors_to_enhance:
                context = {
                    "business_domain": actor.business_domain,
                    "responsibilities": actor.responsibilities,
                    "data_entities": actor.data_entities,
                    "business_operations": actor.business_operations,
                    "logical_type": actor.logical_type.value
                }
                
                request = EnhancementRequest(
                    actor_name=actor.name,
                    context=context,
                    enhancement_type="logical_actor"
                )
                enhancement_requests.append(request)
            
            # Process enhancements
            enhancement_results = self.enhancement_service.enhance_logical_actors(enhancement_requests)
            
            # Apply enhancements
            for i, actor in enumerate(actors_to_enhance):
                if i < len(enhancement_results):
                    enhancement = enhancement_results[i]
                    actor.name = enhancement.enhanced_name
                    actor.business_domain = enhancement.business_domain if hasattr(enhancement, 'business_domain') else actor.business_domain
                    if hasattr(enhancement, 'description'):
                        actor.evidence["llm_description"] = enhancement.description
                    actor.evidence["enhanced"] = True
                    actor.confidence = min(actor.confidence + 0.2, 1.0)
            
            # Update stats
            result.stats.actors_enhanced = len(actors_to_enhance)
            result.stats.llm_calls_made = len(enhancement_requests)
            result.stats.enhancement_batches = 1
            
            logger.info(f"Enhanced {len(actors_to_enhance)} logical actors")
            
        except Exception as e:
            result.warnings.append(f"Logical actor enhancement failed: {e}")
            logger.warning(f"Logical actor enhancement failed: {e}")
    
    def _generate_logical_graph(self, result: LogicalAnalysisResult) -> GraphData:
        """Generate logical dependency graph"""
        try:
            # Generate base SYS/MOD/FUNC nodes
            analysis_result = self.node_generator.generate_graph(
                result.project_structure,
                result.ast_results
            )
            
            # Create GraphData from analysis result
            nodes = []
            for node_dict in analysis_result['graph']['nodes']:
                node_dict_fixed = node_dict.copy()
                if 'Name' in node_dict_fixed:
                    node_dict_fixed['name'] = node_dict_fixed.pop('Name')
                if 'Descr' in node_dict_fixed:
                    node_dict_fixed['descr'] = node_dict_fixed.pop('Descr')
                
                properties = {}
                for key, value in list(node_dict_fixed.items()):
                    if key not in ['uuid', 'type', 'name', 'descr']:
                        properties[key] = node_dict_fixed.pop(key)
                node_dict_fixed['properties'] = properties
                nodes.append(OntologyNode(**node_dict_fixed))
            
            relationships = []
            for rel_dict in analysis_result['graph']['relationships']:
                rel_dict_fixed = rel_dict.copy()
                if 'source' in rel_dict_fixed:
                    rel_dict_fixed['source_uuid'] = rel_dict_fixed.pop('source')
                if 'target' in rel_dict_fixed:
                    rel_dict_fixed['target_uuid'] = rel_dict_fixed.pop('target')
                
                properties = {}
                for key, value in list(rel_dict_fixed.items()):
                    if key not in ['uuid', 'type', 'source_uuid', 'target_uuid']:
                        properties[key] = rel_dict_fixed.pop(key)
                rel_dict_fixed['properties'] = properties
                relationships.append(OntologyRelationship(**rel_dict_fixed))
            
            graph_data = GraphData(
                nodes=nodes,
                relationships=relationships,
                metadata=analysis_result['metadata']
            )
            
            # Add logical actor nodes
            self._add_logical_actors_to_graph(graph_data, result.logical_actors)
            
            # Add logical flow relationships
            self._add_logical_flows_to_graph(graph_data, result.logical_flows, result.function_flows)
            
            # Update stats
            result.stats.sys_nodes = len([n for n in graph_data.nodes if n.type == 'SYS'])
            result.stats.mod_nodes = len([n for n in graph_data.nodes if n.type == 'MOD'])
            result.stats.func_nodes = len([n for n in graph_data.nodes if n.type == 'FUNC'])
            result.stats.actor_nodes = len([n for n in graph_data.nodes if n.type == 'ACTOR'])
            result.stats.relationships = len(graph_data.relationships)
            
            return graph_data
            
        except Exception as e:
            logger.error(f"Logical graph generation failed: {e}", exc_info=True)
            result.errors.append(f"Logical graph generation failed: {e}")
            return GraphData()
    
    def _add_logical_actors_to_graph(self, graph_data: GraphData, logical_actors: List[LogicalActor]):
        """Add logical actors as ACTOR nodes"""
        import uuid
        
        for actor in logical_actors:
            # Create logical actor node
            actor_node = OntologyNode(
                uuid=str(uuid.uuid4()),
                type="ACTOR",
                name=actor.name,
                descr=f"Logical {actor.logical_type.value} in {actor.business_domain} domain",
                properties={
                    "ActorType": actor.logical_type.value,
                    "BusinessDomain": actor.business_domain,
                    "Responsibilities": ", ".join(actor.responsibilities),
                    "DataEntities": ", ".join(actor.data_entities),
                    "BusinessOperations": ", ".join(actor.business_operations),
                    "AbstractionLevel": actor.abstraction_level,
                    "Confidence": actor.confidence,
                    "LogicalActor": True  # Mark as logical actor
                }
            )
            
            graph_data.nodes.append(actor_node)
    
    def _add_logical_flows_to_graph(self, graph_data: GraphData, logical_flows: List[LogicalFlow], 
                                   function_flows: List[FlowRelationship]):
        """Add logical flows as relationships"""
        import uuid
        
        # Create actor lookup
        actor_lookup = {node.name: node for node in graph_data.nodes if node.type == "ACTOR"}
        
        for logical_flow in logical_flows:
            source_actor = actor_lookup.get(logical_flow.source_actor)
            target_actor = actor_lookup.get(logical_flow.target_actor)
            
            if source_actor and target_actor:
                # Create logical flow relationship
                rel = OntologyRelationship(
                    uuid=str(uuid.uuid4()),
                    type="flow",
                    source_uuid=source_actor.uuid,
                    target_uuid=target_actor.uuid,
                    properties={
                        "FlowDescr": logical_flow.business_meaning,
                        "FlowDef": f"logical {logical_flow.flow_type}",
                        "FlowType": logical_flow.flow_type,
                        "DataTransformation": logical_flow.data_transformation,
                        "QualityAttributes": ", ".join(logical_flow.quality_attributes),
                        "Confidence": logical_flow.confidence,
                        "LogicalFlow": True  # Mark as logical flow
                    }
                )
                graph_data.relationships.append(rel)
        
        # Also add function-to-function flows for completeness
        func_lookup = {node.name: node for node in graph_data.nodes if node.type == "FUNC"}
        
        for flow in function_flows:
            source_func = func_lookup.get(flow.source_name)
            target_func = func_lookup.get(flow.target_name)
            
            if source_func and target_func:
                rel = OntologyRelationship(
                    uuid=str(uuid.uuid4()),
                    type="flow",
                    source_uuid=source_func.uuid,
                    target_uuid=target_func.uuid,
                    properties={
                        "FlowDescr": flow.flow_descr,
                        "FlowDef": flow.flow_def,
                        "Confidence": flow.confidence,
                        "LogicalFlow": False  # Mark as technical flow
                    }
                )
                graph_data.relationships.append(rel)
    
    def _detect_dead_code(self, result: LogicalAnalysisResult):
        """Detect unused functions with detailed analysis"""
        if not result.graph_data:
            return
        
        try:
            result.dead_code_analysis = self.dead_code_detector.analyze_dead_code(
                result.ast_results,
                result.graph_data.relationships,
                result.project_structure,
                enable_dynamic_detection=True
            )
            
            result.stats.dead_functions = result.dead_code_analysis.summary.get('total_dead', 0)
            result.stats.isolated_functions = 0
            
            if result.dead_code_analysis.dead_functions:
                dead_function_names = [f.full_name for f in result.dead_code_analysis.dead_functions]
                result.warnings.append(f"Dead code detected: {len(dead_function_names)} unused functions")
                logger.warning(f"Dead code detected: {len(dead_function_names)} functions")
            
        except Exception as e:
            logger.error(f"Dead code detection failed: {e}")
            result.errors.append(f"Dead code detection failed: {e}")
    
    def _analyze_business_domains(self, result: LogicalAnalysisResult) -> List[str]:
        """Analyze business domains from logical actors"""
        domains = set()
        
        for actor in result.logical_actors:
            domains.add(actor.business_domain)
        
        result.stats.business_domains = len(domains)
        return list(domains)
    
    def format_for_export(self, result: LogicalAnalysisResult) -> Dict[str, Any]:
        """Format logical analysis result for JSON export"""
        
        nodes = []
        relationships = []
        
        if result.graph_data:
            # Convert ontology nodes to dict format
            for node in result.graph_data.nodes:
                nodes.append({
                    "uuid": node.uuid,
                    "type": node.type,
                    "Name": node.name,
                    "Descr": node.descr,
                    **node.properties
                })
            
            # Convert relationships to dict format  
            for rel in result.graph_data.relationships:
                relationships.append({
                    "uuid": rel.uuid,
                    "type": rel.type,
                    "source": rel.source_uuid,
                    "target": rel.target_uuid,
                    **rel.properties
                })
        
        # Enhanced metadata with logical analysis information
        enhanced_metadata = {
            "analysis_version": "logical-1.0",
            "analysis_type": "logical_dependencies",
            "timestamp": result.metadata.get('analysis_timestamp'),
            "project_path": result.metadata.get('project_path'),
            "llm_enabled": result.metadata.get('llm_enabled', False),
            "logical_analysis_stats": {
                "files_discovered": result.stats.files_discovered,
                "files_parsed": result.stats.files_parsed,
                "functions_found": result.stats.functions_found,
                "classes_found": result.stats.classes_found,
                "logical_actors": result.stats.logical_actors,
                "business_domains": result.stats.business_domains,
                "logical_flows": result.stats.logical_flows,
                "data_transformations": result.stats.data_transformations,
                "domain_services": result.stats.domain_services,
                "business_entities": result.stats.business_entities,
                "integration_points": result.stats.integration_points,
                "workflow_orchestrators": result.stats.workflow_orchestrators,
                "actors_enhanced": result.stats.actors_enhanced,
                "llm_calls_made": result.stats.llm_calls_made,
                "dead_functions": result.stats.dead_functions,
                "analysis_time_seconds": result.stats.analysis_time_seconds,
                "total_nodes": len(nodes),
                "total_relationships": len(relationships)
            },
            "business_domains": result.business_domains
        }
        
        # Add logical actor breakdown
        logical_actor_breakdown = {}
        for actor in result.logical_actors:
            actor_type = actor.logical_type.value
            logical_actor_breakdown[actor_type] = logical_actor_breakdown.get(actor_type, 0) + 1
        enhanced_metadata["logical_actor_breakdown"] = logical_actor_breakdown
        
        # Add logical flow breakdown
        logical_flow_breakdown = {}
        for flow in result.logical_flows:
            flow_type = flow.flow_type
            logical_flow_breakdown[flow_type] = logical_flow_breakdown.get(flow_type, 0) + 1
        enhanced_metadata["logical_flow_breakdown"] = logical_flow_breakdown
        
        # Add node type breakdown
        node_type_counts = {}
        for node in nodes:
            node_type = node.get('type', 'UNKNOWN')
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
        enhanced_metadata["node_breakdown"] = node_type_counts
        
        # Add relationship type breakdown
        rel_type_counts = {}
        for rel in relationships:
            rel_type = rel.get('type', 'UNKNOWN')
            rel_type_counts[rel_type] = rel_type_counts.get(rel_type, 0) + 1
        enhanced_metadata["relationship_breakdown"] = rel_type_counts
        
        # Add project structure summary
        if result.project_structure:
            enhanced_metadata["project_structure"] = {
                "total_files": result.project_structure.total_files,
                "total_lines": result.project_structure.total_lines,
                "python_files": len(result.project_structure.python_files),
                "directories": len([f for f in result.project_structure.python_files 
                                  if f.path.is_dir()]) if hasattr(result.project_structure, 'python_files') else 0
            }
        
        # Build export data
        export_data = {
            "metadata": enhanced_metadata,
            "nodes": nodes,
            "relationships": relationships
        }
        
        # Add errors and warnings if any
        if result.errors:
            export_data["metadata"]["errors"] = result.errors
        if result.warnings:
            export_data["metadata"]["warnings"] = result.warnings
        
        return export_data