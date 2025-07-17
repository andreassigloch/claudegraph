#!/usr/bin/env python3
"""
Flow-Based Analyzer for Code Architecture Analyzer

Simplified analyzer using flow-based actor detection:
1. Deterministic MOD/FUNC generation
2. Deterministic flow detection  
3. Flow-based actor detection (trigger → func → receiver)
4. Simple LLM bundling for enhancement
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
from .flow_based_detector import FlowBasedActorDetector, ActorFlow, FlowChain, FlowDirection
from .content_deduplicator import ContentDeduplicator
from .dead_code_detector import DeadCodeDetector
from ..llm.simple_bundler import SimpleEnhancementService, EnhancementRequest
from ..llm.client import LLMManager

logger = logging.getLogger(__name__)


@dataclass
class FlowBasedAnalysisStats:
    """Statistics from flow-based analysis"""
    files_discovered: int = 0
    files_parsed: int = 0
    functions_found: int = 0
    classes_found: int = 0
    
    # Flow-based stats
    trigger_actors: int = 0
    receiver_actors: int = 0
    flow_chains: int = 0
    function_flows: int = 0
    
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
class FlowBasedAnalysisResult:
    """Complete result from flow-based analysis"""
    project_structure: ProjectStructure
    ast_results: List[ASTParseResult] = field(default_factory=list)
    flow_actors: List[ActorFlow] = field(default_factory=list) 
    flow_chains: List[FlowChain] = field(default_factory=list)
    function_flows: List[FlowRelationship] = field(default_factory=list)
    graph_data: Optional[GraphData] = None
    dead_code_analysis: Optional[Any] = None  # DeadCodeAnalysis from dead_code_detector
    stats: FlowBasedAnalysisStats = field(default_factory=FlowBasedAnalysisStats)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_successful(self) -> bool:
        """Check if analysis was successful"""
        return (
            self.project_structure.total_files > 0 and
            len(self.ast_results) > 0 and
            (len(self.flow_actors) > 0 or 
             (self.graph_data and len(self.graph_data.nodes) > 0))
        )


class FlowBasedAnalyzer:
    """Simplified analyzer using flow-based actor detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_enabled = config.get('llm_provider', 'none') != 'none'
        
        # Initialize components
        self.project_discoverer = ProjectDiscoverer(config)
        self.ast_parser = ASTParser(config)
        self.flow_detector_engine = FlowDetector()
        self.flow_detector = FlowBasedActorDetector()
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
    
    def analyze(self, project_path: str) -> FlowBasedAnalysisResult:
        """Main analysis method using flow-based approach"""
        
        start_time = time.time()
        result = FlowBasedAnalysisResult(
            project_structure=ProjectStructure(root_path=Path(project_path)),
            metadata={
                'analyzer_version': 'flow-based-1.0',
                'analysis_timestamp': datetime.now().isoformat(),
                'project_path': str(project_path),
                'llm_enabled': self.llm_enabled
            }
        )
        
        try:
            # Phase 1: Project Discovery (deterministic)
            logger.info("Phase 1: Discovering project structure...")
            result.project_structure = self._discover_project(project_path, result)
            
            # Phase 2: AST Parsing (deterministic) 
            logger.info("Phase 2: Parsing AST...")
            result.ast_results = self._parse_ast(result.project_structure, result)
            
            # Phase 3: Function Flow Detection (deterministic)
            logger.info("Phase 3: Detecting function flows...")
            result.function_flows = self._detect_function_flows(result.ast_results, result)
            
            # Phase 4: Flow-based Actor Detection (mostly deterministic)
            logger.info("Phase 4: Detecting flow-based actors...")
            result.flow_actors, result.flow_chains = self._detect_flow_actors(result.ast_results, result.function_flows, result)
            
            # Phase 5: LLM Enhancement (optional)
            if self.llm_enabled:
                logger.info("Phase 5: Enhancing actors with LLM...")
                self._enhance_actors(result.flow_actors, result)
            
            # Phase 6: Graph Generation (deterministic)
            logger.info("Phase 6: Generating final graph...")
            result.graph_data = self._generate_graph(result)
            
            # Phase 7: Dead Code Detection
            logger.info("Phase 7: Detecting dead code...")
            self._detect_dead_code(result)
            
            # Finalize
            result.stats.analysis_time_seconds = time.time() - start_time
            logger.info(f"Analysis completed in {result.stats.analysis_time_seconds:.2f}s")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            result.errors.append(f"Analysis failed: {e}")
            result.stats.analysis_time_seconds = time.time() - start_time
        
        return result
    
    def _discover_project(self, project_path: str, result: FlowBasedAnalysisResult) -> ProjectStructure:
        """Discover project structure"""
        try:
            structure = self.project_discoverer.discover_project(project_path)
            result.stats.files_discovered = structure.total_files
            return structure
        except Exception as e:
            result.errors.append(f"Project discovery failed: {e}")
            return ProjectStructure(root_path=Path(project_path))
    
    def _parse_ast(self, project_structure: ProjectStructure, result: FlowBasedAnalysisResult) -> List[ASTParseResult]:
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
    
    def _detect_function_flows(self, ast_results: List[ASTParseResult], result: FlowBasedAnalysisResult) -> List[FlowRelationship]:
        """Detect function-to-function flows"""
        # Convert AST results to the format expected by FlowDetector
        # FlowDetector expects: { 'file_path': { 'module_name': '...', 'functions': [...] } }
        ast_data = {}
        
        for ast_result in ast_results:
            try:
                file_path = str(ast_result.file_path)
                
                # Convert to the expected format
                ast_data[file_path] = {
                    'module_name': ast_result.module_name,
                    'functions': []
                }
                
                # Convert functions
                for func in ast_result.functions:
                    func_data = {
                        'name': func.name,
                        'full_name': func.full_name,
                        'line_number': getattr(func, 'line_number', 0),
                        'signature': getattr(func, 'signature', f"{func.name}()"),
                        'calls': func.calls,  # Use the calls directly from AST
                        # Include parameter and return information for FlowDef generation
                        'args': func.args,  # Function parameters
                        'parameters': func.args,  # Alias for flow analyzer compatibility
                        'returns': func.returns,  # Return type annotation
                        'return_annotation': func.returns,  # Alias for flow analyzer compatibility
                        'decorators': [d.name for d in func.decorators],
                        'is_async': func.is_async,
                        'docstring': func.docstring
                    }
                    ast_data[file_path]['functions'].append(func_data)
                
            except Exception as e:
                result.warnings.append(f"Flow detection failed for {ast_result.file_path}: {e}")
        
        # Analyze flows using FlowDetector with the properly formatted data
        try:
            flows = self.flow_detector_engine.analyze_flows(ast_data)
            result.stats.function_flows = len(flows)
            return flows
        except Exception as e:
            result.warnings.append(f"FlowDetector analysis failed: {e}")
            return []
    
    def _detect_flow_actors(self, ast_results: List[ASTParseResult], function_flows: List[FlowRelationship], 
                           result: FlowBasedAnalysisResult) -> Tuple[List, List[FlowChain]]:
        """Detect actors from flow entry/exit points"""
        all_actor_flows = []
        all_chains = []
        
        for ast_result in ast_results:
            try:
                actor_flows, chains = self.flow_detector.detect_actors(ast_result, function_flows)
                all_actor_flows.extend(actor_flows)
                all_chains.extend(chains)
            except Exception as e:
                result.warnings.append(f"Actor detection failed for {ast_result.file_path}: {e}")
        
        # Update stats based on flow directions
        trigger_flows = len([f for f in all_actor_flows if f.direction.value == "inbound"])
        receiver_flows = len([f for f in all_actor_flows if f.direction.value == "outbound"])
        result.stats.trigger_actors = trigger_flows
        result.stats.receiver_actors = receiver_flows
        result.stats.flow_chains = len(all_chains)
        
        return all_actor_flows, all_chains
    
    def _enhance_actors(self, actor_flows: List, result: FlowBasedAnalysisResult):
        """Enhance actors using simple LLM bundling"""
        if not self.enhancement_service:
            return
        
        # Filter actors that need enhancement
        actors_to_enhance = [actor for actor in actor_flows if actor.needs_llm_enhancement]
        
        if not actors_to_enhance:
            logger.info("No actors need LLM enhancement")
            return
        
        try:
            logger.info(f"Enhancing {len(actors_to_enhance)} actors with LLM...")
            
            # Use simple bundling service
            enhancement_results = self.enhancement_service.enhance_actor_list(actors_to_enhance)
            
            # Apply enhancements
            for i, actor in enumerate(actors_to_enhance):
                if i < len(enhancement_results):
                    enhancement = enhancement_results[i]
                    actor.name = enhancement.enhanced_name
                    # Add description as metadata
                    if not hasattr(actor, 'metadata'):
                        actor.metadata = {}
                    actor.metadata['description'] = enhancement.description
                    actor.metadata['enhanced'] = True
            
            # Update stats
            stats = self.enhancement_service.get_enhancement_stats()
            result.stats.actors_enhanced = len(actors_to_enhance)
            result.stats.llm_calls_made = stats.get('batch_calls', 0)
            result.stats.enhancement_batches = stats.get('batch_calls', 0)
            
            logger.info(f"Enhanced {len(actors_to_enhance)} actors in {stats.get('batch_calls', 0)} LLM calls")
            
        except Exception as e:
            result.warnings.append(f"Actor enhancement failed: {e}")
            logger.warning(f"Actor enhancement failed: {e}")
    
    def _generate_graph(self, result: FlowBasedAnalysisResult) -> GraphData:
        """Generate final ontology graph"""
        try:
            # Generate SYS/MOD/FUNC nodes (deterministic)
            analysis_result = self.node_generator.generate_graph(
                result.project_structure,
                result.ast_results
            )
            
            # Extract GraphData from analysis result
            from ..graph.node_factory import OntologyNode
            from ..graph.relationship_builder import OntologyRelationship
            from ..graph.ontology_mapper import GraphData
            
            logger.info(f"Analysis result has {len(analysis_result.get('graph', {}).get('nodes', []))} nodes and {len(analysis_result.get('graph', {}).get('relationships', []))} relationships")
            
            # Fix node dict structure to match OntologyNode constructor  
            nodes = []
            for node_dict in analysis_result['graph']['nodes']:
                node_dict_fixed = node_dict.copy()
                # Map 'Name' -> 'name' and 'Descr' -> 'descr'
                if 'Name' in node_dict_fixed:
                    node_dict_fixed['name'] = node_dict_fixed.pop('Name')
                if 'Descr' in node_dict_fixed:
                    node_dict_fixed['descr'] = node_dict_fixed.pop('Descr')
                # Separate properties from main fields
                properties = {}
                for key, value in list(node_dict_fixed.items()):
                    if key not in ['uuid', 'type', 'name', 'descr']:
                        properties[key] = node_dict_fixed.pop(key)
                node_dict_fixed['properties'] = properties
                nodes.append(OntologyNode(**node_dict_fixed))
            
            # Fix relationship dict structure to match OntologyRelationship constructor
            relationships = []
            for rel_dict in analysis_result['graph']['relationships']:
                rel_dict_fixed = rel_dict.copy()
                # Map 'source' -> 'source_uuid' and 'target' -> 'target_uuid'
                if 'source' in rel_dict_fixed:
                    rel_dict_fixed['source_uuid'] = rel_dict_fixed.pop('source')
                if 'target' in rel_dict_fixed:
                    rel_dict_fixed['target_uuid'] = rel_dict_fixed.pop('target')
                # Separate properties from main fields
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
            
            logger.info(f"Created GraphData with {len(graph_data.nodes)} nodes and {len(graph_data.relationships)} relationships")
            
            # Add flow-based actor nodes
            self._add_flow_actors_to_graph(graph_data, result.flow_actors)
            
            # Add flow relationships
            self._add_flow_relationships_to_graph(graph_data, result.function_flows, result.flow_chains, result)
            
            # Add return value flows for functions with return statements
            self._add_return_value_flows(graph_data, result.ast_results)
            
            # Remove content duplicates (same function/class processed multiple times)
            # Uses real UUIDs but removes nodes with identical content
            # Temporarily disabled due to compatibility issue with OntologyRelationship
            # deduplicator = ContentDeduplicator()
            # graph_data = deduplicator.deduplicate_by_content(graph_data)
            
            # Update stats
            result.stats.sys_nodes = len([n for n in graph_data.nodes if n.type == 'SYS'])
            result.stats.mod_nodes = len([n for n in graph_data.nodes if n.type == 'MOD'])
            result.stats.func_nodes = len([n for n in graph_data.nodes if n.type == 'FUNC'])
            result.stats.actor_nodes = len([n for n in graph_data.nodes if n.type == 'ACTOR'])
            result.stats.relationships = len(graph_data.relationships)
            
            return graph_data
            
        except Exception as e:
            logger.error(f"Graph generation failed: {e}", exc_info=True)
            result.errors.append(f"Graph generation failed: {e}")
            return GraphData()
    
    
    def _add_flow_actors_to_graph(self, graph_data: GraphData, actor_flows: List):
        """Add flow-based actors as ACTOR nodes from ActorFlow list"""
        existing_actor_names = set()
        
        # Collect existing actor names to avoid duplicates
        for node in graph_data.nodes:
            if node.type == "ACTOR":
                existing_actor_names.add(node.name)
        
        # Extract unique actors from flows
        unique_actors = {}
        for flow in actor_flows:
            if flow.actor_name not in unique_actors:
                unique_actors[flow.actor_name] = {
                    'name': flow.actor_name,
                    'actor_type': flow.actor_type,
                    'confidence': flow.confidence,
                    'needs_llm_enhancement': flow.needs_llm_enhancement,
                    'operations': [],
                    'code_context': flow.code_context  # Store code_context for description
                }
            unique_actors[flow.actor_name]['operations'].append(flow.operation)
        
        for actor_name, actor_data in unique_actors.items():
            # Skip if actor already exists
            if actor_name in existing_actor_names:
                continue
                
            # Generate real UUID 
            import uuid
            actor_uuid = str(uuid.uuid4())
            
            operations_str = ", ".join(set(actor_data['operations']))
            
            # Use optimized description for web_request actors
            if actor_data['actor_type'] == 'web_request':
                descr = actor_data.get('code_context', '/')  # Just the path
            else:
                descr = f"{actor_data['actor_type']} - operations: {operations_str}"
            
            actor_node = OntologyNode(
                uuid=actor_uuid,
                type="ACTOR",
                name=actor_name,
                descr=descr,
                properties={
                    "ActorType": actor_data['actor_type'],
                    "Operations": operations_str,
                    "Confidence": actor_data['confidence']
                }
            )
            
            graph_data.nodes.append(actor_node)
            existing_actor_names.add(actor_name)
    
    def _add_flow_relationships_to_graph(self, graph_data: GraphData, function_flows: List[FlowRelationship], 
                                        flow_chains: List[FlowChain], result):
        """Add flow relationships to graph"""
        
        # Create comprehensive lookup map for function nodes by name
        func_lookup = {}
        func_nodes_by_simple_name = {}  # Track multiple nodes with same simple name
        
        for node in graph_data.nodes:
            if node.type == 'FUNC':
                # Primary lookup: exact node name (now clean names without prefixes)
                func_lookup[node.name] = node
                
                # Track simple names for collision detection
                if node.name not in func_nodes_by_simple_name:
                    func_nodes_by_simple_name[node.name] = []
                func_nodes_by_simple_name[node.name].append(node)
                
                # Secondary lookup: description (often contains full_name)
                if node.descr and node.descr != node.name:
                    func_lookup[node.descr] = node
        
        # Add function-to-function flows - use "flow" type as required by ontology
        flows_added = 0
        flows_skipped = 0
        
        for flow in function_flows:
            # Map deterministic UUIDs to real node UUIDs
            source_node = None
            target_node = None
            
            # Enhanced source node lookup with multiple strategies
            source_lookups = [
                getattr(flow, 'source_name', ''),
                getattr(flow, 'source_function', ''),
                flow.source_name.split('.')[-1] if hasattr(flow, 'source_name') and '.' in flow.source_name else '',
                flow.source_name.replace('.', '_') if hasattr(flow, 'source_name') else ''
            ]
            # Remove empty strings
            source_lookups = [s for s in source_lookups if s]
            
            for lookup_key in source_lookups:
                if lookup_key in func_lookup:
                    source_node = func_lookup[lookup_key]
                    break
            
            # Enhanced target node lookup with multiple strategies  
            target_lookups = [
                getattr(flow, 'target_name', ''),
                getattr(flow, 'target_function', ''),
                flow.target_name.split('.')[-1] if hasattr(flow, 'target_name') and '.' in flow.target_name else '',
                flow.target_name.replace('.', '_') if hasattr(flow, 'target_name') else ''
            ]
            # Remove empty strings
            target_lookups = [s for s in target_lookups if s]
            
            for lookup_key in target_lookups:
                if lookup_key in func_lookup:
                    target_node = func_lookup[lookup_key]
                    break
            
            # Only create relationship if both nodes exist
            if source_node and target_node:
                import uuid
                rel = OntologyRelationship(
                    uuid=str(uuid.uuid4()),
                    type="flow",
                    source_uuid=source_node.uuid,
                    target_uuid=target_node.uuid,
                    properties={
                        "FlowDescr": flow.flow_descr,
                        "FlowDef": flow.flow_def,
                        "Confidence": flow.confidence,
                        "source_name": getattr(flow, 'source_name', ''),
                        "target_name": getattr(flow, 'target_name', '')
                    }
                )
                graph_data.relationships.append(rel)
                flows_added += 1
            else:
                flows_skipped += 1
                # Log first few skipped flows for debugging
                if flows_skipped <= 5:
                    src_name = getattr(flow, 'source_name', 'Unknown')
                    tgt_name = getattr(flow, 'target_name', 'Unknown')
                    logger.warning(f"Skipped flow: {src_name} -> {tgt_name} (nodes not found)")
        
        logger.info(f"Flow relationships: {flows_added} added, {flows_skipped} skipped from {len(function_flows)} total")
        
        # Add bidirectional actor ↔ function relationships for ALL detected actor flows
        # (not just those in chains, since chain building might miss some due to name mismatches)
        all_actor_flows = []
        for chain in flow_chains:
            all_actor_flows.extend(chain.actor_flows)
        
        # Also add any actor flows that weren't included in chains (from the original detection)
        if hasattr(result, 'flow_actors') and result.flow_actors:
            all_actor_flows.extend(result.flow_actors)
        
        # Remove duplicates
        unique_flows = {}
        for flow in all_actor_flows:
            key = (flow.actor_name, flow.direction.value, flow.function_name, flow.operation)
            unique_flows[key] = flow
        
        for actor_flow in unique_flows.values():
            # Find actor node
            actor_node = None
            for node in graph_data.nodes:
                if (node.type == "ACTOR" and 
                    node.name == actor_flow.actor_name):
                    actor_node = node
                    break
            
            # Find function node - try multiple approaches
            func_node = None
            # 1. Direct match with clean function name
            for node in graph_data.nodes:
                if (node.type == "FUNC" and 
                    node.name == actor_flow.function_name):
                    func_node = node
                    break
            
            # 2. If no direct match, try pattern matching
            if not func_node:
                for node in graph_data.nodes:
                    if node.type == "FUNC":
                        # Extract method name (e.g., "get_data" from both "client.get_data" and "HttpClient.get_data")
                        flow_method = actor_flow.function_name.split('.')[-1] if '.' in actor_flow.function_name else actor_flow.function_name
                        # Compare with clean node name
                        if flow_method == node.name:
                            func_node = node
                            break
            
            # Create directional flow relationship
            if actor_node and func_node:
                import uuid
                
                # Determine source and target based on flow direction
                if actor_flow.direction.value == "inbound":  # Actor → Function
                    source_uuid = actor_node.uuid
                    target_uuid = func_node.uuid
                    flow_descr = f"{actor_flow.actor_name} {actor_flow.operation} to {func_node.name}"
                    flow_def = f"{actor_flow.actor_type} → function"
                else:  # FlowDirection.OUTBOUND: Function → Actor
                    source_uuid = func_node.uuid
                    target_uuid = actor_node.uuid
                    flow_descr = f"{func_node.name} {actor_flow.operation} to {actor_flow.actor_name}"
                    flow_def = f"function → {actor_flow.actor_type}"
                
                rel = OntologyRelationship(
                    uuid=str(uuid.uuid4()),
                    type="flow",
                    source_uuid=source_uuid,
                    target_uuid=target_uuid,
                    properties={
                        "FlowDescr": flow_descr,
                        "FlowDef": flow_def,
                        "direction": actor_flow.direction.value,
                        "operation": actor_flow.operation,
                        "actor_type": actor_flow.actor_type,
                        "Confidence": actor_flow.confidence
                    }
                )
                graph_data.relationships.append(rel)
            else:
                # Log missing connections for debugging
                if not actor_node:
                    logger.warning(f"Actor node not found: {actor_flow.actor_name}")
                if not func_node:
                    logger.warning(f"Function node not found: {actor_flow.function_name}")
    
    def _add_return_value_flows(self, graph_data: GraphData, ast_results: List[ASTParseResult]):
        """Skip adding virtual return value flows - return values are internal data flow, not external actors"""
        # Removed virtual ReturnValue actor creation - functions returning values
        # do not interact with external actors, this is internal data flow
        pass
    
    def _detect_dead_code(self, result: FlowBasedAnalysisResult):
        """Detect unused functions with detailed analysis"""
        if not result.graph_data:
            return
        
        try:
            # Perform enhanced dead code analysis with dynamic pattern detection
            result.dead_code_analysis = self.dead_code_detector.analyze_dead_code(
                result.ast_results,
                result.graph_data.relationships,
                result.project_structure,
                enable_dynamic_detection=True  # Enable dynamic pattern detection
            )
            
            # Update stats from detailed analysis
            result.stats.dead_functions = result.dead_code_analysis.summary.get('total_dead', 0)
            result.stats.isolated_functions = 0  # Will be tracked separately in detailed analysis
            
            # Add warnings for dead code (maintain backward compatibility)
            if result.dead_code_analysis.dead_functions:
                dead_function_names = [f.full_name for f in result.dead_code_analysis.dead_functions]
                result.warnings.append(f"Dead code detected: {len(dead_function_names)} unused functions: {', '.join(dead_function_names)}")
                logger.warning(f"Dead code detected: {len(dead_function_names)} functions")
            
            logger.info(f"Dead code analysis completed: {result.stats.dead_functions} dead functions, "
                       f"{len(result.dead_code_analysis.duplicates)} duplicates, "
                       f"{len(result.dead_code_analysis.orphaned)} orphaned")
                       
        except Exception as e:
            logger.error(f"Dead code detection failed: {e}")
            result.errors.append(f"Dead code detection failed: {e}")
            
            # Fallback to simple detection
            self._simple_dead_code_detection(result)
    
    def _simple_dead_code_detection(self, result: FlowBasedAnalysisResult):
        """Fallback simple dead code detection if enhanced detection fails"""
        func_nodes = [n for n in result.graph_data.nodes if n.type == 'FUNC']
        flow_rels = [r for r in result.graph_data.relationships if r.type == 'flow']
        
        dead_functions = []
        
        for func in func_nodes:
            func_uuid = func.uuid
            func_name = func.name
            
            # Count incoming and outgoing flows
            incoming = [r for r in flow_rels if r.target_uuid == func_uuid]
            outgoing = [r for r in flow_rels if r.source_uuid == func_uuid]
            
            total_connections = len(incoming) + len(outgoing)
            
            if total_connections == 0:
                dead_functions.append(func_name)
        
        # Update stats
        result.stats.dead_functions = len(dead_functions)
        
        # Add warnings for dead code
        if dead_functions:
            result.warnings.append(f"Dead code detected: {len(dead_functions)} unused functions: {', '.join(dead_functions)}")
            logger.warning(f"Dead code detected: {dead_functions}")
    
    def format_for_export(self, result: FlowBasedAnalysisResult) -> Dict[str, Any]:
        """Format analysis result for JSON export"""
        
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
        
        # Enhanced metadata collection for Phase 2.2
        enhanced_metadata = {
            "analysis_version": "flow-based-1.0",
            "timestamp": result.metadata.get('analysis_timestamp'),
            "project_path": result.metadata.get('project_path'),
            "llm_enabled": result.metadata.get('llm_enabled', False),
            "analysis_stats": {
                "files_discovered": result.stats.files_discovered,
                "files_parsed": result.stats.files_parsed,
                "functions_found": result.stats.functions_found,
                "classes_found": result.stats.classes_found,
                "trigger_actors": result.stats.trigger_actors,
                "receiver_actors": result.stats.receiver_actors,
                "flow_chains": result.stats.flow_chains,
                "function_flows": result.stats.function_flows,
                "actors_enhanced": result.stats.actors_enhanced,
                "llm_calls_made": result.stats.llm_calls_made,
                "dead_functions": result.stats.dead_functions,
                "isolated_functions": result.stats.isolated_functions,
                "analysis_time_seconds": result.stats.analysis_time_seconds,
                "total_nodes": len(nodes),
                "total_relationships": len(relationships)
            }
        }
        
        # Add detailed dead code analysis if available
        if result.dead_code_analysis:
            enhanced_metadata["dead_code_analysis"] = {
                "total_functions_analyzed": result.dead_code_analysis.total_functions,
                "dead_functions_count": len(result.dead_code_analysis.dead_functions),
                "duplicates_count": len(result.dead_code_analysis.duplicates),
                "orphaned_count": len(result.dead_code_analysis.orphaned),
                "unreachable_count": len(result.dead_code_analysis.unreachable),
                "summary": result.dead_code_analysis.summary,
                "by_type": {
                    "duplicates": [{"name": f.name, "module": f.module, "reason": f.reason} 
                                 for f in result.dead_code_analysis.duplicates[:10]],  # Top 10
                    "orphaned": [{"name": f.name, "module": f.module, "reason": f.reason} 
                               for f in result.dead_code_analysis.orphaned[:10]],  # Top 10
                    "unreachable": [{"name": f.name, "module": f.module, "reason": f.reason} 
                                  for f in result.dead_code_analysis.unreachable[:10]]  # Top 10
                }
            }
        
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