#!/usr/bin/env python3
"""
Node Generator for Code Architecture Analyzer

Converts AST parsing results into ontology-compliant nodes and relationships
according to the defined schema.
"""

import uuid
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

from .ast_parser import ASTParseResult, FunctionInfo, ClassInfo
from .project_discoverer import ProjectStructure, ProjectFile

try:
    from ..detection.pattern_matcher import ActorDetectionResult, DetectionMatch
except ImportError:
    # During development, detection module might not be available
    ActorDetectionResult = None
    DetectionMatch = None


logger = logging.getLogger(__name__)


@dataclass
class OntologyNode:
    """Represents a node in the ontology graph."""
    uuid: str
    type: str
    name: str
    descr: str
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary format."""
        result = {
            "uuid": self.uuid,
            "type": self.type,
            "Name": self.name,
            "Descr": self.descr
        }
        result.update(self.properties)
        return result


@dataclass
class OntologyRelationship:
    """Represents a relationship in the ontology graph."""
    uuid: str
    type: str
    source: str  # Source node UUID
    target: str  # Target node UUID
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary format."""
        result = {
            "uuid": self.uuid,
            "type": self.type,
            "source": self.source,
            "target": self.target
        }
        result.update(self.properties)
        return result


@dataclass
class GraphData:
    """Complete graph data structure."""
    nodes: List[OntologyNode] = field(default_factory=list)
    relationships: List[OntologyRelationship] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class NodeGenerator:
    """
    Generates ontology-compliant nodes and relationships from AST parsing results.
    
    Converts project structure and AST data into SYS, MOD, and FUNC nodes
    with proper relationships according to the ontology schema.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_client=None):
        """Initialize node generator with configuration."""
        self.config = config or {}
        
        # Graph configuration
        graph_config = self.config.get('graph', {})
        node_config = graph_config.get('nodes', {})
        
        self.max_name_length = node_config.get('max_name_length', 25)
        self.naming_style = node_config.get('naming_style', 'PascalCase')
        self.uuid_prefix = node_config.get('uuid_prefix', '')
        
        # Node filtering
        self.include_private_functions = node_config.get('include_private_functions', True)
        self.include_test_functions = node_config.get('include_test_functions', False)
        self.min_function_lines = node_config.get('min_function_lines', 1)
        
        # Flow analysis configuration
        flow_config = self.config.get('flow_analysis', {})
        self.enable_flow_analysis = flow_config.get('enabled', True)
        self.llm_enhanced_flows = flow_config.get('llm_enhanced_descriptions', True)
        self.max_flow_description_length = flow_config.get('max_description_length', 200)
        
        # Initialize flow analysis components
        if self.enable_flow_analysis:
            from .flow_analyzer import FlowDetector, FlowEnhancer
            self.flow_detector = FlowDetector()
            self.flow_enhancer = FlowEnhancer(
                llm_client=llm_client, 
                enabled=self.llm_enhanced_flows and llm_client is not None
            )
        else:
            self.flow_detector = None
            self.flow_enhancer = None
        
        # Internal state
        self._node_cache = {}  # For deterministic UUID generation
        self._name_cache = set()  # For name uniqueness
        self._uuid_counter = 0
        
        logger.debug("Node generator initialized with flow analysis: %s", self.enable_flow_analysis)
    
    def generate_graph(self, project_structure: ProjectStructure, 
                      ast_results: List[ASTParseResult],
                      actor_results: Optional[List[Any]] = None) -> GraphData:
        """
        Generate complete ontology graph from project structure and AST results.
        
        Args:
            project_structure: Discovered project structure
            ast_results: List of AST parsing results
            
        Returns:
            GraphData with nodes and relationships
        """
        graph = GraphData()
        
        try:
            # Generate SYS node for the project
            sys_node = self._generate_system_node(project_structure)
            graph.nodes.append(sys_node)
            
            # Generate MOD nodes for each file and FUNC nodes for functions
            mod_nodes = []
            func_nodes = []
            actor_nodes = []
            
            for ast_result in ast_results:
                # Generate MOD node for the file
                mod_node = self._generate_module_node(ast_result, project_structure)
                mod_nodes.append(mod_node)
                graph.nodes.append(mod_node)
                
                # Generate compose relationship: SYS -> MOD
                compose_rel = self._generate_relationship(
                    'compose', sys_node.uuid, mod_node.uuid,
                    f"System contains module {mod_node.name}"
                )
                graph.relationships.append(compose_rel)
                
                # Generate FUNC nodes for functions (handling endpoints properly)
                # Only process standalone functions here, not methods (methods are processed separately below)
                for func_info in ast_result.functions:
                    if self._should_include_function(func_info) and not func_info.is_method:
                        # Always create FUNC node (no special endpoint handling)
                        func_node = self._generate_function_node(func_info, ast_result)
                        func_nodes.append(func_node)
                        graph.nodes.append(func_node)
                        
                        # Generate allocate relationship: MOD -> FUNC
                        allocate_rel = self._generate_relationship(
                            'allocate', mod_node.uuid, func_node.uuid,
                            f"Module allocates function {func_node.name}"
                        )
                        graph.relationships.append(allocate_rel)
                
                # Generate FUNC nodes for class methods (handling endpoints properly)
                for class_info in ast_result.classes:
                    for method_info in class_info.methods:
                        if self._should_include_function(method_info):
                            # Always create FUNC node for methods
                            func_node = self._generate_function_node(method_info, ast_result, class_info)
                            func_nodes.append(func_node)
                            graph.nodes.append(func_node)
                            
                            # Generate allocate relationship: MOD -> FUNC
                            allocate_rel = self._generate_relationship(
                                'allocate', mod_node.uuid, func_node.uuid,
                                f"Module allocates method {func_node.name}"
                            )
                            graph.relationships.append(allocate_rel)
            
            # Generate ACTOR nodes from actor detection results
            if actor_results and ActorDetectionResult is not None:
                actor_nodes = self._generate_actor_nodes(actor_results, func_nodes)
                graph.nodes.extend(actor_nodes)
                
                # Generate flow relationships between actors and functions
                actor_flow_relationships = self._generate_flow_relationships(actor_results, func_nodes, actor_nodes)
                graph.relationships.extend(actor_flow_relationships)
            
            # Skip function-to-function flow relationships in NodeGenerator
            # These are now handled in FlowBasedAnalyzer to avoid UUID conflicts
            if self.enable_flow_analysis and False:  # Disabled - handled in FlowBasedAnalyzer
                function_flow_relationships = self._generate_function_flow_relationships(ast_results, func_nodes)
                graph.relationships.extend(function_flow_relationships)
                
                # Generate function-to-actor flow relationships (RECEIVER flows)
                if actor_results and actor_nodes:
                    func_actor_relationships = self._generate_function_actor_flow_relationships(ast_results, func_nodes, actor_nodes)
                    graph.relationships.extend(func_actor_relationships)
                
                # Generate actor-to-function flow relationships (TRIGGER flows)
                trigger_relationships, new_actors = self._generate_trigger_flow_relationships(actor_nodes, func_nodes)
                graph.relationships.extend(trigger_relationships)
                graph.nodes.extend(new_actors)
                
                # Generate FCHAIN nodes from complete flow pattern
                all_flow_relationships = function_flow_relationships + trigger_relationships
                fchain_nodes = self._generate_fchain_nodes(all_flow_relationships)
                graph.nodes.extend(fchain_nodes)
            
            # Generate metadata
            graph.metadata = self._generate_graph_metadata(
                project_structure, ast_results, graph
            )
            
            total_actors = len(actor_nodes) if actor_nodes else 0
            logger.info(f"Generated graph: {len(graph.nodes)} nodes ({total_actors} actors), {len(graph.relationships)} relationships")
            
        except Exception as e:
            error_msg = f"Error generating graph: {e}"
            logger.error(error_msg)
            graph.errors.append(error_msg)
        
        return graph
    
    def _generate_system_node(self, project_structure: ProjectStructure) -> OntologyNode:
        """Generate SYS node for the project."""
        project_name = project_structure.root_path.name
        normalized_name = self._normalize_name(project_name)
        
        descr = f"Python project with {project_structure.total_files} modules and {project_structure.total_lines} lines of code"
        
        return OntologyNode(
            uuid=self._generate_uuid('SYS', normalized_name),
            type='SYS',
            name=normalized_name,
            descr=descr,
            properties={
                'Name': normalized_name,
                'Descr': descr
            },
            metadata={
                'original_name': project_name,
                'path': str(project_structure.root_path),
                'total_files': project_structure.total_files,
                'total_lines': project_structure.total_lines,
                'total_size_bytes': project_structure.total_size_bytes
            }
        )
    
    def _generate_module_node(self, ast_result: ASTParseResult, 
                             project_structure: ProjectStructure) -> OntologyNode:
        """Generate MOD node for a Python file."""
        module_name = ast_result.module_name or ast_result.file_path.stem
        # Use original module name for MOD nodes - no normalization
        original_name = module_name
        
        # Find corresponding project file for metadata
        project_file = None
        for pf in project_structure.python_files:
            if pf.path == ast_result.file_path:
                project_file = pf
                break
        
        descr = f"Python module with {len(ast_result.functions)} functions"
        if len(ast_result.classes) > 0:
            descr += f" and {len(ast_result.classes)} classes"
        
        if ast_result.docstring:
            # Use first sentence of docstring if available
            first_sentence = ast_result.docstring.split('.')[0].strip()
            if first_sentence and len(first_sentence) < 100:
                descr = first_sentence
        
        properties = {
            'Name': original_name,
            'Descr': descr
        }
        
        metadata = {
            'original_name': module_name,
            'file_path': str(ast_result.file_path),
            'relative_path': str(ast_result.file_path.relative_to(project_structure.root_path)),
            'function_count': len(ast_result.functions),
            'class_count': len(ast_result.classes),
            'import_count': len(ast_result.imports),
            'has_docstring': bool(ast_result.docstring)
        }
        
        if project_file:
            metadata.update({
                'line_count': project_file.line_count,
                'size_bytes': project_file.size_bytes,
                'is_test': project_file.is_test,
                'is_main': project_file.is_main
            })
        
        return OntologyNode(
            uuid=self._generate_uuid('MOD', original_name),
            type='MOD',
            name=original_name,
            descr=descr,
            properties=properties,
            metadata=metadata
        )
    
    def _generate_function_node(self, func_info: FunctionInfo, ast_result: ASTParseResult,
                               class_info: Optional[ClassInfo] = None) -> OntologyNode:
        """Generate FUNC node for a function or method."""
        if class_info:
            # Method: include class name
            func_name = f"{class_info.name}.{func_info.name}"
        else:
            # Standalone function: include module name for uniqueness
            module_name = ast_result.module_name or Path(ast_result.file_path).stem
            func_name = f"{module_name}.{func_info.name}"
        
        # Use original function name for FUNC nodes - no normalization
        original_name = func_name
        
        # Generate description
        if func_info.docstring:
            # Use first sentence of docstring
            first_sentence = func_info.docstring.split('.')[0].strip()
            if first_sentence and len(first_sentence) < 100:
                descr = first_sentence
            else:
                descr = f"Function {func_info.name}"
        else:
            descr = f"Function {func_info.name}"
            if func_info.is_method:
                descr = f"Method {func_info.name}"
            if func_info.is_async:
                descr = f"Async {descr.lower()}"
        
        properties = {
            'Name': original_name,
            'Descr': descr
        }
        
        metadata = {
            'original_name': func_info.name,
            'full_name': func_info.full_name,
            'line_number': func_info.line_number,
            'end_line_number': func_info.end_line_number,
            'line_count': func_info.end_line_number - func_info.line_number + 1,
            'args': func_info.args,
            'arg_count': len(func_info.args),
            'returns': func_info.returns,
            'is_method': func_info.is_method,
            'is_classmethod': func_info.is_classmethod,
            'is_staticmethod': func_info.is_staticmethod,
            'is_property': func_info.is_property,
            'is_async': func_info.is_async,
            'parent_class': func_info.parent_class,
            'decorator_count': len(func_info.decorators),
            'decorators': [d.name for d in func_info.decorators],
            'call_count': len(func_info.calls),
            'complexity': func_info.complexity,
            'has_docstring': bool(func_info.docstring),
            'module': ast_result.module_name
        }
        
        return OntologyNode(
            uuid=self._generate_uuid('FUNC', original_name),
            type='FUNC',
            name=original_name,
            descr=descr,
            properties=properties,
            metadata=metadata
        )
    
    def _generate_relationship(self, rel_type: str, source_uuid: str, target_uuid: str,
                              description: str = "") -> OntologyRelationship:
        """Generate a relationship between two nodes."""
        rel_id = f"{source_uuid}-{rel_type}-{target_uuid}"
        
        properties = {}
        if rel_type == 'flow':
            # Flow relationships need FlowDescr and FlowDef
            properties['FlowDescr'] = description
            properties['FlowDef'] = description
        
        return OntologyRelationship(
            uuid=self._generate_uuid('REL', rel_id),
            type=rel_type,
            source=source_uuid,
            target=target_uuid,
            properties=properties,
            metadata={
                'description': description,
                'created_at': datetime.utcnow().isoformat()
            }
        )
    
    def _normalize_name(self, name: str, prefix: str = "") -> str:
        """Normalize name according to ontology naming conventions."""
        # Remove special characters and replace with underscores
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        
        # Remove multiple underscores
        clean_name = re.sub(r'_+', '_', clean_name)
        
        # Remove leading/trailing underscores
        clean_name = clean_name.strip('_')
        
        # Convert to PascalCase if configured
        if self.naming_style == 'PascalCase':
            if clean_name:
                parts = clean_name.split('_')
                clean_name = ''.join(word.capitalize() for word in parts if word)
        
        # Add prefix if provided and name doesn't start with it
        if prefix and not clean_name.startswith(prefix):
            clean_name = prefix + clean_name
        
        # Ensure name starts with uppercase letter
        if clean_name and not clean_name[0].isupper():
            clean_name = clean_name[0].upper() + clean_name[1:]
        
        # Truncate to max length
        if len(clean_name) > self.max_name_length:
            clean_name = clean_name[:self.max_name_length]
        
        # Ensure uniqueness
        original_name = clean_name
        counter = 1
        while clean_name in self._name_cache:
            suffix = str(counter)
            max_base = self.max_name_length - len(suffix)
            clean_name = original_name[:max_base] + suffix
            counter += 1
        
        self._name_cache.add(clean_name)
        return clean_name
    
    def _generate_uuid(self, node_type: str = None, identifier: str = None) -> str:
        """Generate random UUID for a node or relationship."""
        generated_uuid = str(uuid.uuid4())
        
        # Add prefix if configured
        if self.uuid_prefix:
            generated_uuid = f"{self.uuid_prefix}-{generated_uuid}"
        
        return generated_uuid
    
    def _should_include_function(self, func_info: FunctionInfo) -> bool:
        """Determine if a function should be included in the graph."""
        # Filter constructor methods (__init__)
        if func_info.name == '__init__':
            return False
        
        # Filter private functions
        if not self.include_private_functions and func_info.name.startswith('_'):
            return False
        
        # Filter test functions
        if not self.include_test_functions and self._is_test_function(func_info):
            return False
        
        # Filter by minimum lines
        line_count = func_info.end_line_number - func_info.line_number + 1
        if line_count < self.min_function_lines:
            return False
        
        return True
    
    def _is_test_function(self, func_info: FunctionInfo) -> bool:
        """Check if function is a test function."""
        return (
            func_info.name.startswith('test_') or
            func_info.name.endswith('_test') or
            any(d.name in ['pytest.mark', 'unittest', 'nose'] for d in func_info.decorators)
        )
    
    def _generate_graph_metadata(self, project_structure: ProjectStructure,
                                ast_results: List[ASTParseResult], graph: GraphData) -> Dict[str, Any]:
        """Generate metadata for the complete graph."""
        total_functions = sum(len(r.functions) for r in ast_results)
        total_classes = sum(len(r.classes) for r in ast_results)
        total_imports = sum(len(r.imports) for r in ast_results)
        
        # Count node types
        node_counts = {}
        for node in graph.nodes:
            node_counts[node.type] = node_counts.get(node.type, 0) + 1
        
        # Count relationship types
        rel_counts = {}
        for rel in graph.relationships:
            rel_counts[rel.type] = rel_counts.get(rel.type, 0) + 1
        
        metadata = {
            'generator_version': '1.0.0',
            'generated_at': datetime.utcnow().isoformat(),
            'project_path': str(project_structure.root_path),
            'project_name': project_structure.root_path.name,
            'total_files_analyzed': len(ast_results),
            'total_functions_found': total_functions,
            'total_classes_found': total_classes,
            'total_imports_found': total_imports,
            'node_counts': node_counts,
            'relationship_counts': rel_counts,
            'naming_style': self.naming_style,
            'max_name_length': self.max_name_length,
            'configuration': {
                'include_private_functions': self.include_private_functions,
                'include_test_functions': self.include_test_functions,
                'min_function_lines': self.min_function_lines
            }
        }
        
        return metadata
    
    def _generate_actor_nodes(self, actor_results: List[Any], func_nodes: List[OntologyNode]) -> List[OntologyNode]:
        """Generate ACTOR nodes from actor detection results."""
        actor_nodes = []
        actor_cache = {}  # To avoid duplicate actors
        
        try:
            for actor_result in actor_results:
                for match in actor_result.high_confidence_matches:
                    # Create unique actor identifier (without module to deduplicate across modules)
                    actor_type = match.actor_type.value if hasattr(match.actor_type, 'value') else str(match.actor_type)
                    library_type = match.context.get('library_type', 'unknown')
                    import_module = match.evidence.get('import_module', 'unknown')
                    actor_key = f"{actor_type}_{library_type}_{import_module}"
                    
                    # If we already created this actor, just add the source module
                    if actor_key in actor_cache:
                        existing_actor = actor_cache[actor_key]
                        if actor_result.module_name not in existing_actor.metadata['source_modules']:
                            existing_actor.metadata['source_modules'].append(actor_result.module_name)
                            existing_actor.metadata['evidence_count'] += 1
                        continue
                    
                    # Generate meaningful name based on actor type and library
                    if hasattr(match, 'enhanced_name') and match.enhanced_name:
                        actor_name = self._normalize_name(match.enhanced_name)
                    elif import_module and import_module != 'unknown':
                        # Use actual library name when available
                        actor_name = self._normalize_name(f"{actor_type}{import_module.title()}")
                    else:
                        # Fallback to generic name
                        actor_name = self._normalize_name(f"{actor_type}{library_type.title()}")
                    
                    # Generate meaningful description
                    if hasattr(match, 'enhanced_description') and match.enhanced_description:
                        descr = match.enhanced_description
                    else:
                        # Create specific description based on actual library
                        if import_module and import_module != 'unknown':
                            descr = f"{actor_type} using {import_module}"
                        else:
                            descr = f"{actor_type} using {library_type}"
                    
                    # Create ACTOR node
                    actor_node = OntologyNode(
                        uuid=self._generate_uuid('ACTOR', actor_key),
                        type='ACTOR',
                        name=actor_name,
                        descr=descr,
                        properties={
                            'Name': actor_name,
                            'Descr': descr
                        },
                        metadata={
                            'actor_type': actor_type,
                            'library_type': library_type,
                            'import_module': import_module,
                            'detection_confidence': match.confidence,
                            'detection_method': match.context.get('detection_method', 'unknown'),
                            'evidence_count': 1,
                            'source_modules': [actor_result.module_name]
                        }
                    )
                    
                    actor_nodes.append(actor_node)
                    actor_cache[actor_key] = actor_node
                    
                    logger.debug(f"Generated ACTOR node: {actor_name}")
        
        except Exception as e:
            logger.error(f"Error generating actor nodes: {e}")
        
        # Filter out phantom and redundant actors
        filtered_actors = self._filter_phantom_actors(actor_nodes, func_nodes)
        
        logger.info(f"Generated {len(actor_nodes)} actor nodes, filtered to {len(filtered_actors)} valid actors")
        return filtered_actors
    
    def _generate_flow_relationships(self, actor_results: List[Any], func_nodes: List[OntologyNode], 
                                   actor_nodes: List[OntologyNode]) -> List[OntologyRelationship]:
        """Generate flow relationships between actors and functions."""
        relationships = []
        
        try:
            # Create lookup maps
            func_lookup = {node.metadata.get('full_name', node.metadata.get('original_name', '')): node 
                          for node in func_nodes if node.type == 'FUNC'}
            
            actor_lookup = {}
            for node in actor_nodes:
                if node.type == 'ACTOR':
                    for module in node.metadata.get('source_modules', []):
                        key = f"{node.metadata.get('actor_type')}_{node.metadata.get('library_type')}_{module}"
                        actor_lookup[key] = node
            
            # Generate relationships based on detection matches
            for actor_result in actor_results:
                for match in actor_result.high_confidence_matches:
                    # Find corresponding actor node
                    actor_type = match.actor_type.value if hasattr(match.actor_type, 'value') else str(match.actor_type)
                    library_type = match.context.get('library_type', 'unknown')
                    actor_key = f"{actor_type}_{library_type}_{actor_result.module_name}"
                    
                    actor_node = actor_lookup.get(actor_key)
                    if not actor_node:
                        continue
                    
                    # Find function node if available
                    func_node = None
                    if match.function_name:
                        func_node = func_lookup.get(match.function_name)
                    
                    if func_node:
                        # Create flow relationship from function to actor (typical usage pattern)
                        flow_rel = self._generate_relationship(
                            'flow', 
                            func_node.uuid, 
                            actor_node.uuid,
                            f"Function {func_node.name} uses {actor_node.name}"
                        )
                        # Add required flow properties
                        flow_rel.properties['FlowDescr'] = f"Uses {actor_type} service"
                        flow_rel.properties['FlowDef'] = f"Function calls {library_type} library for {actor_type} operations"
                        
                        relationships.append(flow_rel)
                        
                        logger.debug(f"Generated flow relationship: {func_node.name} -> {actor_node.name}")
        
        except Exception as e:
            logger.error(f"Error generating flow relationships: {e}")
        
        return relationships
    
    def _generate_function_flow_relationships(self, ast_results: List[ASTParseResult], 
                                            func_nodes: List[OntologyNode]) -> List[OntologyRelationship]:
        """Generate flow relationships between functions based on AST analysis."""
        relationships = []
        
        try:
            logger.info("Generating function-to-function flow relationships")
            
            # Prepare AST data for flow analysis
            ast_data = {}
            for ast_result in ast_results:
                module_name = ast_result.module_name or Path(ast_result.file_path).stem
                
                # Convert AST result to format expected by flow analyzer
                file_data = {
                    'module_name': module_name,
                    'functions': []
                }
                
                # Add regular functions
                for func_info in ast_result.functions:
                    # Convert args to proper parameter format
                    parameters = []
                    if func_info.args:
                        for arg in func_info.args:
                            if isinstance(arg, dict):
                                parameters.append(arg)
                            elif isinstance(arg, str):
                                # Convert string args to dict format
                                parameters.append({
                                    'name': arg,
                                    'annotation': 'Any'
                                })
                    
                    func_data = {
                        'name': func_info.name,
                        'full_name': func_info.full_name,
                        'line_number': func_info.line_number,
                        'calls': func_info.calls or [],
                        'parameters': parameters,
                        'args': func_info.args or [],  # Keep original for fallback
                        'return_annotation': func_info.returns or 'None',
                        'decorators': [d.name for d in (func_info.decorators or [])],
                        'is_async': func_info.is_async,
                        'docstring': func_info.docstring or ''
                    }
                    file_data['functions'].append(func_data)
                
                # Add class methods
                for class_info in ast_result.classes:
                    for method_info in class_info.methods:
                        # Convert args to proper parameter format
                        parameters = []
                        if method_info.args:
                            for arg in method_info.args:
                                if isinstance(arg, dict):
                                    parameters.append(arg)
                                elif isinstance(arg, str):
                                    # Convert string args to dict format
                                    parameters.append({
                                        'name': arg,
                                        'annotation': 'Any'
                                    })
                        
                        func_data = {
                            'name': method_info.name,
                            'full_name': method_info.full_name,
                            'line_number': method_info.line_number,
                            'calls': method_info.calls or [],
                            'parameters': parameters,
                            'args': method_info.args or [],  # Keep original for fallback
                            'return_annotation': method_info.returns or 'None',
                            'decorators': [d.name for d in (method_info.decorators or [])],
                            'is_async': method_info.is_async,
                            'docstring': method_info.docstring or ''
                        }
                        file_data['functions'].append(func_data)
                
                ast_data[ast_result.file_path] = file_data
            
            # Analyze flows
            flow_relationships = self.flow_detector.analyze_flows(ast_data)
            
            # Enhance with LLM if enabled
            if self.llm_enhanced_flows:
                flow_relationships = self.flow_enhancer.enhance_flow_descriptions(flow_relationships)
            
            # Create lookup map for function nodes
            func_lookup = {}
            for node in func_nodes:
                if node.type == 'FUNC':
                    # Try multiple lookup keys
                    full_name = node.metadata.get('full_name', '')
                    original_name = node.metadata.get('original_name', '')
                    simple_name = node.name
                    
                    if full_name:
                        func_lookup[full_name] = node
                    if original_name:
                        func_lookup[original_name] = node
                    if simple_name:
                        func_lookup[simple_name] = node
            
            # Convert flow relationships to ontology relationships
            for flow_rel in flow_relationships:
                source_node = None
                target_node = None
                
                # Find source and target nodes with better lookup
                source_node = None
                target_node = None
                
                # Try multiple lookup strategies for source
                source_lookups = [
                    flow_rel.source_name,
                    flow_rel.source_name.split('.')[-1],
                    f"{flow_rel.source_name.split('.')[-1]}"
                ]
                for lookup_key in source_lookups:
                    if lookup_key in func_lookup:
                        source_node = func_lookup[lookup_key]
                        break
                
                # Try multiple lookup strategies for target
                target_lookups = [
                    flow_rel.target_name,
                    flow_rel.target_name.split('.')[-1],
                    f"{flow_rel.target_name.split('.')[-1]}"
                ]
                for lookup_key in target_lookups:
                    if lookup_key in func_lookup:
                        target_node = func_lookup[lookup_key]
                        break
                
                if source_node and target_node:
                    # Create ontology relationship
                    ontology_rel = self._generate_relationship(
                        'flow',
                        source_node.uuid,
                        target_node.uuid,
                        flow_rel.flow_descr
                    )
                    
                    # Add flow-specific properties
                    ontology_rel.properties['FlowDescr'] = flow_rel.flow_descr
                    ontology_rel.properties['FlowDef'] = flow_rel.flow_def
                    ontology_rel.properties['Confidence'] = flow_rel.confidence
                    ontology_rel.properties['source_name'] = flow_rel.source_name
                    ontology_rel.properties['target_name'] = flow_rel.target_name
                    
                    relationships.append(ontology_rel)
                    
                    logger.debug(f"Generated flow: {source_node.name} -> {target_node.name}")
                else:
                    logger.debug(f"Could not find nodes for flow: {flow_rel.source_name} -> {flow_rel.target_name}")
            
            logger.info(f"Generated {len(relationships)} function flow relationships")
            
        except Exception as e:
            logger.error(f"Error generating function flow relationships: {e}")
        
        return relationships
    
    def _generate_fchain_nodes(self, flow_relationships: List[OntologyRelationship]) -> List[OntologyNode]:
        """Generate FCHAIN nodes from flow relationships."""
        fchain_nodes = []
        
        try:
            if not flow_relationships:
                return fchain_nodes
            
            logger.info("Generating FCHAIN nodes")
            
            # Convert ontology relationships back to flow relationships for chain detection
            flow_rels = []
            for rel in flow_relationships:
                if rel.type == 'flow':
                    # We need to reconstruct source and target names for chain detection
                    source_name = rel.properties.get('source_name', f"func_{rel.source}")
                    target_name = rel.properties.get('target_name', f"func_{rel.target}")
                    
                    # Create a minimal flow relationship for chain detection
                    from .flow_analyzer import FlowRelationship
                    flow_rel = FlowRelationship(
                        source_uuid=rel.source,
                        target_uuid=rel.target,
                        source_name=source_name,
                        target_name=target_name,
                        flow_descr=rel.properties.get('FlowDescr', ''),
                        flow_def=rel.properties.get('FlowDef', ''),
                        context={}
                    )
                    flow_rels.append(flow_rel)
            
            # Detect functional chains
            fchain_data = self.flow_detector.detect_fchains(flow_rels)
            
            # Create FCHAIN nodes
            for chain_data in fchain_data:
                fchain_node = OntologyNode(
                    uuid=self._generate_uuid(f"fchain_{chain_data['name']}"),
                    type='FCHAIN',
                    name=chain_data['name'],
                    description=chain_data['description'],
                    metadata={
                        'functions': chain_data['functions'],
                        'chain_length': chain_data['length'],
                        'generated_by': 'flow_analyzer'
                    }
                )
                fchain_nodes.append(fchain_node)
            
            logger.info(f"Generated {len(fchain_nodes)} FCHAIN nodes")
            
        except Exception as e:
            logger.error(f"Error generating FCHAIN nodes: {e}")
        
        return fchain_nodes
    
    def _generate_function_actor_flow_relationships(self, ast_results: List[ASTParseResult], 
                                                   func_nodes: List[OntologyNode],
                                                   actor_nodes: List[OntologyNode]) -> List[OntologyRelationship]:
        """Generate flow relationships from functions to actors."""
        relationships = []
        
        try:
            logger.info("Generating function-to-actor flow relationships")
            
            # Create lookup maps
            func_lookup = {}
            for node in func_nodes:
                if node.type == 'FUNC':
                    func_lookup[node.metadata.get('full_name', '')] = node
                    func_lookup[node.metadata.get('original_name', '')] = node
                    func_lookup[node.name] = node
            
            actor_lookup = {}
            for node in actor_nodes:
                if node.type == 'ACTOR':
                    # Map by actor name and library patterns
                    actor_name = node.name.lower()
                    actor_lookup[actor_name] = node
                    
                    # Also map by common library patterns
                    if 'http' in actor_name:
                        actor_lookup['httpx'] = node
                        actor_lookup['requests'] = node
                        actor_lookup['aiohttp'] = node
                        actor_lookup['client'] = node
                    elif 'file' in actor_name:
                        actor_lookup['open'] = node
                        actor_lookup['pathlib'] = node
                        actor_lookup['os'] = node
                        actor_lookup['path'] = node
                    elif 'database' in actor_name or 'neo4j' in actor_name or 'graph' in actor_name:
                        actor_lookup['session'] = node
                        actor_lookup['driver'] = node
                        actor_lookup['tx'] = node
                        actor_lookup['graphdatabase'] = node
                        actor_lookup['neo4j'] = node
                        actor_lookup['execute_write'] = node
                        actor_lookup['execute_read'] = node
                        actor_lookup['run'] = node
            
            # Analyze function calls to detect actor usage
            for ast_result in ast_results:
                for func_info in ast_result.functions + [m for c in ast_result.classes for m in c.methods]:
                    func_full_name = func_info.full_name
                    func_node = func_lookup.get(func_full_name)
                    
                    if not func_node:
                        continue
                    
                    calls = func_info.calls or []
                    for call in calls:
                        if isinstance(call, str):
                            call_name = call
                        elif isinstance(call, dict):
                            call_name = call.get('name', '')
                        else:
                            continue
                        
                        # Check if this call matches an actor pattern
                        matched_actor = None
                        call_lower = call_name.lower()
                        
                        # Direct actor lookup
                        if call_lower in actor_lookup:
                            matched_actor = actor_lookup[call_lower]
                        else:
                            # Pattern matching for method calls
                            for pattern, actor in actor_lookup.items():
                                if (pattern in call_lower or 
                                    call_lower.startswith(pattern) or
                                    call_lower.endswith(pattern) or
                                    f'.{pattern}' in call_lower or
                                    f'{pattern}.' in call_lower):
                                    matched_actor = actor
                                    break
                            
                            # Additional database-specific patterns
                            if not matched_actor:
                                db_patterns = ['session.', 'driver.', 'tx.', '.execute_', '.run(']
                                for db_pattern in db_patterns:
                                    if db_pattern in call_lower:
                                        # Find database/neo4j actors
                                        for actor in actor_nodes:
                                            if actor.type == 'ACTOR' and ('database' in actor.name.lower() or 'neo4j' in actor.name.lower()):
                                                matched_actor = actor
                                                break
                                        break
                        
                        if matched_actor:
                            # Generate flow relationship
                            flow_rel = self._generate_relationship(
                                'flow',
                                func_node.uuid,
                                matched_actor.uuid,
                                f"Function {func_node.name} uses {matched_actor.name}"
                            )
                            
                            # Add flow-specific properties
                            purpose = self._infer_actor_usage_purpose(call_name, matched_actor.name)
                            flow_rel.properties['FlowDescr'] = f"{func_info.name} calls {call_name} for {purpose}"
                            flow_rel.properties['FlowDef'] = f"actor_call → {matched_actor.name} service"
                            flow_rel.properties['Confidence'] = 1.0
                            flow_rel.properties['source_name'] = func_info.name
                            flow_rel.properties['target_name'] = matched_actor.name
                            
                            relationships.append(flow_rel)
                            
                            logger.debug(f"Generated func→actor flow: {func_info.name} -> {matched_actor.name}")
            
            logger.info(f"Generated {len(relationships)} function-to-actor flow relationships")
            
        except Exception as e:
            logger.error(f"Error generating function-to-actor flow relationships: {e}")
        
        return relationships
    
    def _infer_actor_usage_purpose(self, call_name: str, actor_name: str) -> str:
        """Infer the purpose of a function calling an actor."""
        call_lower = call_name.lower()
        actor_lower = actor_name.lower()
        
        if 'http' in actor_lower:
            if any(pattern in call_lower for pattern in ['get', 'fetch', 'request']):
                return 'HTTP data retrieval'
            elif any(pattern in call_lower for pattern in ['post', 'put', 'send']):
                return 'HTTP data transmission'
            else:
                return 'HTTP communication'
        elif 'file' in actor_lower:
            if any(pattern in call_lower for pattern in ['open', 'read', 'load']):
                return 'file reading'
            elif any(pattern in call_lower for pattern in ['write', 'save', 'store']):
                return 'file writing'
            else:
                return 'file operations'
        elif 'database' in actor_lower or 'neo4j' in actor_lower or 'graph' in actor_lower:
            if any(pattern in call_lower for pattern in ['execute', 'run', 'query', 'cypher']):
                return 'database query execution'
            elif any(pattern in call_lower for pattern in ['session', 'connect', 'driver']):
                return 'database connection'
            elif any(pattern in call_lower for pattern in ['constraint', 'index', 'schema']):
                return 'database schema management'
            elif any(pattern in call_lower for pattern in ['write', 'create', 'update', 'delete']):
                return 'database write operations'
            else:
                return 'data persistence'
        else:
            return 'external service interaction'
    
    
    def _filter_phantom_actors(self, actor_nodes: List[OntologyNode], func_nodes: List[OntologyNode]) -> List[OntologyNode]:
        """Filter out phantom actors that have no actual connections or evidence."""
        if not actor_nodes:
            return actor_nodes
        
        valid_actors = []
        func_name_lookup = {node.name for node in func_nodes if node.type == 'FUNC'}
        
        # Group actors by type and library for consolidation
        actor_groups = {}
        
        for actor in actor_nodes:
            should_keep = True
            actor_type = actor.properties.get('ActorType', '')
            import_module = actor.metadata.get('import_module', 'unknown')
            library_type = actor.metadata.get('library_type', 'unknown')
            
            # Always keep OnDemand actors
            if actor_type == 'OnDemand':
                valid_actors.append(actor)
                continue
            
            # Filter WebEndpoint actors that reference non-existent functions
            if actor_type == 'WebEndpoint':
                source_function = actor.properties.get('SourceFunction', '')
                if not source_function or source_function not in func_name_lookup:
                    logger.debug(f"Filtering phantom WebEndpoint: {actor.name} -> {source_function} (function not found)")
                    should_keep = False
            
            # Aggressive filtering for Unknown actors
            elif actor_type == 'Unknown':
                # Keep only actors with strong evidence and meaningful libraries
                if import_module in ['neo4j', 'requests', 'httpx', 'fastapi', 'os']:
                    # Keep known important libraries, but consolidate
                    group_key = f"{actor_type}_{import_module}"
                    if group_key not in actor_groups:
                        actor_groups[group_key] = actor
                        should_keep = True
                    else:
                        # Consolidate with existing actor
                        existing = actor_groups[group_key]
                        existing.metadata['source_modules'].extend(actor.metadata.get('source_modules', []))
                        existing.metadata['evidence_count'] += actor.metadata.get('evidence_count', 0)
                        should_keep = False
                else:
                    # Filter out actors with weak evidence
                    evidence_count = actor.metadata.get('evidence_count', 0)
                    source_modules = actor.metadata.get('source_modules', [])
                    if evidence_count < 2 or len(source_modules) < 2:
                        logger.debug(f"Filtering weak Unknown actor: {actor.name} (evidence: {evidence_count}, modules: {len(source_modules)})")
                        should_keep = False
            
            # Filter actors with specific actor types but no meaningful connections
            elif actor_type in ['Database', 'HttpClient', 'FileSystem']:
                # Keep only if they have strong library evidence
                if import_module not in ['unknown'] and import_module:
                    group_key = f"{actor_type}_{import_module}"
                    if group_key not in actor_groups:
                        actor_groups[group_key] = actor
                        should_keep = True
                    else:
                        # Consolidate with existing actor
                        should_keep = False
                else:
                    should_keep = False
            
            if should_keep and actor not in valid_actors:
                valid_actors.append(actor)
                if actor_type != 'OnDemand':
                    group_key = f"{actor_type}_{import_module}"
                    actor_groups[group_key] = actor
            elif not should_keep:
                logger.debug(f"Filtered out phantom actor: {actor.name} (Type: {actor_type}, Module: {import_module})")
        
        return valid_actors
    
    def _generate_trigger_flow_relationships(self, actor_nodes: List[OntologyNode], 
                                           func_nodes: List[OntologyNode]) -> Tuple[List[OntologyRelationship], List[OntologyNode]]:
        """Generate TRIGGER flow relationships from actors to functions."""
        relationships = []
        new_actors = []
        
        try:
            logger.info("Generating trigger (Actor→Function) flow relationships")
            
            # Create lookup maps
            func_lookup = {}
            for node in func_nodes:
                if node.type == 'FUNC':
                    # Map by various name patterns
                    func_lookup[node.metadata.get('full_name', '')] = node
                    func_lookup[node.metadata.get('original_name', '')] = node
                    func_lookup[node.name] = node
                        
                    # Also map by simple function name
                    if '.' in node.metadata.get('full_name', ''):
                        simple_name = node.metadata.get('full_name', '').split('.')[-1]
                        func_lookup[simple_name] = node
            
            # Create ON-DEMAND actor for main/entry point functions
            main_funcs = [node for node in func_nodes if node.type == 'FUNC' and node.name == 'main']
            for main_func in main_funcs:
                # Create ON-DEMAND actor
                on_demand_actor = OntologyNode(
                    uuid=self._generate_uuid('ACTOR', 'on_demand_entry'),
                    type='ACTOR',
                    name='OnDemandEntry',
                    descr='Entry point trigger for main function execution',
                    properties={
                        'Name': 'OnDemandEntry',
                        'Descr': 'Entry point trigger for main function execution',
                        'ActorType': 'OnDemand'
                    },
                    metadata={
                        'actor_type': 'OnDemand',
                        'generated_by': 'trigger_generator'
                    }
                )
                
                # Add to new actors list
                new_actors.append(on_demand_actor)
                actor_nodes.append(on_demand_actor)
                
                # Create trigger flow: ON-DEMAND → main
                trigger_rel = OntologyRelationship(
                    uuid=self._generate_uuid('flow', f"trigger_ondemand_main"),
                    type='flow',
                    source=on_demand_actor.uuid,
                    target=main_func.uuid,
                    properties={
                        'FlowDescr': 'Entry point triggers main function execution',
                        'FlowDef': 'OnDemand trigger → main() execution',
                        'flow_type': 'trigger'
                    }
                )
                relationships.append(trigger_rel)
                logger.debug(f"Generated ON-DEMAND trigger: {on_demand_actor.name} → {main_func.name}")
            
            # Generate trigger relationships for WebEndpoint actors
            for actor_node in actor_nodes:
                if actor_node.type != 'ACTOR':
                    continue
                
                actor_type = actor_node.properties.get('ActorType', '')
                
                # WebEndpoint actors trigger their source functions
                if actor_type == 'WebEndpoint':
                    source_function = actor_node.properties.get('SourceFunction', '')
                    if source_function and source_function in func_lookup:
                        target_func = func_lookup[source_function]
                        
                        # Create trigger relationship
                        trigger_rel = self._generate_relationship(
                            'flow',
                            actor_node.uuid,
                            target_func.uuid,
                            f"{actor_node.name} triggers {target_func.name}"
                        )
                        
                        # Add trigger-specific properties
                        http_method = actor_node.properties.get('HttpMethod', 'GET')
                        trigger_rel.properties['FlowDescr'] = f"{actor_node.name} receives {http_method} request and triggers {source_function}"
                        trigger_rel.properties['FlowDef'] = f"HTTP {http_method} → function_call"
                        trigger_rel.properties['Confidence'] = 1.0
                        trigger_rel.properties['source_name'] = actor_node.name
                        trigger_rel.properties['target_name'] = source_function
                        trigger_rel.properties['trigger_type'] = 'http_endpoint'
                        
                        relationships.append(trigger_rel)
                        logger.debug(f"Generated trigger flow: {actor_node.name} -> {source_function}")
                    else:
                        # Simple endpoint - no trigger needed, it's self-contained
                        logger.debug(f"Simple endpoint {actor_node.name} - no trigger relationship needed")
                
                # External service actors can trigger entry point functions
                elif actor_type in ['HttpClient', 'ExternalApi']:
                    # Look for functions that might be triggered by external events
                    # This could be expanded based on specific patterns
                    entry_point_patterns = ['main', 'handler', 'process', 'start', 'run']
                    
                    for pattern in entry_point_patterns:
                        if pattern in func_lookup:
                            target_func = func_lookup[pattern]
                            
                            trigger_rel = self._generate_relationship(
                                'flow',
                                actor_node.uuid,
                                target_func.uuid,
                                f"{actor_node.name} triggers {target_func.name}"
                            )
                            
                            trigger_rel.properties['FlowDescr'] = f"{actor_node.name} initiates {pattern} function"
                            trigger_rel.properties['FlowDef'] = f"external_trigger → function_call"
                            trigger_rel.properties['Confidence'] = 0.8
                            trigger_rel.properties['source_name'] = actor_node.name
                            trigger_rel.properties['target_name'] = pattern
                            trigger_rel.properties['trigger_type'] = 'external_service'
                            
                            relationships.append(trigger_rel)
                            logger.debug(f"Generated external trigger: {actor_node.name} -> {pattern}")
                            break  # Only one trigger per external actor
            
            logger.info(f"Generated {len(relationships)} trigger flow relationships")
            
        except Exception as e:
            logger.error(f"Error generating trigger flow relationships: {e}")
        
        return relationships, new_actors