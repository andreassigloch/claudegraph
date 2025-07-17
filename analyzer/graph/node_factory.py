#!/usr/bin/env python3
"""
Node Factory for Code Architecture Analyzer

Handles creation of ontology-compliant nodes from AST parsing results.
"""

import uuid
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

from ..core.ast_parser import ASTParseResult, FunctionInfo, ClassInfo
from ..core.project_discoverer import ProjectStructure, ProjectFile

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


class NodeFactory:
    """Factory for creating different types of ontology nodes."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize node factory with configuration."""
        self.config = config or {}
        self.include_test_functions = self.config.get('analysis', {}).get('include_test_functions', False)
        self.min_function_lines = self.config.get('analysis', {}).get('min_function_lines', 1)
        self.exclude_private_functions = self.config.get('analysis', {}).get('exclude_private_functions', False)
    
    def create_system_node(self, project_structure: ProjectStructure) -> OntologyNode:
        """Generate system node for the entire project."""
        project_path = project_structure.root_path
        project_name = project_path.name if project_path.name else "unknown_project"
        
        # Ensure project name is valid
        normalized_name = self._normalize_name(project_name)
        
        return OntologyNode(
            uuid=self._generate_uuid("SYS", normalized_name),
            type="SYS",
            name=normalized_name,
            descr=f"System representing the {project_name} project",
            properties={
                "ProjectPath": str(project_path),
                "TotalFiles": project_structure.total_files,
                "PythonFiles": len(project_structure.python_files),
                "TotalLines": project_structure.total_lines,
                "ProjectType": "Python"
            }
        )
    
    def create_module_node(self, ast_result: ASTParseResult, project_structure: ProjectStructure) -> OntologyNode:
        """Generate module node for a Python file."""
        project_file = next((f for f in project_structure.python_files 
                           if f.path == ast_result.file_path), None)
        
        if not project_file:
            # Create a basic project file representation
            project_file = ProjectFile(
                path=ast_result.file_path,
                relative_path=ast_result.file_path.relative_to(project_structure.root_path) 
                             if project_structure.root_path in ast_result.file_path.parents 
                             else ast_result.file_path,
                size_bytes=0,
                line_count=len(ast_result.functions) * 10,  # Rough estimate
                module_name=ast_result.module_name
            )
        
        normalized_name = self._normalize_name(ast_result.module_name)
        
        return OntologyNode(
            uuid=self._generate_uuid("MOD", normalized_name),
            type="MOD",
            name=normalized_name,
            descr=f"Module {ast_result.module_name}",
            properties={
                "FilePath": str(ast_result.file_path),
                "RelativePath": str(project_file.relative_path),
                "ModuleName": ast_result.module_name,
                "LinesOfCode": project_file.line_count,
                "FunctionCount": len(ast_result.functions),
                "ClassCount": len(ast_result.classes),
                "ImportCount": len(ast_result.imports)
            }
        )
    
    def create_function_node(self, func_info: FunctionInfo, ast_result: ASTParseResult, 
                           module_node: OntologyNode) -> Optional[OntologyNode]:
        """Generate function node from function info."""
        if not self._should_include_function(func_info):
            return None
        
        function_name = func_info.name
        normalized_name = self._normalize_name(function_name)
        
        # Build description
        descr_parts = [f"Function {function_name}"]
        if func_info.is_async:
            descr_parts.append("(async)")
        if func_info.decorators:
            # Extract decorator names (handle both string and DecoratorInfo objects)
            decorator_names = []
            for decorator in func_info.decorators:
                if hasattr(decorator, 'name'):
                    decorator_names.append(decorator.name)
                else:
                    decorator_names.append(str(decorator))
            descr_parts.append(f"with decorators: {', '.join(decorator_names)}")
        if func_info.docstring:
            descr_parts.append(f"- {func_info.docstring[:100]}...")
        
        return OntologyNode(
            uuid=self._generate_uuid("FUNC", f"{module_node.name}.{normalized_name}"),
            type="FUNC",
            name=normalized_name,
            descr=" ".join(descr_parts),
            properties={
                "FunctionName": function_name,
                "ModuleName": ast_result.module_name,
                "IsAsync": func_info.is_async,
                "IsMethod": func_info.is_method,
                "IsPrivate": function_name.startswith('_'),
                "IsProperty": getattr(func_info, 'is_property', False),
                "LineNumber": func_info.line_number,
                "EndLineNumber": func_info.end_line_number,
                "ParameterCount": len(func_info.args),
                "Parameters": func_info.args,
                "ReturnType": getattr(func_info, 'return_type', func_info.returns),
                "Decorators": [decorator.name if hasattr(decorator, 'name') else str(decorator) for decorator in func_info.decorators],
                "CallCount": len(func_info.calls),
                "FunctionCalls": [call for call in func_info.calls],
                "HasDocstring": bool(func_info.docstring),
                "Complexity": func_info.complexity if hasattr(func_info, 'complexity') else 1
            }
        )
    
    def create_actor_nodes(self, actor_results: List[Any], func_nodes: List[OntologyNode]) -> List[OntologyNode]:
        """Generate actor nodes from detection results."""
        actor_nodes = []
        
        if not actor_results:
            logger.warning("No actor results provided for node generation")
            return actor_nodes
        
        for actor_result in actor_results:
            try:
                # Handle different actor result formats
                if hasattr(actor_result, 'detected_actors'):
                    # ActorDetectionResult format
                    actors = actor_result.detected_actors
                elif hasattr(actor_result, 'actors'):
                    # Direct actors list
                    actors = actor_result.actors
                elif isinstance(actor_result, dict) and 'actors' in actor_result:
                    # Dictionary format
                    actors = actor_result['actors']
                else:
                    logger.warning(f"Unexpected actor result format: {type(actor_result)}")
                    continue
                
                for actor in actors:
                    actor_node = self._create_single_actor_node(actor, func_nodes)
                    if actor_node:
                        actor_nodes.append(actor_node)
                        
            except Exception as e:
                logger.error(f"Error processing actor result: {e}")
                continue
        
        # Filter out phantom actors (actors with no real connections)
        filtered_nodes = self._filter_phantom_actors(actor_nodes, func_nodes)
        
        logger.info(f"Generated {len(filtered_nodes)} actor nodes (filtered from {len(actor_nodes)})")
        return filtered_nodes
    
    def _create_single_actor_node(self, actor: Any, func_nodes: List[OntologyNode]) -> Optional[OntologyNode]:
        """Create a single actor node."""
        try:
            # Extract actor properties based on format
            if hasattr(actor, 'actor_type'):
                actor_type = actor.actor_type
                actor_name = getattr(actor, 'name', str(actor_type))
                confidence = getattr(actor, 'confidence', 0.8)
                evidence = getattr(actor, 'evidence', {})
            elif isinstance(actor, dict):
                actor_type = actor.get('type', 'Unknown')
                actor_name = actor.get('name', str(actor_type))
                confidence = actor.get('confidence', 0.8)
                evidence = actor.get('evidence', {})
            else:
                logger.warning(f"Unknown actor format: {type(actor)}")
                return None
            
            normalized_name = self._normalize_name(actor_name)
            
            return OntologyNode(
                uuid=self._generate_uuid("ACTOR", normalized_name),
                type="ACTOR",
                name=normalized_name,
                descr=f"External actor of type {actor_type}",
                properties={
                    "ActorType": str(actor_type),
                    "ActorName": actor_name,
                    "Confidence": confidence,
                    "Evidence": evidence
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating actor node: {e}")
            return None
    
    def _should_include_function(self, func_info: FunctionInfo) -> bool:
        """Determine if a function should be included in the graph."""
        # Skip test functions if configured
        if not self.include_test_functions and self._is_test_function(func_info):
            return False
        
        # Skip very short functions if configured
        if self.min_function_lines > 1:
            lines = (func_info.end_line_number or func_info.line_number) - func_info.line_number + 1
            if lines < self.min_function_lines:
                return False
        
        # Skip private functions if configured
        if self.exclude_private_functions and func_info.name.startswith('_'):
            return False
        
        return True
    
    def _is_test_function(self, func_info: FunctionInfo) -> bool:
        """Check if a function is a test function."""
        name = func_info.name.lower()
        
        # Check decorator names (handle both string and DecoratorInfo objects)
        decorator_names = []
        for decorator in func_info.decorators:
            if hasattr(decorator, 'name'):
                decorator_names.append(decorator.name)
            else:
                decorator_names.append(str(decorator))
        
        return (name.startswith('test_') or 
                name.endswith('_test') or
                'test' in decorator_names or
                any(dec_name.startswith('pytest.') for dec_name in decorator_names))
    
    def _filter_phantom_actors(self, actor_nodes: List[OntologyNode], func_nodes: List[OntologyNode]) -> List[OntologyNode]:
        """Filter out actors that have no real connections to functions."""
        if not actor_nodes:
            return []
        
        # For now, return all actors - more sophisticated filtering can be added later
        # This would require analyzing actual flow relationships
        return actor_nodes
    
    def _normalize_name(self, name: str) -> str:
        """Normalize a name for use in the ontology."""
        if not name:
            name = "unnamed"
        
        # Remove problematic characters and normalize
        normalized = re.sub(r'[^a-zA-Z0-9_.]', '_', name)
        normalized = re.sub(r'_+', '_', normalized)  # Collapse multiple underscores
        normalized = normalized.strip('_')
        
        if not normalized:
            normalized = "unnamed"
        
        # Ensure it doesn't start with a number
        if normalized[0].isdigit():
            normalized = f"_{normalized}"
        
        return normalized
    
    def _generate_uuid(self, node_type: str = None, identifier: str = None) -> str:
        """Generate a UUID for a node."""
        if node_type and identifier:
            # Create deterministic UUID based on type and identifier
            combined = f"{node_type}_{identifier}"
            return str(uuid.uuid5(uuid.NAMESPACE_OID, combined))
        else:
            # Generate random UUID
            return str(uuid.uuid4())