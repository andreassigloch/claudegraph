#!/usr/bin/env python3
"""
UUID Deduplication for Code Architecture Analyzer

Ensures unique UUIDs across all nodes and relationships in the graph.
"""

import logging
import uuid
from typing import Dict, Set, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UUIDCollision:
    """Represents a UUID collision that was resolved"""
    original_uuid: str
    new_uuid: str
    node_type: str
    node_name: str
    context: str


class UUIDDeduplicator:
    """Ensures unique UUIDs in graph data"""
    
    def __init__(self):
        self.used_uuids: Set[str] = set()
        self.uuid_collisions: List[UUIDCollision] = []
        self.uuid_mapping: Dict[str, str] = {}  # old_uuid -> new_uuid mapping
        self.namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # DNS namespace
    
    def deduplicate_graph_data(self, graph_data):
        """Remove duplicate UUIDs from graph data"""
        
        # First pass: collect all existing UUIDs and fix duplicates in nodes
        self._deduplicate_nodes(graph_data.nodes)
        
        # Second pass: fix any relationship UUIDs and update references
        self._deduplicate_relationships(graph_data.relationships)
        
        # Third pass: update relationship source/target references
        self._update_relationship_references(graph_data.relationships)
        
        # Log results
        if self.uuid_collisions:
            logger.warning(f"Fixed {len(self.uuid_collisions)} UUID collisions")
            for collision in self.uuid_collisions:
                logger.debug(f"UUID collision: {collision.node_name} ({collision.node_type}) "
                           f"{collision.original_uuid} → {collision.new_uuid}")
        
        return graph_data
    
    def _deduplicate_nodes(self, nodes):
        """Deduplicate node UUIDs"""
        
        for i, node in enumerate(nodes):
            if node.uuid in self.used_uuids:
                # Generate new unique UUID
                original_uuid = node.uuid
                new_uuid = self._generate_unique_uuid(node.type, node.name, i)
                
                # Update node
                node.uuid = new_uuid
                
                # Record UUID mapping for relationship updates
                self.uuid_mapping[original_uuid] = new_uuid
                
                # Record collision
                self.uuid_collisions.append(UUIDCollision(
                    original_uuid=original_uuid,
                    new_uuid=new_uuid,
                    node_type=node.type,
                    node_name=node.name,
                    context=f"node_{i}"
                ))
                
                logger.debug(f"Fixed duplicate node UUID: {node.name} ({node.type}) "
                           f"{original_uuid} → {new_uuid}")
            
            self.used_uuids.add(node.uuid)
    
    def _deduplicate_relationships(self, relationships):
        """Deduplicate relationship UUIDs"""
        
        for i, rel in enumerate(relationships):
            if rel.uuid in self.used_uuids:
                # Generate new unique UUID
                original_uuid = rel.uuid
                new_uuid = self._generate_unique_uuid("REL", f"{rel.type}_{i}", i)
                
                # Update relationship
                rel.uuid = new_uuid
                
                # Record collision
                self.uuid_collisions.append(UUIDCollision(
                    original_uuid=original_uuid,
                    new_uuid=new_uuid,
                    node_type="REL",
                    node_name=rel.type,
                    context=f"rel_{i}"
                ))
                
                logger.debug(f"Fixed duplicate relationship UUID: {rel.type} "
                           f"{original_uuid} → {new_uuid}")
            
            self.used_uuids.add(rel.uuid)
    
    def _update_relationship_references(self, relationships):
        """Update relationship source/target references to new UUIDs"""
        
        for rel in relationships:
            # Update source reference if it was remapped
            if rel.source in self.uuid_mapping:
                old_source = rel.source
                rel.source = self.uuid_mapping[old_source]
                logger.debug(f"Updated relationship source: {old_source} → {rel.source}")
            
            # Update target reference if it was remapped
            if rel.target in self.uuid_mapping:
                old_target = rel.target
                rel.target = self.uuid_mapping[old_target]
                logger.debug(f"Updated relationship target: {old_target} → {rel.target}")
    
    def _generate_unique_uuid(self, node_type: str, identifier: str, index: int) -> str:
        """Generate a unique UUID that won't collide"""
        
        # Try different suffixes until we get a unique UUID
        for suffix in [f"_{index}", f"_dup_{index}", f"_v{index}", f"_alt{index}"]:
            unique_string = f"{node_type}-{identifier}{suffix}"
            candidate_uuid = str(uuid.uuid5(self.namespace, unique_string))
            
            if candidate_uuid not in self.used_uuids:
                return candidate_uuid
        
        # Fallback to random UUID if all else fails
        return str(uuid.uuid4())
    
    def get_collision_summary(self) -> Dict[str, int]:
        """Get summary of UUID collisions by type"""
        summary = {}
        for collision in self.uuid_collisions:
            node_type = collision.node_type
            summary[node_type] = summary.get(node_type, 0) + 1
        return summary