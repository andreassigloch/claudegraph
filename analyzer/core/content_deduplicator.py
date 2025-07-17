#!/usr/bin/env python3
"""
Content-based deduplication for graph nodes

Removes duplicate nodes based on their actual content (name, type, function)
rather than UUID. Uses real UUIDs and removes content duplicates.
"""

import logging
import uuid
from typing import Dict, Set, List, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class ContentDeduplicator:
    """Remove duplicate nodes based on content, keep real UUIDs"""
    
    def __init__(self):
        self.removed_count = 0
        self.uuid_mappings = {}  # old_uuid -> kept_uuid
    
    def deduplicate_by_content(self, graph_data):
        """Remove nodes with same content, update relationship references"""
        
        # Group nodes by content signature
        node_groups = self._group_nodes_by_content(graph_data.nodes)
        
        # Keep one node from each group, track UUID mappings
        unique_nodes = []
        for signature, nodes in node_groups.items():
            if len(nodes) > 1:
                # Multiple nodes with same content - keep first, map others
                kept_node = nodes[0]
                unique_nodes.append(kept_node)
                
                for duplicate_node in nodes[1:]:
                    self.uuid_mappings[duplicate_node.uuid] = kept_node.uuid
                    self.removed_count += 1
                    logger.debug(f"Removed duplicate: {duplicate_node.name} ({duplicate_node.type}) "
                               f"{duplicate_node.uuid} → {kept_node.uuid}")
            else:
                # Unique content
                unique_nodes.append(nodes[0])
        
        # Update graph with unique nodes
        graph_data.nodes = unique_nodes
        
        # Update relationship references
        self._update_relationship_references(graph_data.relationships)
        
        # Remove relationships that now have same source and target
        graph_data.relationships = self._remove_duplicate_relationships(graph_data.relationships)
        
        if self.removed_count > 0:
            logger.info(f"Content deduplication: removed {self.removed_count} duplicate nodes")
        
        return graph_data
    
    def _group_nodes_by_content(self, nodes) -> Dict[str, List]:
        """Group nodes by their content signature"""
        groups = defaultdict(list)
        
        for node in nodes:
            signature = self._get_content_signature(node)
            groups[signature].append(node)
        
        return groups
    
    def _get_content_signature(self, node) -> str:
        """Generate content signature for a node (excluding UUID)"""
        # Use type + name as primary signature
        signature = f"{node.type}:{node.name}"
        
        # For functions, include class context if available
        if node.type == "FUNC" and hasattr(node, 'properties'):
            full_name = node.properties.get('full_name', '')
            if full_name:
                signature = f"{node.type}:{full_name}"
        
        return signature
    
    def _update_relationship_references(self, relationships):
        """Update relationship source/target to point to kept nodes"""
        for rel in relationships:
            # Handle both OntologyRelationship (source_uuid/target_uuid) and dict formats (source/target)
            source_attr = 'source_uuid' if hasattr(rel, 'source_uuid') else 'source'
            target_attr = 'target_uuid' if hasattr(rel, 'target_uuid') else 'target'
            
            source_value = getattr(rel, source_attr)
            target_value = getattr(rel, target_attr)
            
            if source_value in self.uuid_mappings:
                old_source = source_value
                setattr(rel, source_attr, self.uuid_mappings[old_source])
                logger.debug(f"Updated relationship source: {old_source} → {getattr(rel, source_attr)}")
            
            if target_value in self.uuid_mappings:
                old_target = target_value
                setattr(rel, target_attr, self.uuid_mappings[old_target])
                logger.debug(f"Updated relationship target: {old_target} → {getattr(rel, target_attr)}")
    
    def _remove_duplicate_relationships(self, relationships) -> List:
        """Remove relationships that became duplicates after node merging"""
        unique_rels = []
        seen_signatures = set()
        
        for rel in relationships:
            # Create signature for relationship
            rel_signature = f"{rel.type}:{rel.source}:{rel.target}"
            
            if rel_signature not in seen_signatures:
                unique_rels.append(rel)
                seen_signatures.add(rel_signature)
            else:
                logger.debug(f"Removed duplicate relationship: {rel.type} {rel.source} → {rel.target}")
        
        return unique_rels