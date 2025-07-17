"""
Analysis Comparator - Compare analysis results by content rather than UUIDs.

This module provides functionality to compare two analysis results in a deterministic way,
focusing on the actual content (names, types, relationships) rather than random UUIDs.
"""

import json
from typing import Dict, List, Set, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AnalysisComparator:
    """Compare analysis results by content rather than UUIDs."""
    
    def __init__(self):
        self.comparison_cache = {}
    
    def compare_analyses(self, result1: dict, result2: dict) -> dict:
        """
        Compare two analysis results by name and content.
        
        Args:
            result1: First analysis result dictionary
            result2: Second analysis result dictionary
            
        Returns:
            Detailed comparison report dictionary
        """
        comparison = {
            'nodes_match': self._compare_nodes(result1, result2),
            'relationships_match': self._compare_relationships(result1, result2),
            'metadata_match': self._compare_metadata(result1, result2),
            'summary': {}
        }
        
        # Generate summary
        nodes_identical = comparison['nodes_match']['identical']
        rels_identical = comparison['relationships_match']['identical']
        meta_identical = comparison['metadata_match']['identical']
        
        comparison['summary'] = {
            'identical': all([nodes_identical, rels_identical, meta_identical]),
            'nodes_identical': nodes_identical,
            'relationships_identical': rels_identical,
            'metadata_identical': meta_identical,
            'differences_count': (
                len(comparison['nodes_match'].get('differences', [])) +
                len(comparison['relationships_match'].get('differences', []))
            ),
            'total_nodes': {
                'first': len(result1.get('nodes', [])),
                'second': len(result2.get('nodes', []))
            },
            'total_relationships': {
                'first': len(result1.get('relationships', [])),
                'second': len(result2.get('relationships', []))
            }
        }
        
        return comparison
    
    def _compare_nodes(self, result1: dict, result2: dict) -> dict:
        """Compare nodes by name, type, and description."""
        nodes1 = {self._node_key(n): n for n in result1.get('nodes', [])}
        nodes2 = {self._node_key(n): n for n in result2.get('nodes', [])}
        
        # Find nodes only in first or second
        only_in_first = set(nodes1.keys()) - set(nodes2.keys())
        only_in_second = set(nodes2.keys()) - set(nodes1.keys())
        
        # Find content differences in common nodes
        common_nodes = set(nodes1.keys()) & set(nodes2.keys())
        content_differences = []
        
        for node_key in common_nodes:
            node1 = nodes1[node_key]
            node2 = nodes2[node_key]
            
            diffs = self._find_node_content_differences(node1, node2)
            if diffs:
                content_differences.append({
                    'node_key': node_key,
                    'differences': diffs
                })
        
        return {
            'identical': len(only_in_first) == 0 and len(only_in_second) == 0 and len(content_differences) == 0,
            'only_in_first': sorted(list(only_in_first)),
            'only_in_second': sorted(list(only_in_second)),
            'content_differences': content_differences,
            'common_nodes_count': len(common_nodes),
            'differences': list(only_in_first) + list(only_in_second) + [d['node_key'] for d in content_differences]
        }
    
    def _node_key(self, node: dict) -> str:
        """Generate unique key for node based on content."""
        node_type = node.get('type', 'UNKNOWN')
        node_name = node.get('Name', 'UNNAMED')
        # Use first 50 chars of description to handle long descriptions
        node_desc = node.get('Descr', '')[:50]
        return f"{node_type}:{node_name}:{node_desc}"
    
    def _find_node_content_differences(self, node1: dict, node2: dict) -> List[str]:
        """Find differences in node content (excluding UUID)."""
        differences = []
        
        # Compare important fields (excluding UUID)
        important_fields = ['type', 'Name', 'Descr', 'ActorType', 'Operations', 'Confidence']
        
        for field in important_fields:
            val1 = node1.get(field)
            val2 = node2.get(field)
            
            if val1 != val2:
                differences.append(f"{field}: '{val1}' vs '{val2}'")
        
        return differences
    
    def _compare_relationships(self, result1: dict, result2: dict) -> dict:
        """Compare relationships by source/target names and type."""
        # Build name mapping from UUIDs
        name_map1 = self._build_uuid_to_name_map(result1.get('nodes', []))
        name_map2 = self._build_uuid_to_name_map(result2.get('nodes', []))
        
        # Convert relationships to content-based keys
        rels1 = set()
        rels1_details = {}
        for rel in result1.get('relationships', []):
            rel_key = self._relationship_key(rel, name_map1)
            rels1.add(rel_key)
            rels1_details[rel_key] = rel
        
        rels2 = set()
        rels2_details = {}
        for rel in result2.get('relationships', []):
            rel_key = self._relationship_key(rel, name_map2)
            rels2.add(rel_key)
            rels2_details[rel_key] = rel
        
        # Find differences
        only_in_first = rels1 - rels2
        only_in_second = rels2 - rels1
        common_rels = rels1 & rels2
        
        # Check for content differences in common relationships
        content_differences = []
        for rel_key in common_rels:
            rel1 = rels1_details[rel_key]
            rel2 = rels2_details[rel_key]
            
            diffs = self._find_relationship_content_differences(rel1, rel2)
            if diffs:
                content_differences.append({
                    'relationship_key': rel_key,
                    'differences': diffs
                })
        
        return {
            'identical': len(only_in_first) == 0 and len(only_in_second) == 0 and len(content_differences) == 0,
            'only_in_first': sorted(list(only_in_first)),
            'only_in_second': sorted(list(only_in_second)),
            'content_differences': content_differences,
            'common_relationships_count': len(common_rels),
            'differences': list(only_in_first) + list(only_in_second) + [d['relationship_key'] for d in content_differences]
        }
    
    def _relationship_key(self, rel: dict, name_map: dict) -> str:
        """Generate unique key for relationship based on names."""
        source_name = name_map.get(rel.get('source'), 'UNKNOWN')
        target_name = name_map.get(rel.get('target'), 'UNKNOWN')
        rel_type = rel.get('type', 'UNKNOWN')
        
        # Include additional relationship properties for uniqueness
        flow_descr = rel.get('FlowDescr', '')[:30]  # First 30 chars
        direction = rel.get('direction', '')
        operation = rel.get('operation', '')
        
        # Create comprehensive key
        key_parts = [source_name, rel_type, target_name]
        if flow_descr:
            key_parts.append(f"desc:{flow_descr}")
        if direction:
            key_parts.append(f"dir:{direction}")
        if operation:
            key_parts.append(f"op:{operation}")
        
        return "--".join(key_parts)
    
    def _find_relationship_content_differences(self, rel1: dict, rel2: dict) -> List[str]:
        """Find differences in relationship content (excluding UUID)."""
        differences = []
        
        # Compare important fields (excluding UUIDs)
        important_fields = ['type', 'FlowDescr', 'FlowDef', 'direction', 'operation', 'actor_type', 'Confidence']
        
        for field in important_fields:
            val1 = rel1.get(field)
            val2 = rel2.get(field)
            
            if val1 != val2:
                differences.append(f"{field}: '{val1}' vs '{val2}'")
        
        return differences
    
    def _build_uuid_to_name_map(self, nodes: list) -> dict:
        """Build mapping from UUID to node name."""
        return {node.get('uuid'): node.get('Name', 'UNNAMED') for node in nodes}
    
    def _compare_metadata(self, result1: dict, result2: dict) -> dict:
        """Compare metadata sections (excluding timestamps and paths)."""
        meta1 = result1.get('metadata', {})
        meta2 = result2.get('metadata', {})
        
        # Fields to compare (excluding variable ones like timestamps)
        stable_fields = [
            'analysis_version', 
            'llm_enabled'
        ]
        
        # Stats fields to compare
        stats_fields = [
            'files_discovered',
            'files_parsed', 
            'functions_found',
            'classes_found',
            'trigger_actors',
            'receiver_actors',
            'flow_chains'
        ]
        
        differences = []
        
        # Compare stable metadata fields
        for field in stable_fields:
            val1 = meta1.get(field)
            val2 = meta2.get(field)
            
            if val1 != val2:
                differences.append(f"metadata.{field}: '{val1}' vs '{val2}'")
        
        # Compare stats (if available)
        stats1 = meta1.get('analysis_stats', {})
        stats2 = meta2.get('analysis_stats', {})
        
        for field in stats_fields:
            val1 = stats1.get(field)
            val2 = stats2.get(field)
            
            if val1 != val2:
                differences.append(f"stats.{field}: {val1} vs {val2}")
        
        return {
            'identical': len(differences) == 0,
            'differences': differences
        }
    
    def compare_files(self, file1_path: str, file2_path: str) -> dict:
        """
        Compare two analysis result files.
        
        Args:
            file1_path: Path to first analysis file
            file2_path: Path to second analysis file
            
        Returns:
            Comparison result dictionary
        """
        try:
            with open(file1_path, 'r') as f1:
                result1 = json.load(f1)
            
            with open(file2_path, 'r') as f2:
                result2 = json.load(f2)
            
            comparison = self.compare_analyses(result1, result2)
            
            # Add file information
            comparison['file_info'] = {
                'file1': str(Path(file1_path).name),
                'file2': str(Path(file2_path).name),
                'file1_size': Path(file1_path).stat().st_size,
                'file2_size': Path(file2_path).stat().st_size
            }
            
            return comparison
            
        except FileNotFoundError as e:
            return {'error': f'File not found: {e}'}
        except json.JSONDecodeError as e:
            return {'error': f'Invalid JSON: {e}'}
        except Exception as e:
            return {'error': f'Comparison failed: {e}'}
    
    def generate_comparison_report(self, comparison: dict) -> str:
        """Generate a human-readable comparison report."""
        lines = []
        
        lines.append("=" * 60)
        lines.append("ANALYSIS COMPARISON REPORT")
        lines.append("=" * 60)
        
        # Summary
        summary = comparison.get('summary', {})
        if summary.get('identical', False):
            lines.append("✅ RESULTS ARE FUNCTIONALLY IDENTICAL!")
            lines.append("   (UUIDs may differ but content is the same)")
        else:
            lines.append("❌ RESULTS DIFFER IN CONTENT!")
            lines.append(f"   Found {summary.get('differences_count', 0)} differences")
        
        lines.append("")
        
        # Node comparison
        lines.append("📊 NODE ANALYSIS:")
        nodes_match = comparison.get('nodes_match', {})
        lines.append(f"   Nodes in first:  {summary.get('total_nodes', {}).get('first', 0)}")
        lines.append(f"   Nodes in second: {summary.get('total_nodes', {}).get('second', 0)}")
        lines.append(f"   Common nodes:    {nodes_match.get('common_nodes_count', 0)}")
        
        if nodes_match.get('only_in_first'):
            lines.append(f"   Only in first:   {len(nodes_match['only_in_first'])} nodes")
            for node in nodes_match['only_in_first'][:3]:  # Show first 3
                lines.append(f"     - {node}")
            if len(nodes_match['only_in_first']) > 3:
                lines.append(f"     ... and {len(nodes_match['only_in_first']) - 3} more")
        
        if nodes_match.get('only_in_second'):
            lines.append(f"   Only in second:  {len(nodes_match['only_in_second'])} nodes")
            for node in nodes_match['only_in_second'][:3]:  # Show first 3
                lines.append(f"     - {node}")
            if len(nodes_match['only_in_second']) > 3:
                lines.append(f"     ... and {len(nodes_match['only_in_second']) - 3} more")
        
        lines.append("")
        
        # Relationship comparison
        lines.append("🔗 RELATIONSHIP ANALYSIS:")
        rels_match = comparison.get('relationships_match', {})
        lines.append(f"   Relationships in first:  {summary.get('total_relationships', {}).get('first', 0)}")
        lines.append(f"   Relationships in second: {summary.get('total_relationships', {}).get('second', 0)}")
        lines.append(f"   Common relationships:    {rels_match.get('common_relationships_count', 0)}")
        
        if rels_match.get('only_in_first'):
            lines.append(f"   Only in first:   {len(rels_match['only_in_first'])} relationships")
        
        if rels_match.get('only_in_second'):
            lines.append(f"   Only in second:  {len(rels_match['only_in_second'])} relationships")
        
        lines.append("")
        
        # Metadata comparison
        lines.append("📋 METADATA ANALYSIS:")
        meta_match = comparison.get('metadata_match', {})
        if meta_match.get('identical', True):
            lines.append("   ✅ Metadata matches (stable fields)")
        else:
            lines.append("   ❌ Metadata differs:")
            for diff in meta_match.get('differences', []):
                lines.append(f"     - {diff}")
        
        # File information (if available)
        file_info = comparison.get('file_info')
        if file_info:
            lines.append("")
            lines.append("📁 FILE INFORMATION:")
            lines.append(f"   First file:  {file_info['file1']} ({file_info['file1_size']} bytes)")
            lines.append(f"   Second file: {file_info['file2']} ({file_info['file2_size']} bytes)")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


def compare_analysis_files(file1_path: str, file2_path: str, output_report: bool = False) -> dict:
    """
    Convenience function to compare two analysis files.
    
    Args:
        file1_path: Path to first analysis file
        file2_path: Path to second analysis file  
        output_report: Whether to print a human-readable report
        
    Returns:
        Comparison result dictionary
    """
    comparator = AnalysisComparator()
    comparison = comparator.compare_files(file1_path, file2_path)
    
    if output_report and 'error' not in comparison:
        report = comparator.generate_comparison_report(comparison)
        print(report)
    
    return comparison