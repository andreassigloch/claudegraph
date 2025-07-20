#!/usr/bin/env python3
"""
Simple analysis that bypasses the graph generator issues
"""

import sys
sys.path.insert(0, '.')

from analyzer.core.analyzer import DeterministicAnalyzer
from neo4j_client.client import Neo4jClient
import json
from pathlib import Path

def simple_analysis():
    """Simple analysis that creates basic nodes manually"""
    
    print("🔍 Simple RealPyTest Analysis")
    print("=" * 50)
    
    try:
        # Initialize analyzer
        analyzer = DeterministicAnalyzer()
        
        # Analyze project
        result = analyzer.analyze("/Users/andreas/Documents/Tools/Eclipse_workspace/RealPyTest")
        
        print(f"📊 Analysis Stats:")
        print(f"   📁 Files discovered: {result.stats.files_discovered}")
        print(f"   📄 Files parsed: {result.stats.files_parsed}")
        print(f"   🔧 Functions found: {result.stats.functions_found}")
        print(f"   📦 Classes found: {result.stats.classes_found}")
        print(f"   🎭 Actors detected: {result.stats.actors_detected}")
        
        # Manually create basic graph structure
        project_name = "RealPyTest"
        
        # Create basic nodes
        nodes = [
            {
                "uuid": f"sys-{project_name.lower()}",
                "type": "SYS",
                "Name": project_name,
                "Descr": "Real Python test project with chatbot and hospital data"
            }
        ]
        
        # Add module nodes from AST results
        for i, ast_result in enumerate(result.ast_results):
            nodes.append({
                "uuid": f"mod-{ast_result.module_name}",
                "type": "MOD", 
                "Name": ast_result.module_name,
                "Descr": f"Module: {ast_result.file_path}",
                "FilePath": str(ast_result.file_path)
            })
            
            # Add function nodes
            for func in ast_result.functions:
                nodes.append({
                    "uuid": f"func-{func.name}-{i}",
                    "type": "FUNC",
                    "Name": func.name,
                    "Descr": f"Function in {ast_result.module_name}",
                    "Module": ast_result.module_name
                })
        
        # Create relationships
        relationships = []
        
        # System -> Module relationships
        for i, ast_result in enumerate(result.ast_results):
            relationships.append({
                "uuid": f"rel-sys-mod-{i}",
                "type": "compose",
                "source": f"sys-{project_name.lower()}",
                "target": f"mod-{ast_result.module_name}"
            })
            
            # Module -> Function relationships
            for j, func in enumerate(ast_result.functions):
                relationships.append({
                    "uuid": f"rel-mod-func-{i}-{j}",
                    "type": "compose",
                    "source": f"mod-{ast_result.module_name}",
                    "target": f"func-{func.name}-{i}"
                })
        
        # Create graph data
        graph_data = {
            "nodes": nodes,
            "relationships": relationships,
            "metadata": {
                "project_name": project_name,
                "total_nodes": len(nodes),
                "total_relationships": len(relationships),
                "analysis_stats": result.stats.__dict__
            }
        }
        
        print(f"\n🔗 Generated Graph:")
        print(f"   📊 Nodes: {len(nodes)}")
        print(f"   🔗 Relationships: {len(relationships)}")
        
        # Store in Neo4j
        print("\n💾 Storing in Neo4j...")
        neo4j_client = Neo4jClient()
        success = neo4j_client.store_graph(graph_data)
        
        if success:
            print("✅ Successfully stored in Neo4j")
            
            # Test a query
            systems = neo4j_client.execute_query("MATCH (s:SYS) WHERE s.Name = 'RealPyTest' RETURN s.Name as name")
            if systems:
                print(f"🎯 Found system: {systems[0]['name']}")
        else:
            print("❌ Failed to store in Neo4j")
        
        # Save to file
        output_file = "realpytest_architecture.json"
        with open(output_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        print(f"📄 Saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    simple_analysis()