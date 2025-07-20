#!/usr/bin/env python3
"""
Test script for ClaudeGraph system
"""

import sys
import json
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, '.')

from analyzer.core.analyzer import DeterministicAnalyzer
from analyzer.graph.builder import OntologyGraphBuilder
from neo4j_client.client import Neo4jClient

def test_claudegraph_analysis(project_path: str):
    """Test ClaudeGraph analysis on a project"""
    
    print(f"🔍 Testing ClaudeGraph analysis on: {project_path}")
    
    try:
        # Stage 1: Initialize analyzer
        print("📦 Initializing DeterministicAnalyzer...")
        analyzer = DeterministicAnalyzer()
        print("✅ DeterministicAnalyzer initialized")
        
        # Stage 2: Analyze project
        print("🔬 Analyzing project structure...")
        result = analyzer.analyze(project_path)
        
        if result.errors:
            print(f"⚠️  Analysis completed with {len(result.errors)} errors:")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"   - {error}")
        else:
            print("✅ Analysis completed without errors")
        
        # Display statistics
        stats = result.stats
        print(f"\n📊 Analysis Statistics:")
        print(f"   📁 Files discovered: {stats.files_discovered}")
        print(f"   📄 Files parsed: {stats.files_parsed}")
        print(f"   🔧 Functions found: {stats.functions_found}")
        print(f"   📦 Classes found: {stats.classes_found}")
        print(f"   🎭 Actors detected: {stats.actors_detected}")
        print(f"   📊 Nodes generated: {stats.nodes_generated}")
        print(f"   🔗 Relationships: {stats.relationships_generated}")
        print(f"   ⏱️  Analysis time: {stats.analysis_time_seconds:.2f}s")
        
        # Stage 3: Build graph (if we have data)
        if hasattr(result, 'graph_data') and result.graph_data:
            print("\n🔗 Building architecture graph...")
            graph_builder = OntologyGraphBuilder()
            graph_json = graph_builder.build_graph(result)
            
            print(f"✅ Graph built with {len(graph_json.get('nodes', []))} nodes")
            
            # Stage 4: Store in Neo4j
            print("💾 Storing in Neo4j...")
            neo4j_client = Neo4jClient()
            success = neo4j_client.store_graph(graph_json)
            
            if success:
                print("✅ Successfully stored in Neo4j")
                
                # Test a simple query
                systems = neo4j_client.execute_query("MATCH (s:SYS) RETURN s.Name as name LIMIT 5")
                if systems:
                    print(f"🎯 Found systems: {[s['name'] for s in systems]}")
            else:
                print("❌ Failed to store in Neo4j")
            
            # Save to file for inspection
            output_file = "claudegraph_test_result.json"
            with open(output_file, 'w') as f:
                json.dump(graph_json, f, indent=2)
            print(f"📄 Results saved to: {output_file}")
            
        else:
            print("⚠️  No graph data was generated")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 ClaudeGraph System Test")
    print("=" * 50)
    
    # Test with AiSE_Test project
    aise_test_path = "/Users/andreas/Documents/Tools/VSCode/AssistantTestClaude"
    
    if not Path(aise_test_path).exists():
        print(f"❌ Test project not found: {aise_test_path}")
        return
    
    success = test_claudegraph_analysis(aise_test_path)
    
    if success:
        print("\n🎉 ClaudeGraph test completed successfully!")
        print("\n🔗 Next steps:")
        print("   • Access Neo4j Browser: http://localhost:7475")
        print("   • Try architecture queries via Neo4j client")
        print("   • Test /command GrphArchitect integration")
    else:
        print("\n❌ ClaudeGraph test failed")

if __name__ == "__main__":
    main()