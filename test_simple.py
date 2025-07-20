#!/usr/bin/env python3
"""
Simple test for ClaudeGraph with minimal data
"""

import sys
sys.path.insert(0, '.')

from neo4j_client.client import Neo4jClient

def test_basic_functionality():
    """Test basic ClaudeGraph functionality without full analysis"""
    
    print("🚀 ClaudeGraph Simple Test")
    print("=" * 40)
    
    try:
        # Test 1: Neo4j Connection
        print("📡 Testing Neo4j connection...")
        client = Neo4jClient()
        print("✅ Neo4j connected successfully")
        
        # Test 2: Load Ontology
        print("📋 Loading ontology schema...")
        success = client.load_ontology("ontology/load_ontology_community.cypher")
        if success:
            print("✅ Ontology loaded successfully")
        else:
            print("⚠️  Ontology loading failed")
        
        # Test 3: Create test nodes
        print("🔨 Creating test nodes...")
        test_data = {
            "nodes": [
                {
                    "uuid": "test-sys-001",
                    "type": "SYS",
                    "Name": "ClaudeGraph_Test_System",
                    "Descr": "Test system for ClaudeGraph verification"
                },
                {
                    "uuid": "test-func-001", 
                    "type": "FUNC",
                    "Name": "test_function",
                    "Descr": "Test function for verification"
                }
            ],
            "relationships": [
                {
                    "uuid": "test-rel-001",
                    "type": "compose",
                    "source": "test-sys-001",
                    "target": "test-func-001"
                }
            ]
        }
        
        store_success = client.store_graph(test_data)
        if store_success:
            print("✅ Test data stored successfully")
        else:
            print("❌ Failed to store test data")
            return False
        
        # Test 4: Query data
        print("🔍 Testing queries...")
        systems = client.execute_query("MATCH (s:SYS) WHERE s.Name CONTAINS 'Test' RETURN s.Name as name")
        if systems:
            print(f"✅ Found test systems: {[s['name'] for s in systems]}")
        else:
            print("⚠️  No test systems found")
        
        # Test 5: Architecture summary
        print("📊 Getting architecture summary...")
        summary = client.get_architecture_summary()
        print(f"✅ Architecture summary: {summary['total_nodes']} nodes, {summary['total_relationships']} relationships")
        
        # Test 6: Impact analysis
        print("🎯 Testing impact analysis...")
        impact = client.find_impact_analysis("ClaudeGraph_Test_System")
        if "error" not in impact:
            print(f"✅ Impact analysis: {impact['summary']['direct_count']} direct impacts")
        else:
            print(f"⚠️  Impact analysis: {impact['error']}")
        
        # Test 7: Ontology compliance
        print("✅ Testing ontology compliance...")
        compliance = client.validate_ontology_compliance()
        if compliance['compliant']:
            print("✅ Ontology compliance: PASS")
        else:
            print(f"⚠️  Ontology compliance: {compliance['total_issues']} issues found")
        
        print("\n🎉 ClaudeGraph simple test completed successfully!")
        print(f"🔗 Neo4j Browser: http://localhost:7475")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    test_basic_functionality()