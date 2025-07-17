#!/usr/bin/env python3
"""
Comprehensive analysis that includes all ontology components
"""

import sys
import uuid
sys.path.insert(0, '.')

from analyzer.core.analyzer import DeterministicAnalyzer
from neo4j_client.client import Neo4jClient
import json
from pathlib import Path
import re

def generate_uuid():
    """Generate a short UUID for nodes"""
    return str(uuid.uuid4())[:8]

def comprehensive_analysis():
    """Complete analysis that includes all ontology components"""
    
    print("🔍 Comprehensive RealPyTest Analysis")
    print("=" * 60)
    
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
        
        # Clear existing data
        print("\n🧹 Clearing existing data...")
        neo4j_client = Neo4jClient()
        neo4j_client.execute_query("MATCH (n) WHERE NOT n:METADATA DETACH DELETE n")
        
        project_name = "RealPyTest"
        nodes = []
        relationships = []
        
        # 1. Create SYS node
        sys_uuid = generate_uuid()
        nodes.append({
            "uuid": sys_uuid,
            "type": "SYS",
            "Name": project_name,
            "Descr": "Hospital chatbot system with RAG agent and Neo4j integration"
        })
        
        # 2. Create UC (Use Case) nodes
        use_cases = [
            ("Hospital Query", "User queries hospital information through chatbot"),
            ("Wait Time Check", "Check current wait times at hospitals"),
            ("Data ETL", "Extract, transform, load hospital data from CSV"),
            ("Async Testing", "Test system with async requests"),
            ("Requirement Optimization", "Optimize requirements using ChatGPT")
        ]
        
        uc_nodes = {}
        for uc_name, uc_desc in use_cases:
            uc_uuid = generate_uuid()
            uc_nodes[uc_name] = uc_uuid
            nodes.append({
                "uuid": uc_uuid,
                "type": "UC",
                "Name": uc_name,
                "Descr": uc_desc
            })
            # System composes use cases
            relationships.append({
                "uuid": generate_uuid(),
                "type": "compose",
                "source": sys_uuid,
                "target": uc_uuid
            })
        
        # 3. Create ACTOR nodes from detected actors
        print("🎭 Creating ACTOR nodes...")
        actor_nodes = {}
        seen_actors = set()
        
        for actor_result in result.actor_results:
            for actor in actor_result.detected_actors:
                actor_name = actor.actor_type.value
                if actor_name not in seen_actors:
                    seen_actors.add(actor_name)
                    actor_uuid = generate_uuid()
                    actor_nodes[actor_name] = actor_uuid
                    nodes.append({
                        "uuid": actor_uuid,
                        "type": "ACTOR",
                        "Name": actor_name,
                        "Descr": f"Actor: {actor.pattern_name} - {actor.function_name or 'Unknown'}",
                        "Confidence": actor.confidence,
                        "Pattern": actor.pattern_name,
                        "Module": actor.module_name
                    })
        
        # 4. Create MOD nodes
        print("📦 Creating MOD nodes...")
        mod_nodes = {}
        for ast_result in result.ast_results:
            mod_uuid = generate_uuid()
            mod_nodes[ast_result.module_name] = mod_uuid
            nodes.append({
                "uuid": mod_uuid,
                "type": "MOD",
                "Name": ast_result.module_name,
                "Descr": f"Module: {ast_result.file_path}",
                "FilePath": str(ast_result.file_path)
            })
            # System composes modules
            relationships.append({
                "uuid": generate_uuid(),
                "type": "compose",
                "source": sys_uuid,
                "target": mod_uuid
            })
        
        # 5. Create FUNC nodes
        print("🔧 Creating FUNC nodes...")
        func_nodes = {}
        for ast_result in result.ast_results:
            mod_uuid = mod_nodes[ast_result.module_name]
            
            for func in ast_result.functions:
                func_uuid = generate_uuid()
                func_key = f"{ast_result.module_name}.{func.name}"
                func_nodes[func_key] = func_uuid
                
                nodes.append({
                    "uuid": func_uuid,
                    "type": "FUNC",
                    "Name": func.name,
                    "Descr": f"Function in {ast_result.module_name}: {func.docstring[:100] if func.docstring else 'No description'}",
                    "Module": ast_result.module_name,
                    "LineNumber": func.line_number
                })
                
                # Module composes functions
                relationships.append({
                    "uuid": generate_uuid(),
                    "type": "compose",
                    "source": mod_uuid,
                    "target": func_uuid
                })
        
        # 6. Create FCHAIN (Functional Chain) nodes
        print("⛓️  Creating FCHAIN nodes...")
        fchain_nodes = {}
        
        # Create functional chains based on module groupings
        chains = [
            ("Chatbot API Chain", ["chatbot_api/src/main", "chatbot_api/src/agents/hospital_rag_agent"], "Main chatbot processing chain"),
            ("Hospital Data Chain", ["hospital_neo4j_etl/src/hospital_bulk_csv_write"], "Hospital data ETL processing"),
            ("Wait Time Chain", ["chatbot_api/src/tools/wait_times"], "Hospital wait time processing"),
            ("Testing Chain", ["tests/async_agent_requests", "tests/sync_agent_requests"], "System testing workflows"),
            ("Requirement Chain", ["Reqopt"], "Requirement optimization workflow")
        ]
        
        for chain_name, chain_modules, chain_desc in chains:
            fchain_uuid = generate_uuid()
            fchain_nodes[chain_name] = fchain_uuid
            
            nodes.append({
                "uuid": fchain_uuid,
                "type": "FCHAIN",
                "Name": chain_name,
                "Descr": chain_desc
            })
            
            # System composes functional chains
            relationships.append({
                "uuid": generate_uuid(),
                "type": "compose",
                "source": sys_uuid,
                "target": fchain_uuid
            })
            
            # Functional chain composes relevant functions
            for module_name in chain_modules:
                for func_key, func_uuid in func_nodes.items():
                    if func_key.startswith(module_name):
                        relationships.append({
                            "uuid": generate_uuid(),
                            "type": "compose",
                            "source": fchain_uuid,
                            "target": func_uuid
                        })
        
        # 7. Create flow relationships between functions
        print("🌊 Creating flow relationships...")
        
        # Create flow relationships based on function calls and logical flow
        flow_patterns = [
            ("query_hospital_agent", "invoke_agent_with_retry", "Query triggers agent invocation"),
            ("get_current_wait_times", "_get_current_hospitals", "Wait time check requires hospital list"),
            ("get_current_wait_times", "_get_current_wait_time_minutes", "Wait time calculation"),
            ("load_hospital_graph_from_csv", "_set_uniqueness_constraints", "ETL process sets constraints first"),
            ("make_bulk_requests", "make_async_post", "Bulk requests use async posting")
        ]
        
        for source_func, target_func, flow_desc in flow_patterns:
            source_uuid = None
            target_uuid = None
            
            # Find source and target function UUIDs
            for func_key, func_uuid in func_nodes.items():
                if func_key.endswith(f".{source_func}"):
                    source_uuid = func_uuid
                elif func_key.endswith(f".{target_func}"):
                    target_uuid = func_uuid
            
            if source_uuid and target_uuid:
                relationships.append({
                    "uuid": generate_uuid(),
                    "type": "flow",
                    "source": source_uuid,
                    "target": target_uuid,
                    "FlowDescr": flow_desc,
                    "FlowDef": f"Flow from {source_func} to {target_func}"
                })
        
        # 8. Create SCHEMA nodes for data structures
        print("📋 Creating SCHEMA nodes...")
        
        # Infer schemas from CSV files and function signatures
        schemas = [
            ("Hospital", "Hospital entity with name, location, wait times", "{'name': 'string', 'location': 'string', 'wait_time': 'integer'}"),
            ("Patient", "Patient entity with demographics", "{'id': 'string', 'name': 'string', 'age': 'integer'}"),
            ("Visit", "Hospital visit record", "{'patient_id': 'string', 'hospital_id': 'string', 'timestamp': 'datetime'}"),
            ("ChatQuery", "User chat query structure", "{'query': 'string', 'context': 'object', 'response': 'string'}"),
            ("WaitTime", "Hospital wait time data", "{'hospital_id': 'string', 'wait_minutes': 'integer', 'updated_at': 'datetime'}")
        ]
        
        schema_nodes = {}
        for schema_name, schema_desc, schema_struct in schemas:
            schema_uuid = generate_uuid()
            schema_nodes[schema_name] = schema_uuid
            
            nodes.append({
                "uuid": schema_uuid,
                "type": "SCHEMA",
                "Name": schema_name,
                "Descr": schema_desc,
                "Struct": schema_struct
            })
            
            # System composes schemas
            relationships.append({
                "uuid": generate_uuid(),
                "type": "compose",
                "source": sys_uuid,
                "target": schema_uuid
            })
        
        # 9. Create REQ (Requirements) and TEST nodes
        print("✅ Creating REQ and TEST nodes...")
        
        requirements = [
            ("Hospital Query Performance", "System must respond to hospital queries within 2 seconds"),
            ("Data Accuracy", "Hospital data must be accurate and up-to-date"),
            ("Async Processing", "System must handle multiple concurrent requests"),
            ("Error Handling", "System must gracefully handle errors and timeouts"),
            ("CSV Data Loading", "System must load hospital data from CSV files")
        ]
        
        req_nodes = {}
        for req_name, req_desc in requirements:
            req_uuid = generate_uuid()
            req_nodes[req_name] = req_uuid
            
            nodes.append({
                "uuid": req_uuid,
                "type": "REQ",
                "Name": req_name,
                "Descr": req_desc
            })
            
            # Use cases satisfy requirements
            if "Query" in req_name:
                relationships.append({
                    "uuid": generate_uuid(),
                    "type": "satisfy",
                    "source": uc_nodes["Hospital Query"],
                    "target": req_uuid
                })
            elif "Async" in req_name:
                relationships.append({
                    "uuid": generate_uuid(),
                    "type": "satisfy",
                    "source": uc_nodes["Async Testing"],
                    "target": req_uuid
                })
        
        # Create TEST nodes
        test_files = ["tests/async_agent_requests", "tests/sync_agent_requests"]
        for test_file in test_files:
            if test_file in mod_nodes:
                test_uuid = generate_uuid()
                nodes.append({
                    "uuid": test_uuid,
                    "type": "TEST",
                    "Name": f"Test_{test_file.split('/')[-1]}",
                    "Descr": f"Test module: {test_file}"
                })
                
                # Tests verify requirements
                for req_name, req_uuid in req_nodes.items():
                    if "Async" in req_name and "async" in test_file:
                        relationships.append({
                            "uuid": generate_uuid(),
                            "type": "verify",
                            "source": req_uuid,
                            "target": test_uuid
                        })
        
        # 10. Create allocate relationships (actors to use cases)
        print("🎯 Creating allocate relationships...")
        
        # Allocate actors to use cases
        actor_uc_mappings = [
            ("User", "Hospital Query"),
            ("Administrator", "Data ETL"),
            ("Developer", "Async Testing"),
            ("SystemAnalyst", "Requirement Optimization")
        ]
        
        for actor_name, uc_name in actor_uc_mappings:
            if actor_name in actor_nodes and uc_name in uc_nodes:
                relationships.append({
                    "uuid": generate_uuid(),
                    "type": "allocate",
                    "source": actor_nodes[actor_name],
                    "target": uc_nodes[uc_name]
                })
        
        # Create final graph data
        graph_data = {
            "nodes": nodes,
            "relationships": relationships,
            "metadata": {
                "project_name": project_name,
                "total_nodes": len(nodes),
                "total_relationships": len(relationships),
                "ontology_version": "1.1.0",
                "analysis_type": "comprehensive"
            }
        }
        
        print(f"\n🔗 Generated Comprehensive Graph:")
        print(f"   📊 Total Nodes: {len(nodes)}")
        print(f"   🔗 Total Relationships: {len(relationships)}")
        
        # Count nodes by type
        node_counts = {}
        for node in nodes:
            node_type = node["type"]
            node_counts[node_type] = node_counts.get(node_type, 0) + 1
        
        print(f"   📋 Node Types:")
        for node_type, count in sorted(node_counts.items()):
            print(f"      {node_type}: {count}")
        
        # Count relationships by type
        rel_counts = {}
        for rel in relationships:
            rel_type = rel["type"]
            rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1
        
        print(f"   🔗 Relationship Types:")
        for rel_type, count in sorted(rel_counts.items()):
            print(f"      {rel_type}: {count}")
        
        # Store in Neo4j
        print("\n💾 Storing in Neo4j...")
        success = neo4j_client.store_graph(graph_data)
        
        if success:
            print("✅ Successfully stored comprehensive graph in Neo4j")
            
            # Verify storage
            summary = neo4j_client.get_architecture_summary()
            print(f"✅ Verified: {summary['total_nodes']} nodes, {summary['total_relationships']} relationships")
        else:
            print("❌ Failed to store in Neo4j")
        
        # Save to file
        output_file = "comprehensive_realpytest_architecture.json"
        with open(output_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        print(f"📄 Saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    comprehensive_analysis()