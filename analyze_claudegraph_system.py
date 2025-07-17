#!/usr/bin/env python3
"""
Analyze the ClaudeGraph system itself with complete ontology
"""

import sys
import uuid
sys.path.insert(0, '.')

from analyzer.core.analyzer import DeterministicAnalyzer
from neo4j_client.client import Neo4jClient
import json
from pathlib import Path

def generate_uuid():
    """Generate a short UUID for nodes"""
    return str(uuid.uuid4())[:8]

def analyze_claudegraph_system():
    """Analyze the ClaudeGraph system with complete ontology"""
    
    print("🔍 ANALYZING CLAUDEGRAPH SYSTEM")
    print("=" * 50)
    
    try:
        # Initialize analyzer
        analyzer = DeterministicAnalyzer()
        
        # Analyze ClaudeGraph project
        result = analyzer.analyze("/Users/andreas/Documents/Projekte/ClaudeGraph")
        
        print(f"📊 ClaudeGraph Analysis Stats:")
        print(f"   📁 Files discovered: {result.stats.files_discovered}")
        print(f"   📄 Files parsed: {result.stats.files_parsed}")
        print(f"   🔧 Functions found: {result.stats.functions_found}")
        print(f"   📦 Classes found: {result.stats.classes_found}")
        print(f"   🎭 Actors detected: {result.stats.actors_detected}")
        
        # Clear existing ClaudeGraph data
        print("\n🧹 Clearing existing ClaudeGraph data...")
        neo4j_client = Neo4jClient()
        neo4j_client.execute_query("MATCH (n) WHERE n.Name CONTAINS 'ClaudeGraph' OR n.Name CONTAINS 'GrphArchitect' DETACH DELETE n")
        
        project_name = "ClaudeGraph"
        nodes = []
        relationships = []
        
        # 1. Create SYS node
        sys_uuid = generate_uuid()
        nodes.append({
            "uuid": sys_uuid,
            "type": "SYS",
            "Name": project_name,
            "Descr": "Graph-based architecture intelligence system for Claude Code"
        })
        
        # 2. Create UC (Use Case) nodes for ClaudeGraph
        use_cases = [
            ("Architecture Analysis", "Extract and analyze code architecture from projects"),
            ("Graph Storage", "Store architecture data in Neo4j graph database"),
            ("Impact Analysis", "Analyze impact of code changes on architecture"),
            ("Ontology Validation", "Validate architecture against ontology rules"),
            ("Query Interface", "Query architecture using Cypher or patterns"),
            ("Command Integration", "Integrate with Claude Code via /command interface")
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
        
        # 6. Create FCHAIN (Functional Chain) nodes for ClaudeGraph
        print("⛓️  Creating FCHAIN nodes...")
        fchain_nodes = {}
        
        chains = [
            ("Analysis Chain", ["analyzer.core.analyzer", "analyzer.core.project_discoverer", "analyzer.core.ast_parser"], "Code analysis processing chain"),
            ("Graph Chain", ["analyzer.graph.builder", "analyzer.graph.node_factory", "analyzer.graph.relationship_builder"], "Graph building processing chain"),
            ("Neo4j Chain", ["neo4j_client.client"], "Neo4j database operations chain"),
            ("Command Chain", ["commands.grph_architect"], "Command interface processing chain"),
            ("Detection Chain", ["analyzer.detection.pattern_matcher"], "Actor detection processing chain")
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
        
        # 7. Create flow relationships for ClaudeGraph
        print("🌊 Creating flow relationships...")
        
        flow_patterns = [
            ("analyze", "discover_project", "Analysis starts with project discovery"),
            ("analyze", "parse_file", "Analysis includes AST parsing"),
            ("analyze", "detect_actors", "Analysis includes actor detection"),
            ("analyze", "generate_graph", "Analysis generates graph data"),
            ("store_graph", "_create_node", "Graph storage creates nodes"),
            ("store_graph", "_create_relationship", "Graph storage creates relationships"),
            ("execute", "execute_query", "Commands execute Neo4j queries"),
            ("run", "execute", "CLI runs command execution")
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
        
        # 8. Create SCHEMA nodes for ClaudeGraph data structures
        print("📋 Creating SCHEMA nodes...")
        
        schemas = [
            ("AnalysisResult", "Complete analysis result with stats and graph data", "{'stats': 'AnalysisStats', 'graph_data': 'GraphData', 'errors': 'List[str]'}"),
            ("GraphData", "Graph nodes and relationships", "{'nodes': 'List[OntologyNode]', 'relationships': 'List[OntologyRelationship]'}"),
            ("OntologyNode", "Node in ontology graph", "{'uuid': 'str', 'type': 'str', 'properties': 'Dict[str, Any]'}"),
            ("DetectionMatch", "Actor detection match result", "{'actor_type': 'ActorType', 'confidence': 'float', 'pattern_name': 'str'}"),
            ("CommandResult", "Command execution result", "{'status': 'str', 'results': 'List[Dict]', 'message': 'str'}")
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
            
            # System composes schemas initially (will be moved to functions)
            relationships.append({
                "uuid": generate_uuid(),
                "type": "compose",
                "source": sys_uuid,
                "target": schema_uuid
            })
        
        # 9. Create REQ (Requirements) for ClaudeGraph
        print("✅ Creating REQ nodes...")
        
        requirements = [
            ("Ontology Compliance", "System must generate graphs compliant with ontology v1.1.0"),
            ("Performance", "System must analyze projects with <100 files in under 30 seconds"),
            ("Accuracy", "System must detect actors with >80% confidence"),
            ("Integration", "System must integrate with Claude Code via /command interface"),
            ("Persistence", "System must store graphs in Neo4j with ACID properties")
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
            if "Ontology" in req_name:
                relationships.append({
                    "uuid": generate_uuid(),
                    "type": "satisfy",
                    "source": uc_nodes["Ontology Validation"],
                    "target": req_uuid
                })
            elif "Performance" in req_name:
                relationships.append({
                    "uuid": generate_uuid(),
                    "type": "satisfy",
                    "source": uc_nodes["Architecture Analysis"],
                    "target": req_uuid
                })
        
        # 10. Create TEST nodes
        print("🧪 Creating TEST nodes...")
        
        test_modules = ["test_claudegraph", "test_simple", "debug_analysis"]
        for test_module in test_modules:
            test_uuid = generate_uuid()
            nodes.append({
                "uuid": test_uuid,
                "type": "TEST",
                "Name": f"Test_{test_module}",
                "Descr": f"Test module: {test_module}"
            })
            
            # Tests verify requirements
            for req_name, req_uuid in req_nodes.items():
                if "Performance" in req_name:
                    relationships.append({
                        "uuid": generate_uuid(),
                        "type": "verify",
                        "source": req_uuid,
                        "target": test_uuid
                    })
        
        # 11. Create allocate relationships
        print("🎯 Creating allocate relationships...")
        
        actor_uc_mappings = [
            ("Database", "Graph Storage"),
            ("FileSystem", "Architecture Analysis"),
            ("HttpClient", "Command Integration")
        ]
        
        for actor_name, uc_name in actor_uc_mappings:
            if actor_name in actor_nodes and uc_name in uc_nodes:
                relationships.append({
                    "uuid": generate_uuid(),
                    "type": "allocate",
                    "source": actor_nodes[actor_name],
                    "target": uc_nodes[uc_name]
                })
        
        # 12. Create function-actor relationships
        print("🔗 Creating function-actor relationships...")
        
        function_actor_mappings = [
            ("store_graph", "Database", "Function stores data in Neo4j"),
            ("execute_query", "Database", "Function queries Neo4j"),
            ("discover_project", "FileSystem", "Function reads project files"),
            ("parse_file", "FileSystem", "Function parses source files"),
            ("analyze", "FileSystem", "Function analyzes file structure")
        ]
        
        for func_name, actor_name, description in function_actor_mappings:
            if actor_name in actor_nodes:
                actor_uuid = actor_nodes[actor_name]
                
                # Find matching functions
                for func_key, func_uuid in func_nodes.items():
                    if func_key.endswith(f".{func_name}"):
                        relationships.append({
                            "uuid": generate_uuid(),
                            "type": "relation",
                            "source": func_uuid,
                            "target": actor_uuid,
                            "Description": description
                        })
        
        # 13. Create function-schema relationships
        print("📋 Creating function-schema relationships...")
        
        function_schema_mappings = [
            ("analyze", "AnalysisResult", "Function returns analysis results"),
            ("generate_graph", "GraphData", "Function generates graph data"),
            ("detect_actors", "DetectionMatch", "Function detects actor matches"),
            ("execute", "CommandResult", "Function returns command results"),
            ("_create_node", "OntologyNode", "Function creates ontology nodes")
        ]
        
        # Remove system-schema relationships first
        relationships = [r for r in relationships if not (r.get("source") == sys_uuid and any(r.get("target") == s_uuid for s_uuid in schema_nodes.values()))]
        
        for func_name, schema_name, description in function_schema_mappings:
            if schema_name in schema_nodes:
                schema_uuid = schema_nodes[schema_name]
                
                # Find matching functions
                for func_key, func_uuid in func_nodes.items():
                    if func_key.endswith(f".{func_name}"):
                        relationships.append({
                            "uuid": generate_uuid(),
                            "type": "relation",
                            "source": func_uuid,
                            "target": schema_uuid,
                            "Description": description
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
                "analysis_type": "comprehensive_self_analysis"
            }
        }
        
        print(f"\n🔗 ClaudeGraph System Architecture:")
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
        
        # Store in Neo4j
        print("\n💾 Storing ClaudeGraph architecture in Neo4j...")
        success = neo4j_client.store_graph(graph_data)
        
        if success:
            print("✅ Successfully stored ClaudeGraph architecture in Neo4j")
            
            # Verify storage
            summary = neo4j_client.get_architecture_summary()
            print(f"✅ Total in database: {summary['total_nodes']} nodes, {summary['total_relationships']} relationships")
        else:
            print("❌ Failed to store in Neo4j")
        
        # Save to file
        output_file = "claudegraph_self_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        print(f"📄 Saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ ClaudeGraph self-analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    analyze_claudegraph_system()