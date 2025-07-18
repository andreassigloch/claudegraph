#!/usr/bin/env python3
"""
GrphArchitect Command Implementation
Claude Code command for graph-based architecture intelligence
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analyzer.core.analyzer import DeterministicAnalyzer
from analyzer.graph.builder import OntologyGraphBuilder
from neo4j_client.client import Neo4jClient


@dataclass
class ArchitectureCommand:
    """Base class for architecture commands"""
    name: str
    description: str
    
    def execute(self, args: argparse.Namespace) -> Dict[str, Any]:
        raise NotImplementedError


class DesignCommand(ArchitectureCommand):
    """Interactive architecture design for new projects"""
    
    def __init__(self):
        super().__init__("design", "Create architecture design for new project")
    
    def execute(self, args: argparse.Namespace) -> Dict[str, Any]:
        print("🎨 Starting Architecture Design...")
        print("This will guide you through creating a logical architecture following ontology v1.1.0")
        print()
        
        # Interactive design workflow
        project_name = input("Project name (PascalCase, max 25 chars): ").strip()
        if not project_name or len(project_name) > 25:
            return {"error": "Project name required (max 25 characters)"}
        
        description = input("Project description: ").strip()
        
        # Create basic system structure
        nodes = []
        relationships = []
        node_counter = 0
        
        # System node
        sys_uuid = f"sys-{project_name.lower()}-{node_counter}"
        node_counter += 1
        nodes.append({
            "uuid": sys_uuid,
            "type": "SYS",
            "Name": project_name,
            "Descr": description or f"{project_name} system"
        })
        
        print(f"\n✅ Created system: {project_name}")
        
        # Interactive UC creation
        print("\n📋 Define Use Cases (UC) - Enter empty name to finish")
        while True:
            uc_name = input("  Use case name (or press Enter to finish): ").strip()
            if not uc_name:
                break
            if len(uc_name) > 25:
                print("  ⚠️  Name too long (max 25 chars)")
                continue
                
            uc_desc = input(f"  Description for '{uc_name}': ").strip()
            uc_uuid = f"uc-{uc_name.lower()}-{node_counter}"
            node_counter += 1
            
            nodes.append({
                "uuid": uc_uuid,
                "type": "UC",
                "Name": uc_name,
                "Descr": uc_desc or f"{uc_name} use case"
            })
            
            # Connect UC to system
            relationships.append({
                "uuid": f"rel-sys-uc-{node_counter}",
                "type": "compose",
                "source": sys_uuid,
                "target": uc_uuid
            })
            
            print(f"  ✅ Added UC: {uc_name}")
        
        # Interactive ACTOR creation
        print("\n👥 Define Actors (ACTOR) - External entities that interact with the system")
        while True:
            actor_name = input("  Actor name (or press Enter to finish): ").strip()
            if not actor_name:
                break
            if len(actor_name) > 25:
                print("  ⚠️  Name too long (max 25 chars)")
                continue
                
            actor_desc = input(f"  Description for '{actor_name}': ").strip()
            actor_uuid = f"actor-{actor_name.lower()}-{node_counter}"
            node_counter += 1
            
            nodes.append({
                "uuid": actor_uuid,
                "type": "ACTOR",
                "Name": actor_name,
                "Descr": actor_desc or f"{actor_name} external actor"
            })
            
            # CRITICAL: Connect ACTOR to system via compose relationship
            relationships.append({
                "uuid": f"rel-sys-actor-{node_counter}",
                "type": "compose", 
                "source": sys_uuid,
                "target": actor_uuid
            })
            
            print(f"  ✅ Added ACTOR: {actor_name}")
        
        # Interactive SCHEMA creation
        print("\n📋 Define Schemas (SCHEMA) - Data structures and interfaces")
        while True:
            schema_name = input("  Schema name (or press Enter to finish): ").strip()
            if not schema_name:
                break
            if len(schema_name) > 25:
                print("  ⚠️  Name too long (max 25 chars)")
                continue
                
            schema_desc = input(f"  Description for '{schema_name}': ").strip()
            schema_struct = input(f"  Structure (JSON format) for '{schema_name}': ").strip()
            schema_uuid = f"schema-{schema_name.lower()}-{node_counter}"
            node_counter += 1
            
            nodes.append({
                "uuid": schema_uuid,
                "type": "SCHEMA",
                "Name": schema_name,
                "Descr": schema_desc or f"{schema_name} data structure",
                "Struct": schema_struct or f'{{"placeholder": "define structure for {schema_name}"}}'
            })
            
            # CRITICAL: Connect SCHEMA to system via compose relationship
            relationships.append({
                "uuid": f"rel-sys-schema-{node_counter}",
                "type": "compose",
                "source": sys_uuid,
                "target": schema_uuid
            })
            
            print(f"  ✅ Added SCHEMA: {schema_name}")
        
        # Interactive FCHAIN creation
        print("\n🔗 Define Function Chains (FCHAIN) - Logical function sequences")
        uc_names = [n["Name"] for n in nodes if n["type"] == "UC"]
        if uc_names:
            print(f"  Available UCs: {', '.join(uc_names)}")
        
        while True:
            fchain_name = input("  Function chain name (or press Enter to finish): ").strip()
            if not fchain_name:
                break
            if len(fchain_name) > 25:
                print("  ⚠️  Name too long (max 25 chars)")
                continue
                
            fchain_desc = input(f"  Description for '{fchain_name}': ").strip()
            fchain_uuid = f"fchain-{fchain_name.lower()}-{node_counter}"
            node_counter += 1
            
            nodes.append({
                "uuid": fchain_uuid,
                "type": "FCHAIN",
                "Name": fchain_name,
                "Descr": fchain_desc or f"{fchain_name} function chain"
            })
            
            # Connect FCHAIN to UC if specified
            if uc_names:
                uc_choice = input(f"  Connect to UC (from: {', '.join(uc_names)}) or Enter to skip: ").strip()
                if uc_choice and uc_choice in uc_names:
                    uc_node = next(n for n in nodes if n["Name"] == uc_choice and n["type"] == "UC")
                    relationships.append({
                        "uuid": f"rel-uc-fchain-{node_counter}",
                        "type": "compose",
                        "source": uc_node["uuid"],
                        "target": fchain_uuid
                    })
                    print(f"    → Connected to UC: {uc_choice}")
            
            print(f"  ✅ Added FCHAIN: {fchain_name}")
        
        architecture = {
            "nodes": nodes,
            "relationships": relationships,
            "metadata": {
                "created_by": "ClaudeGraph",
                "created_at": str(Path().cwd()),
                "version": "1.0.0",
                "ontology_version": "1.1.0",
                "design_mode": "interactive"
            }
        }
        
        # Save to file
        output_path = Path(args.output) if args.output else Path(f"{project_name}_architecture.json")
        with open(output_path, 'w') as f:
            json.dump(architecture, f, indent=2)
        
        # Store in Neo4j if available
        try:
            neo4j_client = Neo4jClient()
            neo4j_client.store_graph(architecture)
            print(f"\n📊 Architecture stored in Neo4j database")
        except Exception as e:
            print(f"\n⚠️  Could not store in Neo4j: {e}")
        
        print(f"\n🎉 Architecture design completed!")
        print(f"📁 Saved to: {output_path}")
        print(f"📊 Created {len(nodes)} nodes and {len(relationships)} relationships")
        
        return {
            "status": "success",
            "message": f"Architecture design created: {output_path}",
            "nodes": len(nodes),
            "relationships": len(relationships),
            "components": {
                "systems": len([n for n in nodes if n["type"] == "SYS"]),
                "use_cases": len([n for n in nodes if n["type"] == "UC"]),
                "actors": len([n for n in nodes if n["type"] == "ACTOR"]),
                "schemas": len([n for n in nodes if n["type"] == "SCHEMA"]),
                "function_chains": len([n for n in nodes if n["type"] == "FCHAIN"])
            }
        }


class AnalyzeCommand(ArchitectureCommand):
    """Analyze existing code to extract architecture"""
    
    def __init__(self):
        super().__init__("analyze", "Extract architecture from existing code")
    
    def execute(self, args: argparse.Namespace) -> Dict[str, Any]:
        print("🔍 Analyzing Code Architecture...")
        
        # Validate path
        target_path = Path(args.path)
        if not target_path.exists():
            return {"error": f"Path not found: {target_path}"}
        
        try:
            # Initialize analyzer
            analyzer = DeterministicAnalyzer()
            
            # Analyze project
            result = analyzer.analyze(str(target_path))
            
            # Check if analysis was successful
            if not result.is_successful():
                return {"error": f"Analysis failed: {len(result.errors)} errors found"}
            
            # Build graph data directly from result
            graph_data = {
                "nodes": [node.__dict__ for node in result.graph_data.nodes] if result.graph_data else [],
                "relationships": [rel.__dict__ for rel in result.graph_data.relationships] if result.graph_data else [],
                "metadata": result.metadata
            }
            
            # Store in Neo4j if configured
            if args.store_neo4j:
                neo4j_client = Neo4jClient()
                neo4j_client.store_graph(graph_data)
                print("📊 Stored in Neo4j database")
            
            # Save to file
            output_path = Path(args.output) if args.output else Path("architecture_analysis.json")
            with open(output_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
            
            return {
                "status": "success",
                "message": f"Architecture analyzed: {output_path}",
                "nodes": len(graph_data.get("nodes", [])),
                "relationships": len(graph_data.get("relationships", [])),
                "path": str(target_path)
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}


class QueryCommand(ArchitectureCommand):
    """Query architecture using Cypher or patterns"""
    
    def __init__(self):
        super().__init__("query", "Query architecture graph")
    
    def execute(self, args: argparse.Namespace) -> Dict[str, Any]:
        print("🔎 Querying Architecture...")
        
        try:
            neo4j_client = Neo4jClient()
            
            # Handle different query types
            if args.cypher:
                # Direct Cypher query
                result = neo4j_client.execute_query(args.cypher)
            elif args.pattern:
                # Pattern-based query
                result = self._handle_pattern_query(args.pattern, neo4j_client)
            else:
                # Default: show system overview
                result = neo4j_client.execute_query("""
                    MATCH (s:SYS)
                    OPTIONAL MATCH (s)-[:compose*]->(n)
                    RETURN s.Name as system, 
                           labels(n) as node_types,
                           count(n) as count
                    ORDER BY system
                """)
            
            return {
                "status": "success",
                "results": result,
                "query_type": "cypher" if args.cypher else "pattern" if args.pattern else "overview"
            }
            
        except Exception as e:
            return {"error": f"Query failed: {str(e)}"}
    
    def _handle_pattern_query(self, pattern: str, client: Neo4jClient) -> List[Dict]:
        """Handle pattern-based queries - OPTIMIZED for high-level abstraction"""
        # OPTIMIZED: Focus on high-level components, limited results, efficient queries
        patterns = {
            "overview": """
                MATCH (s:SYS)
                OPTIONAL MATCH (s)-[:compose]->(uc:UC)
                OPTIONAL MATCH (s)-[:compose]->(a:ACTOR)
                OPTIONAL MATCH (s)-[:compose]->(fc:FCHAIN)
                OPTIONAL MATCH (s)-[:compose]->(schema:SCHEMA)
                RETURN s.Name as system, 
                       count(DISTINCT uc) as use_cases,
                       count(DISTINCT a) as actors,
                       count(DISTINCT fc) as chains,
                       count(DISTINCT schema) as schemas
                LIMIT 5
            """,
            "actors": """
                MATCH (s:SYS)-[:compose]->(a:ACTOR)
                RETURN a.Name as name, a.Descr as description
                ORDER BY a.Name LIMIT 10
            """,
            "chains": """
                MATCH (s:SYS)-[:compose]->(fc:FCHAIN)
                OPTIONAL MATCH (fc)-[:compose]->(f:FUNC)
                RETURN fc.Name as name, fc.Descr as description, count(f) as function_count
                ORDER BY function_count DESC LIMIT 10
            """,
            "flows": """
                MATCH (f1:FUNC)-[r:flow]->(f2:FUNC)
                RETURN f1.Name + ' → ' + f2.Name as flow, r.FlowDescr as description
                ORDER BY f1.Name LIMIT 10
            """,
            "schemas": """
                MATCH (s:SYS)-[:compose]->(schema:SCHEMA)
                RETURN schema.Name as name, schema.Descr as description, schema.Struct as structure
                ORDER BY schema.Name LIMIT 10
            """,
            "use_cases": """
                MATCH (s:SYS)-[:compose]->(uc:UC)
                OPTIONAL MATCH (uc)-[:satisfy]->(req:REQ)
                RETURN uc.Name as name, uc.Descr as description, count(req) as requirements
                ORDER BY requirements DESC LIMIT 10
            """
        }
        
        query = patterns.get(pattern.lower())
        if not query:
            available = ", ".join(patterns.keys())
            raise ValueError(f"Unknown pattern: {pattern}. Available: {available}")
        
        return client.execute_query(query)


class ImpactCommand(ArchitectureCommand):
    """Analyze impact of changing a component"""
    
    def __init__(self):
        super().__init__("impact", "Analyze change impact")
    
    def execute(self, args: argparse.Namespace) -> Dict[str, Any]:
        print(f"📈 Analyzing Impact of: {args.component}")
        
        try:
            neo4j_client = Neo4jClient()
            
            # OPTIMIZED: Single consolidated query for complete impact analysis
            # Focus on high-level abstraction only (UC, ACTOR, FCHAIN, SCHEMA)
            optimized_query = """
                WITH $component as target_name
                
                // Find target component
                MATCH (target) 
                WHERE target.Name CONTAINS target_name
                
                // Collect all relationships
                WITH target
                
                // Direct impacts - HIGH-LEVEL ONLY
                OPTIONAL MATCH (target)-[r1]-(direct)
                WHERE labels(direct)[0] IN ['UC', 'ACTOR', 'FCHAIN', 'SCHEMA', 'SYS']
                WITH target, collect(DISTINCT direct) as direct_impacts
                
                // Transitive impacts - 2 hops max, HIGH-LEVEL ONLY
                OPTIONAL MATCH (target)-[*2]-(transitive)
                WHERE NOT transitive = target 
                  AND labels(transitive)[0] IN ['UC', 'ACTOR', 'FCHAIN', 'SCHEMA', 'SYS']
                WITH target, direct_impacts, collect(DISTINCT transitive) as transitive_impacts
                
                // Affected requirements/tests - simplified
                OPTIONAL MATCH (target)-[*1..2]-(req:REQ)-[:verify]->(test:TEST)
                WITH target, direct_impacts, transitive_impacts, collect(DISTINCT test) as affected_tests
                
                RETURN {
                    component: target.Name,
                    type: labels(target)[0],
                    direct: [x IN direct_impacts WHERE x IS NOT NULL | {
                        name: x.Name, 
                        type: labels(x)[0]
                    }][0..5],
                    transitive: [x IN transitive_impacts WHERE x IS NOT NULL | {
                        name: x.Name, 
                        type: labels(x)[0]
                    }][0..5],
                    tests: [x IN affected_tests WHERE x IS NOT NULL | x.Name][0..3],
                    summary: {
                        direct_count: size([x IN direct_impacts WHERE x IS NOT NULL]),
                        transitive_count: size([x IN transitive_impacts WHERE x IS NOT NULL]),
                        test_count: size([x IN affected_tests WHERE x IS NOT NULL])
                    }
                } as impact
            """
            
            result = neo4j_client.execute_query(optimized_query, {"component": args.component})
            
            if result and result[0]["impact"]:
                impact = result[0]["impact"]
                return {
                    "status": "success",
                    "component": impact["component"],
                    "component_type": impact["type"],
                    "direct_impacts": impact["direct"],
                    "transitive_impacts": impact["transitive"],
                    "affected_tests": impact["tests"],
                    "impact_summary": impact["summary"]
                }
            else:
                # OPTIMIZED: Quick check for component existence
                exists_query = "MATCH (n) WHERE n.Name CONTAINS $component RETURN count(n) as count"
                exists_result = neo4j_client.execute_query(exists_query, {"component": args.component})
                
                if exists_result and exists_result[0]["count"] > 0:
                    return {
                        "status": "success",
                        "component": args.component,
                        "message": f"Component '{args.component}' exists but has no high-level architectural connections",
                        "direct_impacts": [],
                        "transitive_impacts": [],
                        "affected_tests": [],
                        "impact_summary": {"direct_count": 0, "transitive_count": 0, "test_count": 0}
                    }
                else:
                    return {
                        "status": "not_found",
                        "component": args.component,
                        "message": f"Component '{args.component}' not found in architecture database",
                        "suggestion": "Try: /claudegraph query --pattern overview to see available components",
                        "direct_impacts": [],
                        "transitive_impacts": [],
                        "affected_tests": [],
                        "impact_summary": {"direct_count": 0, "transitive_count": 0, "test_count": 0}
                    }
            
        except Exception as e:
            return {"error": f"Impact analysis failed: {str(e)}"}


class CheckCommand(ArchitectureCommand):
    """Check architectural consistency"""
    
    def __init__(self):
        super().__init__("check", "Check architectural consistency")
    
    def execute(self, args: argparse.Namespace) -> Dict[str, Any]:
        print("✅ Checking Architecture Consistency...")
        print("Validating ontology v1.1.0 compliance and architectural integrity...")
        
        try:
            neo4j_client = Neo4jClient()
            issues = []
            
            # CRITICAL Check 1: ACTOR nodes not connected to system via compose
            actors_not_in_system = neo4j_client.execute_query("""
                MATCH (a:ACTOR)
                WHERE NOT EXISTS((s:SYS)-[:compose]->(a))
                RETURN a.Name as actor_name, a.Descr as description
            """)
            
            if actors_not_in_system:
                issues.append({
                    "type": "actors_not_connected_to_system",
                    "severity": "error",
                    "count": len(actors_not_in_system),
                    "description": "ACTOR nodes must be connected to system via compose relationship",
                    "items": [f"{item['actor_name']}: {item['description']}" for item in actors_not_in_system],
                    "fix_suggestion": "Add compose relationships from SYS to each ACTOR node"
                })
            
            # CRITICAL Check 2: SCHEMA nodes not connected to system via compose
            schemas_not_in_system = neo4j_client.execute_query("""
                MATCH (s:SCHEMA)
                WHERE NOT EXISTS((sys:SYS)-[:compose]->(s))
                RETURN s.Name as schema_name, s.Descr as description
            """)
            
            if schemas_not_in_system:
                issues.append({
                    "type": "schemas_not_connected_to_system", 
                    "severity": "error",
                    "count": len(schemas_not_in_system),
                    "description": "SCHEMA nodes must be connected to system via compose relationship",
                    "items": [f"{item['schema_name']}: {item['description']}" for item in schemas_not_in_system],
                    "fix_suggestion": "Add compose relationships from SYS to each SCHEMA node"
                })
            
            # Check 3: Functions without requirements (ontology rule)
            functions_without_req = neo4j_client.execute_query("""
                MATCH (f:FUNC)
                WHERE NOT EXISTS((f)-[:satisfy]->(:REQ))
                RETURN f.Name as function_name, f.Descr as description
            """)
            
            if functions_without_req:
                issues.append({
                    "type": "missing_requirements",
                    "severity": "warning",
                    "count": len(functions_without_req),
                    "description": "Functions must have at least one requirement (ontology rule)",
                    "items": [f"{item['function_name']}: {item['description']}" for item in functions_without_req],
                    "fix_suggestion": "Create REQ nodes and satisfy relationships for each function"
                })
            
            # Check 4: Requirements without tests (ontology rule)
            requirements_without_tests = neo4j_client.execute_query("""
                MATCH (r:REQ)
                WHERE NOT EXISTS((t:TEST)-[:verify]->(r))
                RETURN r.Name as requirement_name, r.Descr as description
            """)
            
            if requirements_without_tests:
                issues.append({
                    "type": "missing_tests",
                    "severity": "error",
                    "count": len(requirements_without_tests),
                    "description": "Requirements must have at least one test (ontology rule)",
                    "items": [f"{item['requirement_name']}: {item['description']}" for item in requirements_without_tests],
                    "fix_suggestion": "Create TEST nodes and verify relationships for each requirement"
                })
            
            # Check 5: Isolated nodes (ontology rule)
            isolated_nodes = neo4j_client.execute_query("""
                MATCH (n)
                WHERE NOT EXISTS((n)--()) AND NOT labels(n)[0] = 'SYS'
                RETURN n.Name as node_name, labels(n)[0] as node_type, n.Descr as description
            """)
            
            if isolated_nodes:
                issues.append({
                    "type": "isolated_nodes",
                    "severity": "warning", 
                    "count": len(isolated_nodes),
                    "description": "All elements must have at least one link (ontology isolation rule)",
                    "items": [f"{item['node_name']} ({item['node_type']}): {item['description']}" for item in isolated_nodes],
                    "fix_suggestion": "Connect isolated nodes via appropriate relationships"
                })
            
            # Check 6: Functions without input/output (ontology rule)
            functions_without_io = neo4j_client.execute_query("""
                MATCH (f:FUNC)
                WHERE NOT EXISTS((f)-[:flow]-()) AND NOT EXISTS(()-[:flow]->(f))
                RETURN f.Name as function_name, f.Descr as description
            """)
            
            if functions_without_io:
                issues.append({
                    "type": "functions_without_flow",
                    "severity": "warning",
                    "count": len(functions_without_io),
                    "description": "Functions must have at least one input and output (ontology rule)",
                    "items": [f"{item['function_name']}: {item['description']}" for item in functions_without_io],
                    "fix_suggestion": "Add flow relationships to/from functions or connect to ACTOR nodes"
                })
            
            # Check 7: FCHAIN connectivity (ontology rule)
            fchains_with_isolated_functions = neo4j_client.execute_query("""
                MATCH (fc:FCHAIN)-[:compose]->(f:FUNC)
                WITH fc, collect(f) as functions
                WHERE size(functions) > 1
                WITH fc, functions
                UNWIND functions as f1
                UNWIND functions as f2
                WHERE f1 <> f2
                WITH fc, f1, f2, 
                     EXISTS((f1)-[:flow*]-(f2)) as connected
                WITH fc, count(DISTINCT f1) as func_count, 
                     count(CASE WHEN connected THEN 1 END) as connected_count
                WHERE connected_count < func_count * (func_count - 1) / 2
                RETURN fc.Name as fchain_name, func_count, connected_count
            """)
            
            if fchains_with_isolated_functions:
                issues.append({
                    "type": "fchain_connectivity_issues",
                    "severity": "error",
                    "count": len(fchains_with_isolated_functions),
                    "description": "All functions in FCHAIN must be connected via flow relationships",
                    "items": [f"{item['fchain_name']}: {item['func_count']} functions, {item['connected_count']} connections" for item in fchains_with_isolated_functions],
                    "fix_suggestion": "Add flow relationships between functions within each FCHAIN"
                })
            
            # Check 8: Leaf use cases without actors (ontology rule)
            leaf_uc_without_actors = neo4j_client.execute_query("""
                MATCH (uc:UC)
                WHERE NOT EXISTS((uc)-[:compose]->(:UC)) 
                  AND NOT EXISTS((uc)-[:compose]->(:ACTOR))
                RETURN uc.Name as usecase_name, uc.Descr as description
            """)
            
            if leaf_uc_without_actors:
                issues.append({
                    "type": "leaf_usecase_missing_actors",
                    "severity": "warning",
                    "count": len(leaf_uc_without_actors),
                    "description": "Leaf use cases must have at least one composed actor (ontology rule)",
                    "items": [f"{item['usecase_name']}: {item['description']}" for item in leaf_uc_without_actors],
                    "fix_suggestion": "Add compose relationships from leaf UCs to ACTOR nodes"
                })
            
            # Calculate overall health score based on severity
            error_count = sum(issue["count"] for issue in issues if issue["severity"] == "error")
            warning_count = sum(issue["count"] for issue in issues if issue["severity"] == "warning")
            total_issues = error_count + warning_count
            
            # Errors are weighted more heavily than warnings
            health_score = max(0, 100 - (error_count * 15) - (warning_count * 5))
            
            # Categorize architecture health
            if health_score >= 90:
                health_status = "Excellent"
            elif health_score >= 75:
                health_status = "Good"
            elif health_score >= 50:
                health_status = "Fair"
            else:
                health_status = "Poor"
            
            summary_message = f"Architecture health: {health_status} ({health_score}/100)"
            if total_issues > 0:
                summary_message += f" - Found {error_count} errors, {warning_count} warnings"
            else:
                summary_message += " - Architecture is fully compliant with ontology v1.1.0"
            
            return {
                "status": "success",
                "health_score": health_score,
                "health_status": health_status,
                "issues": issues,
                "total_issues": total_issues,
                "error_count": error_count,
                "warning_count": warning_count,
                "summary": summary_message,
                "ontology_version": "1.1.0"
            }
            
        except Exception as e:
            return {"error": f"Consistency check failed: {str(e)}"}


class ComplianceCommand(ArchitectureCommand):
    """Check compliance between logical architecture and actual code"""
    
    def __init__(self):
        super().__init__("compliance", "Check architecture-code compliance")
    
    def execute(self, args: argparse.Namespace) -> Dict[str, Any]:
        print("🔍 Checking Architecture-Code Compliance...")
        print("Comparing logical architecture (Neo4j) vs actual codebase...")
        
        if not args.path:
            return {"error": "Code path required for compliance checking"}
        
        code_path = Path(args.path)
        if not code_path.exists():
            return {"error": f"Code path not found: {code_path}"}
        
        try:
            neo4j_client = Neo4jClient()
            compliance_issues = []
            
            # Get logical architecture from Neo4j
            logical_functions = neo4j_client.execute_query("""
                MATCH (f:FUNC)
                OPTIONAL MATCH (f)-[:allocate]->(m:MOD)
                RETURN f.Name as function_name, f.Descr as description, 
                       m.Name as module_name, m.ModDef as module_file
            """)
            
            logical_actors = neo4j_client.execute_query("""
                MATCH (a:ACTOR)
                RETURN a.Name as actor_name, a.Descr as description
            """)
            
            logical_schemas = neo4j_client.execute_query("""
                MATCH (s:SCHEMA)
                RETURN s.Name as schema_name, s.Descr as description, s.Struct as structure
            """)
            
            logical_fchains = neo4j_client.execute_query("""
                MATCH (fc:FCHAIN)
                OPTIONAL MATCH (fc)-[:compose]->(f:FUNC)
                RETURN fc.Name as fchain_name, fc.Descr as description, 
                       collect(f.Name) as functions
            """)
            
            # Analyze actual codebase (simplified - reuse some grphzer components)
            print("  📁 Scanning codebase for Python files...")
            python_files = list(code_path.rglob("*.py"))
            code_functions = set()
            code_classes = set()
            
            # Simple AST-based function detection
            import ast
            for py_file in python_files:
                if any(exclude in str(py_file) for exclude in ['__pycache__', '.git', 'venv', '.venv']):
                    continue
                    
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Extract meaningful function names (skip private/internal)
                            if not node.name.startswith('_'):
                                code_functions.add(node.name)
                        elif isinstance(node, ast.ClassDef):
                            code_classes.add(node.name)
                            
                except Exception as e:
                    print(f"    ⚠️  Could not parse {py_file}: {e}")
                    continue
            
            print(f"  📊 Found {len(code_functions)} functions and {len(code_classes)} classes in code")
            
            # Compliance Check 1: Missing logical functions in code
            missing_in_code = []
            for func_data in logical_functions:
                func_name = func_data["function_name"]
                # Check if logical function exists in code (fuzzy matching)
                found = any(func_name.lower() in code_func.lower() or 
                           code_func.lower() in func_name.lower() 
                           for code_func in code_functions)
                
                if not found:
                    missing_in_code.append({
                        "name": func_name,
                        "description": func_data["description"],
                        "expected_module": func_data.get("module_name", "Not specified")
                    })
            
            if missing_in_code:
                compliance_issues.append({
                    "type": "logical_functions_missing_in_code",
                    "severity": "error",
                    "count": len(missing_in_code),
                    "description": "Logical functions defined in architecture but not found in code",
                    "items": [f"{item['name']} (expected in {item['expected_module']}): {item['description']}" for item in missing_in_code],
                    "recommendation": "Implement missing logical functions or update architecture"
                })
            
            # Compliance Check 2: FCHAIN implementation verification
            fchain_issues = []
            for fchain_data in logical_fchains:
                fchain_name = fchain_data["fchain_name"]
                expected_functions = fchain_data["functions"]
                
                if expected_functions:
                    implemented_count = sum(1 for func in expected_functions 
                                          if any(func.lower() in code_func.lower() 
                                               for code_func in code_functions))
                    
                    if implemented_count < len(expected_functions):
                        fchain_issues.append({
                            "name": fchain_name,
                            "expected_functions": len(expected_functions),
                            "implemented_functions": implemented_count,
                            "missing_functions": [f for f in expected_functions 
                                                if not any(f.lower() in code_func.lower() 
                                                         for code_func in code_functions)]
                        })
            
            if fchain_issues:
                compliance_issues.append({
                    "type": "fchain_implementation_gaps",
                    "severity": "warning", 
                    "count": len(fchain_issues),
                    "description": "Function chains with incomplete implementation",
                    "items": [f"{item['name']}: {item['implemented_functions']}/{item['expected_functions']} functions implemented" for item in fchain_issues],
                    "recommendation": "Implement missing functions or update FCHAIN definitions"
                })
            
            # Compliance Check 3: ACTOR interface validation
            actor_interface_issues = []
            for actor_data in logical_actors:
                actor_name = actor_data["actor_name"]
                # Check if actor represents external interface that should exist
                # Look for related interface classes, API clients, etc.
                
                interface_found = any(actor_name.lower() in class_name.lower() or
                                    'client' in class_name.lower() or
                                    'service' in class_name.lower() or  
                                    'adapter' in class_name.lower()
                                    for class_name in code_classes)
                
                if not interface_found and actor_name not in ['User', 'System', 'Admin']:
                    actor_interface_issues.append({
                        "name": actor_name,
                        "description": actor_data["description"]
                    })
            
            if actor_interface_issues:
                compliance_issues.append({
                    "type": "missing_actor_interfaces",
                    "severity": "warning",
                    "count": len(actor_interface_issues), 
                    "description": "ACTOR nodes without corresponding interface implementation",
                    "items": [f"{item['name']}: {item['description']}" for item in actor_interface_issues],
                    "recommendation": "Implement interface classes/adapters for external actors"
                })
            
            # Compliance Check 4: SCHEMA structure validation
            schema_issues = []
            for schema_data in logical_schemas:
                schema_name = schema_data["schema_name"]
                # Look for corresponding data classes, models, or schemas
                schema_found = any(schema_name.lower() in class_name.lower() or
                                 'model' in class_name.lower() or
                                 'schema' in class_name.lower()
                                 for class_name in code_classes)
                
                if not schema_found:
                    schema_issues.append({
                        "name": schema_name,
                        "description": schema_data["description"]
                    })
            
            if schema_issues:
                compliance_issues.append({
                    "type": "missing_schema_implementations", 
                    "severity": "warning",
                    "count": len(schema_issues),
                    "description": "SCHEMA nodes without corresponding data structure implementation",
                    "items": [f"{item['name']}: {item['description']}" for item in schema_issues],
                    "recommendation": "Implement data classes/models for schemas or update architecture"
                })
            
            # Calculate compliance score
            total_compliance_issues = sum(issue["count"] for issue in compliance_issues)
            error_count = sum(issue["count"] for issue in compliance_issues if issue["severity"] == "error")
            warning_count = sum(issue["count"] for issue in compliance_issues if issue["severity"] == "warning")
            
            # Compliance score calculation
            compliance_score = max(0, 100 - (error_count * 20) - (warning_count * 10))
            
            if compliance_score >= 90:
                compliance_status = "Excellent"
            elif compliance_score >= 75:
                compliance_status = "Good"
            elif compliance_score >= 50:
                compliance_status = "Fair"
            else:
                compliance_status = "Poor"
            
            summary = f"Architecture-Code Compliance: {compliance_status} ({compliance_score}/100)"
            if total_compliance_issues > 0:
                summary += f" - {error_count} critical gaps, {warning_count} minor gaps"
            else:
                summary += " - Code fully implements logical architecture"
            
            return {
                "status": "success",
                "compliance_score": compliance_score,
                "compliance_status": compliance_status,
                "issues": compliance_issues,
                "total_issues": total_compliance_issues,
                "error_count": error_count,
                "warning_count": warning_count,
                "summary": summary,
                "code_stats": {
                    "python_files_scanned": len(python_files),
                    "functions_found": len(code_functions),
                    "classes_found": len(code_classes)
                },
                "architecture_stats": {
                    "logical_functions": len(logical_functions),
                    "actors": len(logical_actors),
                    "schemas": len(logical_schemas),
                    "function_chains": len(logical_fchains)
                }
            }
            
        except Exception as e:
            return {"error": f"Compliance check failed: {str(e)}"}


class ResolveCommand(ArchitectureCommand):
    """Provide resolution guidance for architecture-code gaps"""
    
    def __init__(self):
        super().__init__("resolve", "Get resolution guidance for architecture gaps")
    
    def execute(self, args: argparse.Namespace) -> Dict[str, Any]:
        print("🔧 Architecture Gap Resolution Advisor...")
        
        if not args.issue_type:
            return self._show_available_resolutions()
        
        issue_type = args.issue_type.lower()
        resolution_strategies = {
            "missing_functions": self._resolve_missing_functions,
            "missing_actors": self._resolve_missing_actors,
            "missing_schemas": self._resolve_missing_schemas,
            "fchain_gaps": self._resolve_fchain_gaps,
            "consistency": self._resolve_consistency_issues,
            "actor_system_connection": self._resolve_actor_system_connection,
            "schema_system_connection": self._resolve_schema_system_connection
        }
        
        if issue_type not in resolution_strategies:
            return {
                "error": f"Unknown issue type: {issue_type}",
                "available_types": list(resolution_strategies.keys())
            }
        
        return resolution_strategies[issue_type](args)
    
    def _show_available_resolutions(self) -> Dict[str, Any]:
        """Show available resolution types"""
        return {
            "status": "info",
            "message": "Architecture Gap Resolution Advisor",
            "available_resolutions": {
                "missing_functions": "Resolve logical functions not implemented in code",
                "missing_actors": "Resolve ACTOR interfaces not found in code",
                "missing_schemas": "Resolve SCHEMA structures not implemented",
                "fchain_gaps": "Resolve incomplete function chain implementations",
                "consistency": "Resolve ontology consistency violations",
                "actor_system_connection": "Fix ACTOR nodes not connected to system",
                "schema_system_connection": "Fix SCHEMA nodes not connected to system"
            },
            "usage": "Use: /claudegraph resolve <issue_type> [options]"
        }
    
    def _resolve_missing_functions(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Provide guidance for implementing missing logical functions"""
        return {
            "status": "guidance",
            "issue_type": "missing_functions",
            "resolution_strategy": "Implement Missing Logical Functions",
            "approaches": [
                {
                    "approach": "Code-First (Recommended)",
                    "description": "Implement the missing logical functions in code",
                    "steps": [
                        "1. Identify the module where function should be implemented",
                        "2. Create function with logical name (e.g., 'process_user_data')",
                        "3. Implement core functionality according to architecture description",
                        "4. Add unit tests to verify implementation",
                        "5. Run compliance check to verify resolution"
                    ],
                    "when_to_use": "When architecture is correct and code needs implementation"
                },
                {
                    "approach": "Architecture-First", 
                    "description": "Update architecture to match existing code",
                    "steps": [
                        "1. Review if logical function is still needed",
                        "2. If not needed, remove from architecture via design mode",
                        "3. If needed but differently named, update function name in Neo4j",
                        "4. Update function descriptions to match actual implementation",
                        "5. Run consistency check to verify resolution"
                    ],
                    "when_to_use": "When code is correct and architecture is outdated"
                }
            ],
            "decision_framework": {
                "choose_code_first_when": [
                    "Architecture represents desired future state",
                    "Missing function is part of new feature development",
                    "Business requirements support the logical function"
                ],
                "choose_architecture_first_when": [
                    "Code already implements equivalent functionality differently",
                    "Logical function no longer matches business needs",
                    "Implementation approach has changed since architecture design"
                ]
            }
        }
    
    def _resolve_missing_actors(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Provide guidance for implementing missing ACTOR interfaces"""
        return {
            "status": "guidance",
            "issue_type": "missing_actors",
            "resolution_strategy": "Implement ACTOR Interface Boundaries",
            "approaches": [
                {
                    "approach": "Interface Implementation",
                    "description": "Create interface classes/adapters for external actors",
                    "steps": [
                        "1. Create interface class (e.g., DatabaseService, PaymentAdapter)",
                        "2. Define methods that interact with external actor",
                        "3. Implement adapter pattern for external service integration",
                        "4. Add interface to module allocation in architecture",
                        "5. Verify actor boundaries are properly enforced"
                    ]
                },
                {
                    "approach": "Architecture Refinement",
                    "description": "Update ACTOR definitions to match implementation reality",
                    "steps": [
                        "1. Review if ACTOR represents actual external boundary",
                        "2. Merge similar actors or split overly broad actors",
                        "3. Update actor descriptions to match interface reality",
                        "4. Ensure actor-system compose relationships exist"
                    ]
                }
            ]
        }
    
    def _resolve_missing_schemas(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Provide guidance for implementing missing SCHEMA structures"""
        return {
            "status": "guidance", 
            "issue_type": "missing_schemas",
            "resolution_strategy": "Implement Data Structure Consistency",
            "approaches": [
                {
                    "approach": "Data Class Implementation",
                    "description": "Create explicit data structures for schemas",
                    "steps": [
                        "1. Create data class/model for schema (e.g., UserProfile, OrderData)",
                        "2. Implement structure according to schema.Struct specification",
                        "3. Add validation and serialization methods",
                        "4. Update functions to use structured data types",
                        "5. Ensure identical interfaces across schema users"
                    ]
                },
                {
                    "approach": "Schema Consolidation", 
                    "description": "Merge or update schemas to match code reality",
                    "steps": [
                        "1. Review existing data structures in code",
                        "2. Consolidate similar schemas or split complex ones",
                        "3. Update schema.Struct to match actual data usage",
                        "4. Ensure schema-system compose relationships exist"
                    ]
                }
            ],
            "critical_requirement": "Functions using same SCHEMA must have identical interfaces"
        }
    
    def _resolve_fchain_gaps(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Provide guidance for completing function chain implementations"""
        return {
            "status": "guidance",
            "issue_type": "fchain_gaps", 
            "resolution_strategy": "Complete Function Chain Implementation",
            "approaches": [
                {
                    "approach": "Sequential Implementation",
                    "description": "Implement missing functions in logical order",
                    "steps": [
                        "1. Review FCHAIN flow relationships to understand sequence",
                        "2. Implement missing functions in dependency order",
                        "3. Ensure flow relationships match actual function calls",
                        "4. Test complete chain execution end-to-end",
                        "5. Update FCHAIN description if implementation differs"
                    ]
                },
                {
                    "approach": "Chain Refactoring",
                    "description": "Update FCHAIN to match implementation reality",
                    "steps": [
                        "1. Analyze actual function call patterns in code",
                        "2. Update FCHAIN composition to match implementation",
                        "3. Add or remove functions from chain as needed",
                        "4. Ensure all functions in chain have flow relationships"
                    ]
                }
            ]
        }
    
    def _resolve_consistency_issues(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Provide guidance for ontology consistency violations"""
        return {
            "status": "guidance",
            "issue_type": "consistency",
            "resolution_strategy": "Fix Ontology Compliance Violations",
            "common_fixes": {
                "isolated_nodes": {
                    "problem": "Nodes without any relationships",
                    "solution": "Add appropriate compose, flow, or satisfy relationships",
                    "steps": ["1. Identify node type and purpose", "2. Connect to parent via compose", "3. Add flow relationships for functions"]
                },
                "functions_without_requirements": {
                    "problem": "FUNC nodes without REQ relationships",
                    "solution": "Create requirements for each function",
                    "steps": ["1. Define functional requirement", "2. Create REQ node", "3. Add satisfy relationship"]
                },
                "requirements_without_tests": {
                    "problem": "REQ nodes without TEST verification",
                    "solution": "Create test cases for requirements", 
                    "steps": ["1. Design test case", "2. Create TEST node", "3. Add verify relationship"]
                }
            }
        }
    
    def _resolve_actor_system_connection(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Provide guidance for connecting ACTOR nodes to system"""
        return {
            "status": "guidance",
            "issue_type": "actor_system_connection",
            "resolution_strategy": "Connect ACTOR Nodes to System",
            "problem": "ACTOR nodes must be connected to system via compose relationship",
            "cypher_fix": """
            // Fix ACTOR system connections
            MATCH (sys:SYS), (a:ACTOR)
            WHERE NOT EXISTS((sys)-[:compose]->(a))
            CREATE (sys)-[:compose]->(a)
            RETURN sys.Name, a.Name
            """,
            "manual_steps": [
                "1. Identify disconnected ACTOR nodes",
                "2. For each ACTOR, create compose relationship from main SYS node",
                "3. Verify all actors are now visible in system composition",
                "4. Run consistency check to confirm resolution"
            ],
            "importance": "Critical for ontology compliance and system boundary definition"
        }
    
    def _resolve_schema_system_connection(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Provide guidance for connecting SCHEMA nodes to system"""
        return {
            "status": "guidance",
            "issue_type": "schema_system_connection", 
            "resolution_strategy": "Connect SCHEMA Nodes to System",
            "problem": "SCHEMA nodes must be connected to system via compose relationship",
            "cypher_fix": """
            // Fix SCHEMA system connections  
            MATCH (sys:SYS), (s:SCHEMA)
            WHERE NOT EXISTS((sys)-[:compose]->(s))
            CREATE (sys)-[:compose]->(s)
            RETURN sys.Name, s.Name
            """,
            "manual_steps": [
                "1. Identify disconnected SCHEMA nodes",
                "2. For each SCHEMA, create compose relationship from main SYS node", 
                "3. Verify all schemas are now visible in system composition",
                "4. Run consistency check to confirm resolution"
            ],
            "importance": "Critical for interface consistency enforcement across functions"
        }


class GrphArchitectCLI:
    """Main CLI for GrphArchitect command"""
    
    def __init__(self):
        self.commands = {
            "design": DesignCommand(),
            "analyze": AnalyzeCommand(),
            "query": QueryCommand(),
            "impact": ImpactCommand(),
            "check": CheckCommand(),
            "compliance": ComplianceCommand(),
            "resolve": ResolveCommand()
        }
    
    def create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="/command GrphArchitect",
            description="Graph-based architecture intelligence for Claude Code"
        )
        
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Design command
        design_parser = subparsers.add_parser("design", help="Create architecture design")
        design_parser.add_argument("--output", "-o", help="Output file path")
        
        # Analyze command
        analyze_parser = subparsers.add_parser("analyze", help="Analyze existing code")
        analyze_parser.add_argument("path", help="Path to analyze")
        analyze_parser.add_argument("--output", "-o", help="Output file path")
        analyze_parser.add_argument("--store-neo4j", action="store_true", help="Store in Neo4j")
        
        # Query command
        query_parser = subparsers.add_parser("query", help="Query architecture")
        query_parser.add_argument("--cypher", help="Direct Cypher query")
        query_parser.add_argument("--pattern", help="Pattern query (functions, actors, flows, etc.)")
        
        # Impact command
        impact_parser = subparsers.add_parser("impact", help="Analyze change impact")
        impact_parser.add_argument("component", help="Component to analyze")
        
        # Check command
        check_parser = subparsers.add_parser("check", help="Check consistency")
        
        # Compliance command
        compliance_parser = subparsers.add_parser("compliance", help="Check architecture-code compliance")
        compliance_parser.add_argument("path", help="Path to codebase to check against architecture")
        
        # Resolve command
        resolve_parser = subparsers.add_parser("resolve", help="Get resolution guidance for architecture gaps")
        resolve_parser.add_argument("issue_type", nargs="?", help="Type of issue to resolve (see available types)")
        
        return parser
    
    def run(self, args: List[str] = None) -> Dict[str, Any]:
        """Run the CLI command"""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        if not parsed_args.command:
            parser.print_help()
            return {"error": "No command specified"}
        
        command = self.commands.get(parsed_args.command)
        if not command:
            return {"error": f"Unknown command: {parsed_args.command}"}
        
        return command.execute(parsed_args)


def main():
    """Main entry point for command-line usage"""
    cli = GrphArchitectCLI()
    result = cli.run()
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        sys.exit(1)
    else:
        print(f"✅ {result.get('message', 'Success')}")
        if 'results' in result:
            print(json.dumps(result['results'], indent=2))


if __name__ == "__main__":
    main()