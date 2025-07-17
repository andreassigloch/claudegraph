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
        
        # Interactive design workflow
        project_name = input("Project name: ").strip()
        if not project_name:
            return {"error": "Project name required"}
        
        description = input("Project description: ").strip()
        
        # Create basic system structure
        architecture = {
            "nodes": [
                {
                    "uuid": f"sys-{project_name.lower()}",
                    "type": "SYS",
                    "Name": project_name,
                    "Descr": description or f"{project_name} system"
                }
            ],
            "relationships": [],
            "metadata": {
                "created_by": "ClaudeGraph",
                "version": "1.0.0",
                "ontology_version": "1.1.0"
            }
        }
        
        # Save to file
        output_path = Path(args.output) if args.output else Path(f"{project_name}_architecture.json")
        with open(output_path, 'w') as f:
            json.dump(architecture, f, indent=2)
        
        return {
            "status": "success",
            "message": f"Architecture design created: {output_path}",
            "nodes": len(architecture["nodes"]),
            "relationships": len(architecture["relationships"])
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
        """Handle pattern-based queries"""
        patterns = {
            "functions": "MATCH (f:FUNC) RETURN f.Name, f.Descr LIMIT 20",
            "actors": "MATCH (a:ACTOR) RETURN a.Name, a.Descr LIMIT 20",
            "flows": "MATCH ()-[r:flow]->() RETURN r.FlowDescr, r.FlowDef LIMIT 20",
            "requirements": "MATCH (r:REQ) RETURN r.Name, r.Descr LIMIT 20",
            "tests": "MATCH (t:TEST) RETURN t.Name, t.Descr LIMIT 20"
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
            
            # Find direct impacts
            direct_query = """
                MATCH (c {Name: $component})
                MATCH (c)-[r]-(affected)
                RETURN affected.Name as name, 
                       labels(affected) as types,
                       type(r) as relationship
                ORDER BY name
            """
            
            direct_impacts = neo4j_client.execute_query(direct_query, {"component": args.component})
            
            # Find transitive impacts
            transitive_query = """
                MATCH (c {Name: $component})
                MATCH path = (c)-[*2..3]-(affected)
                WHERE NOT affected = c
                RETURN DISTINCT affected.Name as name,
                       labels(affected) as types,
                       length(path) as distance
                ORDER BY distance, name
                LIMIT 20
            """
            
            transitive_impacts = neo4j_client.execute_query(transitive_query, {"component": args.component})
            
            # Find affected tests
            test_query = """
                MATCH (c {Name: $component})
                MATCH (c)-[:satisfy]->(r:REQ)-[:verify]->(t:TEST)
                RETURN t.Name as test_name, t.Descr as test_description
            """
            
            affected_tests = neo4j_client.execute_query(test_query, {"component": args.component})
            
            return {
                "status": "success",
                "component": args.component,
                "direct_impacts": direct_impacts,
                "transitive_impacts": transitive_impacts,
                "affected_tests": affected_tests,
                "impact_summary": {
                    "direct": len(direct_impacts),
                    "transitive": len(transitive_impacts),
                    "tests": len(affected_tests)
                }
            }
            
        except Exception as e:
            return {"error": f"Impact analysis failed: {str(e)}"}


class CheckCommand(ArchitectureCommand):
    """Check architectural consistency"""
    
    def __init__(self):
        super().__init__("check", "Check architectural consistency")
    
    def execute(self, args: argparse.Namespace) -> Dict[str, Any]:
        print("✅ Checking Architecture Consistency...")
        
        try:
            neo4j_client = Neo4jClient()
            issues = []
            
            # Check 1: Functions without requirements
            functions_without_req = neo4j_client.execute_query("""
                MATCH (f:FUNC)
                WHERE NOT EXISTS((f)-[:satisfy]->(:REQ))
                RETURN f.Name as function_name
            """)
            
            if functions_without_req:
                issues.append({
                    "type": "missing_requirements",
                    "severity": "warning",
                    "count": len(functions_without_req),
                    "items": [item["function_name"] for item in functions_without_req]
                })
            
            # Check 2: Requirements without tests
            requirements_without_tests = neo4j_client.execute_query("""
                MATCH (r:REQ)
                WHERE NOT EXISTS((r)-[:verify]->(:TEST))
                RETURN r.Name as requirement_name
            """)
            
            if requirements_without_tests:
                issues.append({
                    "type": "missing_tests",
                    "severity": "error",
                    "count": len(requirements_without_tests),
                    "items": [item["requirement_name"] for item in requirements_without_tests]
                })
            
            # Check 3: Isolated nodes
            isolated_nodes = neo4j_client.execute_query("""
                MATCH (n)
                WHERE NOT EXISTS((n)--())
                RETURN n.Name as node_name, labels(n) as node_types
            """)
            
            if isolated_nodes:
                issues.append({
                    "type": "isolated_nodes",
                    "severity": "warning",
                    "count": len(isolated_nodes),
                    "items": [f"{item['node_name']} ({item['node_types']})" for item in isolated_nodes]
                })
            
            # Calculate overall health score
            total_issues = sum(issue["count"] for issue in issues)
            health_score = max(0, 100 - (total_issues * 10))
            
            return {
                "status": "success",
                "health_score": health_score,
                "issues": issues,
                "total_issues": total_issues,
                "summary": f"Found {total_issues} issues" if total_issues > 0 else "Architecture is consistent"
            }
            
        except Exception as e:
            return {"error": f"Consistency check failed: {str(e)}"}


class GrphArchitectCLI:
    """Main CLI for GrphArchitect command"""
    
    def __init__(self):
        self.commands = {
            "design": DesignCommand(),
            "analyze": AnalyzeCommand(),
            "query": QueryCommand(),
            "impact": ImpactCommand(),
            "check": CheckCommand()
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