"""
Neo4j Client for ClaudeGraph
Provides graph database operations using Neo4j driver
"""

import os
from typing import Dict, List, Any, Optional
from neo4j import GraphDatabase, Driver, Session
import json
from pathlib import Path


class Neo4jClient:
    """Neo4j database client for architecture graph operations"""
    
    def __init__(self, uri: str = None, username: str = None, password: str = None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7688")
        self.username = username or os.getenv("NEO4J_USERNAME", "")
        self.password = password or os.getenv("NEO4J_PASSWORD", "")
        self.driver: Optional[Driver] = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j"""
        try:
            if self.username and self.password:
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password)
                )
            else:
                self.driver = GraphDatabase.driver(self.uri)
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print(f"✅ Connected to Neo4j at {self.uri}")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
    
    def execute_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """Execute a Cypher query and return results"""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]
    
    def store_graph(self, graph_data: Dict[str, Any]) -> bool:
        """Store graph data (nodes and relationships) in Neo4j"""
        try:
            with self.driver.session() as session:
                # Clear existing data for this project if specified
                if graph_data.get("metadata", {}).get("clear_existing", False):
                    session.run("MATCH (n) DETACH DELETE n")
                
                # Store nodes
                nodes = graph_data.get("nodes", [])
                for node in nodes:
                    self._create_node(session, node)
                
                # Store relationships
                relationships = graph_data.get("relationships", [])
                for rel in relationships:
                    self._create_relationship(session, rel)
                
                print(f"✅ Stored {len(nodes)} nodes and {len(relationships)} relationships")
                return True
                
        except Exception as e:
            print(f"❌ Failed to store graph: {e}")
            return False
    
    def _create_node(self, session: Session, node_data: Dict):
        """Create a single node in Neo4j"""
        node_type = node_data.get("type")
        uuid = node_data.get("uuid")
        
        if not node_type or not uuid:
            raise ValueError(f"Node missing type or uuid: {node_data}")
        
        # Build properties
        properties = {k: v for k, v in node_data.items() if k not in ["type"]}
        
        # Create Cypher query
        query = f"""
        MERGE (n:{node_type} {{uuid: $uuid}})
        SET n += $properties
        RETURN n
        """
        
        session.run(query, {"uuid": uuid, "properties": properties})
    
    def _create_relationship(self, session: Session, rel_data: Dict):
        """Create a single relationship in Neo4j"""
        rel_type = rel_data.get("type")
        source_uuid = rel_data.get("source")
        target_uuid = rel_data.get("target")
        uuid = rel_data.get("uuid")
        
        if not all([rel_type, source_uuid, target_uuid, uuid]):
            raise ValueError(f"Relationship missing required fields: {rel_data}")
        
        # Build properties
        properties = {k: v for k, v in rel_data.items() 
                     if k not in ["type", "source", "target"]}
        
        # Create Cypher query
        query = f"""
        MATCH (source {{uuid: $source_uuid}})
        MATCH (target {{uuid: $target_uuid}})
        MERGE (source)-[r:{rel_type} {{uuid: $uuid}}]->(target)
        SET r += $properties
        RETURN r
        """
        
        session.run(query, {
            "source_uuid": source_uuid,
            "target_uuid": target_uuid,
            "uuid": uuid,
            "properties": properties
        })
    
    def load_ontology(self, ontology_path: str = None) -> bool:
        """Load ontology constraints and indexes"""
        if not ontology_path:
            ontology_path = Path(__file__).parent.parent / "ontology" / "load_ontology.cypher"
        
        try:
            with open(ontology_path, 'r') as f:
                cypher_script = f.read()
            
            # Split on semicolons and execute each statement
            statements = [stmt.strip() for stmt in cypher_script.split(';') 
                         if stmt.strip() and not stmt.strip().startswith('//')]
            
            with self.driver.session() as session:
                for statement in statements:
                    if statement:
                        session.run(statement)
            
            print(f"✅ Loaded ontology from {ontology_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load ontology: {e}")
            return False
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get summary of current architecture in database"""
        query = """
        MATCH (n)
        WITH labels(n)[0] as node_type, count(n) as count
        RETURN node_type, count
        ORDER BY count DESC
        """
        
        node_counts = self.execute_query(query)
        
        query = """
        MATCH ()-[r]->()
        WITH type(r) as rel_type, count(r) as count
        RETURN rel_type, count
        ORDER BY count DESC
        """
        
        rel_counts = self.execute_query(query)
        
        return {
            "node_types": {item["node_type"]: item["count"] for item in node_counts},
            "relationship_types": {item["rel_type"]: item["count"] for item in rel_counts},
            "total_nodes": sum(item["count"] for item in node_counts),
            "total_relationships": sum(item["count"] for item in rel_counts)
        }
    
    def find_impact_analysis(self, component_name: str) -> Dict[str, Any]:
        """Analyze impact of changing a component"""
        # Find the component
        find_query = """
        MATCH (c)
        WHERE c.Name = $name
        RETURN c.uuid as uuid, labels(c) as types
        """
        
        component = self.execute_query(find_query, {"name": component_name})
        if not component:
            return {"error": f"Component not found: {component_name}"}
        
        component_uuid = component[0]["uuid"]
        
        # Find direct dependencies
        direct_query = """
        MATCH (c {uuid: $uuid})
        MATCH (c)-[r]-(affected)
        RETURN DISTINCT affected.uuid as uuid, 
               affected.Name as name,
               labels(affected) as types,
               type(r) as relationship_type
        """
        
        direct_impacts = self.execute_query(direct_query, {"uuid": component_uuid})
        
        # Find transitive dependencies (2-3 hops)
        transitive_query = """
        MATCH (c {uuid: $uuid})
        MATCH path = (c)-[*2..3]-(affected)
        WHERE NOT affected.uuid = $uuid
        RETURN DISTINCT affected.uuid as uuid,
               affected.Name as name,
               labels(affected) as types,
               length(path) as distance
        ORDER BY distance, name
        LIMIT 50
        """
        
        transitive_impacts = self.execute_query(transitive_query, {"uuid": component_uuid})
        
        return {
            "component": component_name,
            "component_uuid": component_uuid,
            "direct_impacts": direct_impacts,
            "transitive_impacts": transitive_impacts,
            "summary": {
                "direct_count": len(direct_impacts),
                "transitive_count": len(transitive_impacts)
            }
        }
    
    def validate_ontology_compliance(self) -> Dict[str, Any]:
        """Validate that the graph complies with ontology rules"""
        issues = []
        
        # Check 1: All nodes have required properties
        required_props_query = """
        MATCH (n)
        WHERE n.Name IS NULL OR n.Descr IS NULL
        RETURN n.uuid as uuid, labels(n) as types, n.Name as name
        """
        
        missing_props = self.execute_query(required_props_query)
        if missing_props:
            issues.append({
                "type": "missing_required_properties",
                "severity": "error",
                "count": len(missing_props),
                "items": missing_props
            })
        
        # Check 2: SCHEMA nodes have Struct property
        schema_struct_query = """
        MATCH (s:SCHEMA)
        WHERE s.Struct IS NULL
        RETURN s.uuid as uuid, s.Name as name
        """
        
        missing_struct = self.execute_query(schema_struct_query)
        if missing_struct:
            issues.append({
                "type": "schema_missing_struct",
                "severity": "error",
                "count": len(missing_struct),
                "items": missing_struct
            })
        
        # Check 3: Flow relationships have required properties
        flow_props_query = """
        MATCH ()-[r:flow]->()
        WHERE r.FlowDescr IS NULL OR r.FlowDef IS NULL
        RETURN r.uuid as uuid, r.FlowDescr as descr, r.FlowDef as def
        """
        
        missing_flow_props = self.execute_query(flow_props_query)
        if missing_flow_props:
            issues.append({
                "type": "flow_missing_properties",
                "severity": "error",
                "count": len(missing_flow_props),
                "items": missing_flow_props
            })
        
        return {
            "compliant": len(issues) == 0,
            "issues": issues,
            "total_issues": sum(issue["count"] for issue in issues)
        }
    
    def export_graph(self, output_path: str = None) -> bool:
        """Export current graph to JSON format"""
        if not output_path:
            output_path = "exported_architecture.json"
        
        try:
            # Export nodes
            nodes_query = """
            MATCH (n)
            RETURN n.uuid as uuid, labels(n)[0] as type, properties(n) as properties
            """
            
            nodes_result = self.execute_query(nodes_query)
            nodes = []
            
            for node in nodes_result:
                node_data = {"uuid": node["uuid"], "type": node["type"]}
                node_data.update(node["properties"])
                nodes.append(node_data)
            
            # Export relationships
            rels_query = """
            MATCH (source)-[r]->(target)
            RETURN r.uuid as uuid, type(r) as type, 
                   source.uuid as source, target.uuid as target,
                   properties(r) as properties
            """
            
            rels_result = self.execute_query(rels_query)
            relationships = []
            
            for rel in rels_result:
                rel_data = {
                    "uuid": rel["uuid"],
                    "type": rel["type"],
                    "source": rel["source"],
                    "target": rel["target"]
                }
                rel_data.update(rel["properties"])
                relationships.append(rel_data)
            
            # Create export data
            export_data = {
                "nodes": nodes,
                "relationships": relationships,
                "metadata": {
                    "exported_at": str(self.execute_query("RETURN datetime()")[0]["datetime()"]),
                    "total_nodes": len(nodes),
                    "total_relationships": len(relationships)
                }
            }
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"✅ Exported graph to {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to export graph: {e}")
            return False