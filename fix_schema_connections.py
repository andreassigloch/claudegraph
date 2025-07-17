#!/usr/bin/env python3
"""
Fix schema connections - connect schemas to functions that use them instead of system
"""

import sys
import uuid
sys.path.insert(0, '.')

from neo4j_client.client import Neo4jClient

def fix_schema_connections():
    """Fix schema connections to point to functions instead of system"""
    
    print("🔧 Fixing Schema Connections")
    print("=" * 40)
    
    client = Neo4jClient()
    
    # First, remove existing system-schema relationships
    print("🗑️  Removing system-schema relationships...")
    removed = client.execute_query('''
        MATCH (s:SYS)-[r:compose]->(schema:SCHEMA)
        DELETE r
        RETURN count(r) as removed
    ''')
    print(f"✅ Removed {removed[0]['removed']} system-schema relationships")
    
    # Define function-schema relationships based on data usage
    function_schema_mappings = [
        # Hospital schema usage
        ("load_hospital_graph_from_csv", "Hospital", "Function loads hospital data from CSV"),
        ("_get_current_hospitals", "Hospital", "Function retrieves hospital entities"),
        ("get_most_available_hospital", "Hospital", "Function queries hospital availability"),
        
        # Patient schema usage
        ("load_hospital_graph_from_csv", "Patient", "Function loads patient data from CSV"),
        
        # Visit schema usage
        ("load_hospital_graph_from_csv", "Visit", "Function loads visit data from CSV"),
        
        # WaitTime schema usage
        ("_get_current_wait_time_minutes", "WaitTime", "Function calculates wait time data"),
        ("get_current_wait_times", "WaitTime", "Function retrieves wait time information"),
        ("get_most_available_hospital", "WaitTime", "Function uses wait time for availability"),
        
        # ChatQuery schema usage
        ("query_hospital_agent", "ChatQuery", "Function processes chat queries"),
        ("invoke_agent_with_retry", "ChatQuery", "Function handles query retry logic")
    ]
    
    relationships_added = 0
    
    for func_name, schema_name, description in function_schema_mappings:
        try:
            # Find all function instances with this name
            func_results = client.execute_query(
                "MATCH (f:FUNC {Name: $func_name}) RETURN f.uuid as uuid, f.Module as module",
                {"func_name": func_name}
            )
            
            # Find schema
            schema_result = client.execute_query(
                "MATCH (s:SCHEMA {Name: $schema_name}) RETURN s.uuid as uuid",
                {"schema_name": schema_name}
            )
            
            if func_results and schema_result:
                schema_uuid = schema_result[0]["uuid"]
                
                # Create relationship for each function instance
                for func in func_results:
                    func_uuid = func["uuid"]
                    
                    # Create the relationship
                    client.execute_query('''
                        MATCH (f:FUNC {uuid: $func_uuid})
                        MATCH (s:SCHEMA {uuid: $schema_uuid})
                        CREATE (f)-[r:relation {
                            uuid: $rel_uuid,
                            RelationType: "uses",
                            Description: $description
                        }]->(s)
                    ''', {
                        "func_uuid": func_uuid,
                        "schema_uuid": schema_uuid,
                        "rel_uuid": str(uuid.uuid4())[:8],
                        "description": description
                    })
                    
                    relationships_added += 1
                    print(f"✅ {func_name} ({func['module']}) → {schema_name}")
            
            else:
                if not func_results:
                    print(f"⚠️  Function not found: {func_name}")
                if not schema_result:
                    print(f"⚠️  Schema not found: {schema_name}")
                    
        except Exception as e:
            print(f"❌ Error creating relationship {func_name} → {schema_name}: {e}")
    
    print(f"\n🎉 Added {relationships_added} function-schema relationships")
    
    # Verify the new relationships
    print("\n📊 Verification:")
    func_schema_rels = client.execute_query('''
        MATCH (f:FUNC)-[r:relation]->(s:SCHEMA)
        RETURN f.Name as function, s.Name as schema, r.Description as description
        ORDER BY schema, function
    ''')
    
    print(f"\n🔗 Function → Schema Relationships ({len(func_schema_rels)}):")
    current_schema = None
    for rel in func_schema_rels:
        if rel['schema'] != current_schema:
            current_schema = rel['schema']
            print(f"\n📋 {current_schema}:")
        print(f"   ← {rel['function']}: {rel['description']}")
    
    # Show updated relationship summary
    print(f"\n📊 Updated Relationship Summary:")
    summary = client.get_architecture_summary()
    for rel_type, count in sorted(summary['relationship_types'].items()):
        print(f"   {rel_type}: {count}")
    
    print(f"\n🎯 Total: {summary['total_nodes']} nodes, {summary['total_relationships']} relationships")
    
    # Show example data flow chain
    print(f"\n⛓️  Example Data Flow Chain:")
    data_chain = client.execute_query('''
        MATCH (uc:UC)<-[:allocate]-(a:ACTOR)<-[:relation]-(f:FUNC)-[:relation]->(s:SCHEMA)
        RETURN uc.Name as use_case, a.Name as actor, f.Name as function, s.Name as schema
        LIMIT 5
    ''')
    
    if data_chain:
        print("   Use Case → Actor → Function → Schema:")
        for chain in data_chain:
            print(f"   {chain['use_case']} → {chain['actor']} → {chain['function']} → {chain['schema']}")
    
    return relationships_added

if __name__ == "__main__":
    fix_schema_connections()