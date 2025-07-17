#!/usr/bin/env python3
"""
Add missing function-actor relationships to complete functional chains
"""

import sys
import uuid
sys.path.insert(0, '.')

from neo4j_client.client import Neo4jClient

def add_function_actor_relationships():
    """Add the missing function-actor relationships"""
    
    print("🔗 Adding Function-Actor Relationships")
    print("=" * 50)
    
    client = Neo4jClient()
    
    # Define function-actor relationships based on code analysis
    # These represent functions that interact with external systems (actors)
    
    function_actor_mappings = [
        # HTTP Client interactions
        ("make_async_post", "HttpClient", "relation", "Function makes HTTP requests"),
        ("make_bulk_requests", "HttpClient", "relation", "Function makes bulk HTTP requests"),
        ("query_hospital_agent", "HttpClient", "relation", "Function queries external API"),
        ("invoke_agent_with_retry", "HttpClient", "relation", "Function invokes HTTP agent with retry"),
        
        # Database interactions
        ("load_hospital_graph_from_csv", "Database", "relation", "Function loads data into Neo4j"),
        ("_set_uniqueness_constraints", "Database", "relation", "Function sets database constraints"),
        
        # File System interactions
        ("load_hospital_graph_from_csv", "FileSystem", "relation", "Function reads CSV files"),
        
        # Web Endpoint interactions (FastAPI routes)
        ("get_status", "WebEndpoint", "relation", "Function provides web endpoint"),
        ("query_hospital_agent", "WebEndpoint", "relation", "Function serves web endpoint")
    ]
    
    relationships_added = 0
    
    for func_name, actor_name, rel_type, description in function_actor_mappings:
        try:
            # Find function UUID
            func_result = client.execute_query(
                "MATCH (f:FUNC {Name: $func_name}) RETURN f.uuid as uuid, f.Module as module",
                {"func_name": func_name}
            )
            
            # Find actor UUID
            actor_result = client.execute_query(
                "MATCH (a:ACTOR {Name: $actor_name}) RETURN a.uuid as uuid",
                {"actor_name": actor_name}
            )
            
            if func_result and actor_result:
                # Create relationship for each function instance found
                for func in func_result:
                    func_uuid = func["uuid"]
                    actor_uuid = actor_result[0]["uuid"]
                    
                    # Create the relationship
                    client.execute_query('''
                        MATCH (f:FUNC {uuid: $func_uuid})
                        MATCH (a:ACTOR {uuid: $actor_uuid})
                        CREATE (f)-[r:relation {
                            uuid: $rel_uuid,
                            RelationType: $rel_type,
                            Description: $description
                        }]->(a)
                    ''', {
                        "func_uuid": func_uuid,
                        "actor_uuid": actor_uuid,
                        "rel_uuid": str(uuid.uuid4())[:8],
                        "rel_type": rel_type,
                        "description": description
                    })
                    
                    relationships_added += 1
                    print(f"✅ {func_name} ({func['module']}) → {actor_name}")
            
            else:
                if not func_result:
                    print(f"⚠️  Function not found: {func_name}")
                if not actor_result:
                    print(f"⚠️  Actor not found: {actor_name}")
                    
        except Exception as e:
            print(f"❌ Error creating relationship {func_name} → {actor_name}: {e}")
    
    print(f"\n🎉 Added {relationships_added} function-actor relationships")
    
    # Verify the relationships
    print("\n📊 Verification:")
    summary = client.get_architecture_summary()
    print(f"Total relationships: {summary['total_relationships']}")
    
    # Show function-actor relationships
    func_actor_rels = client.execute_query('''
        MATCH (f:FUNC)-[r:relation]->(a:ACTOR)
        RETURN f.Name as function, a.Name as actor, r.Description as description
        ORDER BY actor, function
    ''')
    
    print(f"\n🔗 Function → Actor Relationships ({len(func_actor_rels)}):")
    for rel in func_actor_rels:
        print(f"   {rel['function']} → {rel['actor']}: {rel['description']}")
    
    # Show a complete functional chain example
    print("\n⛓️  Example Functional Chain:")
    chain_example = client.execute_query('''
        MATCH (uc:UC {Name: "Hospital Query"})<-[:allocate]-(a:ACTOR)<-[:relation]-(f:FUNC)
        RETURN uc.Name as use_case, a.Name as actor, f.Name as function
        LIMIT 5
    ''')
    
    if chain_example:
        print("   Use Case → Actor → Function:")
        for chain in chain_example:
            print(f"   {chain['use_case']} → {chain['actor']} → {chain['function']}")
    else:
        print("   No complete chains found yet")
    
    return relationships_added

if __name__ == "__main__":
    add_function_actor_relationships()