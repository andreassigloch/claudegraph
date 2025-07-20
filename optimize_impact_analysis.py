#!/usr/bin/env python3
"""
Optimized impact analysis with single query
"""

def create_optimized_impact_query():
    """Single query for complete impact analysis"""
    return """
    WITH $component as target_name
    
    // Find the target component
    MATCH (target) 
    WHERE target.Name CONTAINS target_name
    
    // Get direct impacts (1 hop)
    OPTIONAL MATCH (target)-[r1]-(direct)
    WITH target, collect(DISTINCT {
        node: direct.Name,
        type: labels(direct)[0],
        relationship: type(r1),
        direction: CASE WHEN startNode(r1) = target THEN 'outgoing' ELSE 'incoming' END
    }) as direct_impacts
    
    // Get transitive impacts (2-3 hops)
    OPTIONAL MATCH (target)-[*2..3]-(transitive)
    WHERE NOT transitive = target
    WITH target, direct_impacts, collect(DISTINCT {
        node: transitive.Name,
        type: labels(transitive)[0]
    }) as transitive_impacts
    
    // Get affected tests
    OPTIONAL MATCH (target)-[*1..3]-(req:REQ)-[:verify]->(test:TEST)
    WITH target, direct_impacts, transitive_impacts, collect(DISTINCT {
        test: test.Name,
        requirement: req.Name
    }) as affected_tests
    
    // Get functional chains
    OPTIONAL MATCH (target)-[*1..2]-(fchain:FCHAIN)
    WITH target, direct_impacts, transitive_impacts, affected_tests, collect(DISTINCT fchain.Name) as chains
    
    RETURN {
        component: target.Name,
        component_type: labels(target)[0],
        direct_impacts: direct_impacts,
        transitive_impacts: transitive_impacts,
        affected_tests: affected_tests,
        functional_chains: chains,
        impact_summary: {
            direct_count: size(direct_impacts),
            transitive_count: size(transitive_impacts),
            test_count: size(affected_tests),
            chain_count: size(chains)
        }
    } as impact_analysis
    """

def create_high_level_architecture_query():
    """Get architecture overview in single query"""
    return """
    // System overview
    MATCH (s:SYS)
    OPTIONAL MATCH (s)-[:compose]->(uc:UC)
    OPTIONAL MATCH (s)-[:compose]->(actor:ACTOR)
    OPTIONAL MATCH (s)-[:compose]->(fchain:FCHAIN)
    OPTIONAL MATCH (s)-[:compose]->(schema:SCHEMA)
    
    WITH s, 
         collect(DISTINCT uc.Name) as use_cases,
         collect(DISTINCT actor.Name) as actors,
         collect(DISTINCT fchain.Name) as chains,
         collect(DISTINCT schema.Name) as schemas
    
    // High-level metrics only
    OPTIONAL MATCH (s)-[:compose*]->(f:FUNC)
    OPTIONAL MATCH (s)-[:compose*]->(req:REQ)
    OPTIONAL MATCH (s)-[:compose*]->(test:TEST)
    
    RETURN {
        system: s.Name,
        description: s.Descr,
        architecture: {
            use_cases: use_cases,
            actors: actors,
            functional_chains: chains,
            schemas: schemas
        },
        metrics: {
            total_functions: count(DISTINCT f),
            total_requirements: count(DISTINCT req),
            total_tests: count(DISTINCT test)
        }
    } as architecture_summary
    """

def create_cached_patterns():
    """Pre-computed common patterns"""
    return {
        "flows": """
            MATCH (f1:FUNC)-[r:flow]->(f2:FUNC)
            RETURN f1.Name as source, f2.Name as target, r.FlowDescr as description
            LIMIT 10
        """,
        
        "actors": """
            MATCH (s:SYS)-[:compose]->(a:ACTOR)
            RETURN a.Name as actor, a.Descr as description
            ORDER BY a.Name
        """,
        
        "chains": """
            MATCH (s:SYS)-[:compose]->(fc:FCHAIN)
            OPTIONAL MATCH (fc)-[:compose]->(f:FUNC)
            RETURN fc.Name as chain, fc.Descr as description, count(f) as function_count
            ORDER BY function_count DESC
        """,
        
        "schemas": """
            MATCH (s:SYS)-[:compose]->(schema:SCHEMA)
            RETURN schema.Name as name, schema.Descr as description, schema.Struct as structure
            ORDER BY schema.Name
        """
    }

if __name__ == "__main__":
    print("Optimized queries created")
    print("- Single query for impact analysis")
    print("- Consolidated architecture overview")
    print("- Cached pattern queries")
    print("- High-level abstraction focus")