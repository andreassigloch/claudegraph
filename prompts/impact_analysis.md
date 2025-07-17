# Impact Analysis Prompt

You are analyzing the impact of changing a component in a software architecture. Use the Neo4j MCP to query the architecture graph and provide a comprehensive impact analysis.

## Your Task

Given a component name, analyze:
1. **Direct impacts**: Components directly connected to this one
2. **Transitive impacts**: Components affected through chains of dependencies
3. **Test implications**: Which tests need to be updated
4. **Risk assessment**: Categorize the change risk level

## Query Pattern

Use these Cypher queries as starting points:

### Find Direct Dependencies
```cypher
MATCH (c {Name: $component_name})
MATCH (c)-[r]-(affected)
RETURN DISTINCT affected.Name as name, 
       labels(affected) as types,
       type(r) as relationship_type,
       r.FlowDescr as flow_description
ORDER BY name
```

### Find Transitive Dependencies
```cypher
MATCH (c {Name: $component_name})
MATCH path = (c)-[*2..3]-(affected)
WHERE NOT affected = c
RETURN DISTINCT affected.Name as name,
       labels(affected) as types,
       length(path) as distance
ORDER BY distance, name
LIMIT 20
```

### Find Affected Tests
```cypher
MATCH (c {Name: $component_name})
MATCH (c)-[:satisfy]->(r:REQ)-[:verify]->(t:TEST)
RETURN t.Name as test_name, t.Descr as test_description
```

### Find Requirements Chain
```cypher
MATCH (c {Name: $component_name})
MATCH (c)-[:satisfy]->(r:REQ)
RETURN r.Name as requirement, r.Descr as description
```

## Response Format

Structure your response as follows:

### 📊 Impact Analysis for [Component Name]

#### 🎯 Direct Impacts
- List directly connected components
- Explain the type of relationship (flow, composition, etc.)
- Highlight critical dependencies

#### 🔗 Transitive Impacts
- Show components affected through chains
- Indicate distance/depth of impact
- Identify potential cascade effects

#### 🧪 Testing Implications
- List tests that need updates
- Identify missing test coverage
- Suggest new tests if needed

#### ⚠️ Risk Assessment
- **Low Risk**: Only affects isolated components
- **Medium Risk**: Affects multiple components but with clear boundaries
- **High Risk**: Affects core components with many dependencies

#### 💡 Recommendations
- Suggest mitigation strategies
- Recommend testing approach
- Highlight architectural improvements

## Example Usage

When user asks: "What happens if I change the SearchAgent component?"

1. Execute the queries above with $component_name = "SearchAgent"
2. Analyze the results
3. Provide structured response with concrete findings
4. Include specific component names and relationship types
5. Give actionable recommendations

## Key Principles

- Focus on **actionable insights**, not just data
- **Prioritize by impact severity** (critical path components first)
- **Explain WHY** something is impacted, not just what
- **Suggest concrete next steps** for the developer
- **Use the ontology** to understand relationship semantics (flow vs compose vs satisfy)

Remember: The goal is to help Claude Code users make informed decisions about code changes by understanding the full architectural impact.