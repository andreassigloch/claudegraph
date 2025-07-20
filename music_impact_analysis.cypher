// Music Component Impact Analysis Queries
// Analyze the impact of changes to music-related components

// 1. Find all music-related components in the system
MATCH (n)
WHERE n.Name =~ '.*[Mm]usic.*' 
   OR n.Name =~ '.*[Aa]udio.*' 
   OR n.Name =~ '.*[Ss]ound.*'
   OR n.Descr =~ '.*[Mm]usic.*' 
   OR n.Descr =~ '.*[Aa]udio.*'
RETURN n.Name as Component, labels(n)[0] as Type, n.Descr as Description
ORDER BY Type, n.Name;

// 2. Direct dependencies of music components
MATCH (music)-[r]->(dependent)
WHERE music.Name =~ '.*[Mm]usic.*' OR music.Descr =~ '.*[Mm]usic.*'
RETURN music.Name as MusicComponent, type(r) as Relationship, dependent.Name as DependentComponent, labels(dependent)[0] as DependentType;

// 3. Components that depend on music functionality
MATCH (dependent)-[r]->(music)
WHERE music.Name =~ '.*[Mm]usic.*' OR music.Descr =~ '.*[Mm]usic.*'
RETURN dependent.Name as DependentComponent, type(r) as Relationship, music.Name as MusicComponent, labels(dependent)[0] as DependentType;

// 4. Impact radius - components within 3 hops of music functionality
MATCH path = (music)-[*1..3]-(affected)
WHERE music.Name =~ '.*[Mm]usic.*' OR music.Descr =~ '.*[Mm]usic.*'
RETURN DISTINCT affected.Name as AffectedComponent, 
       labels(affected)[0] as ComponentType,
       length(path) as Distance,
       affected.Descr as Description
ORDER BY Distance, ComponentType, affected.Name;

// 5. Test cases that verify music functionality
MATCH (test:TEST)-[:verify]->(req:REQ)
WHERE req.Descr =~ '.*[Mm]usic.*' OR test.Descr =~ '.*[Mm]usic.*'
RETURN test.Name as TestCase, test.Descr as TestDescription, req.Name as Requirement;

// 6. Use cases related to music
MATCH (uc:UC)
WHERE uc.Name =~ '.*[Mm]usic.*' OR uc.Descr =~ '.*[Mm]usic.*'
OPTIONAL MATCH (uc)-[r:relation]-(related:UC)
RETURN uc.Name as MusicUseCase, uc.Descr as Description, 
       collect({relatedUC: related.Name, relationship: r.RelDescr}) as RelatedUseCases;

// 7. Function chains involving music
MATCH (fchain:FCHAIN)-[:compose]->(func:FUNC)
WHERE fchain.Name =~ '.*[Mm]usic.*' 
   OR fchain.Descr =~ '.*[Mm]usic.*'
   OR func.Name =~ '.*[Mm]usic.*'
RETURN DISTINCT fchain.Name as FunctionChain, 
       fchain.Descr as ChainDescription,
       collect(func.Name) as Functions;

// 8. Module allocation for music functions
MATCH (func:FUNC)-[:allocate]->(mod:MOD)
WHERE func.Name =~ '.*[Mm]usic.*' OR mod.Name =~ '.*[Mm]usic.*'
RETURN mod.Name as Module, collect(func.Name) as MusicFunctions;

// 9. Schema usage by music components
MATCH (component)-[:use]->(schema:SCHEMA)
WHERE component.Name =~ '.*[Mm]usic.*' OR schema.Name =~ '.*[Mm]usic.*'
RETURN component.Name as Component, schema.Name as UsedSchema, schema.Descr as SchemaDescription;

// 10. Complete impact summary
MATCH (music)
WHERE music.Name =~ '.*[Mm]usic.*' OR music.Descr =~ '.*[Mm]usic.*'
OPTIONAL MATCH (music)-[r1]-(direct)
OPTIONAL MATCH (music)-[*2]-(indirect)
WITH music, 
     count(DISTINCT direct) as DirectConnections,
     count(DISTINCT indirect) as IndirectConnections
RETURN music.Name as MusicComponent,
       labels(music)[0] as ComponentType,
       DirectConnections,
       IndirectConnections,
       DirectConnections + IndirectConnections as TotalImpact
ORDER BY TotalImpact DESC;