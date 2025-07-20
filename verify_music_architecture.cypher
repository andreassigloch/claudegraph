// Verification Queries for Background Music Architecture Extension
// These queries verify that all new nodes are properly connected to the system

// 1. Verify ACTOR is connected to system
MATCH (sys:SYS {Name: "DailyWisdomApp"})-[:compose]->(actor:ACTOR {Name: "MusicService"})
RETURN sys.Name as System, actor.Name as Actor, actor.Descr as Description

// 2. Verify SCHEMA is connected to system
MATCH (sys:SYS {Name: "DailyWisdomApp"})-[:compose]->(schema:SCHEMA {Name: "MusicTrack"})
RETURN sys.Name as System, schema.Name as Schema, schema.Descr as Description

// 3. Verify complete music flow path
MATCH path = (user:ACTOR {Name: "User"})-[:flow*]-(music:ACTOR {Name: "MusicService"})
RETURN path

// 4. Verify all music functions are allocated to module
MATCH (mod:MOD {Name: "MusicManager"})<-[:allocate]-(func:FUNC)
WHERE func.Name IN ["PlayMusic", "PauseMusic", "SelectMusicTrack", "LoadMusicPreferences"]
RETURN mod.Name as Module, collect(func.Name) as AllocatedFunctions

// 5. Verify BackgroundMusicFlow chain composition
MATCH (fchain:FCHAIN {Name: "BackgroundMusicFlow"})-[:compose]->(func:FUNC)
RETURN fchain.Name as FunctionChain, collect(func.Name) as Functions

// 6. Verify use case integration
MATCH (uc1:UC {Name: "DisplayWisdom"})-[r:relation]->(uc2:UC {Name: "PlayBackgroundMusic"})
RETURN uc1.Name as MainUseCase, r.RelDescr as Relationship, uc2.Name as MusicUseCase

// 7. Verify requirement satisfaction
MATCH (func:FUNC)-[:satisfy]->(req:REQ {Name: "BackgroundMusicReq"})
RETURN req.Name as Requirement, collect(func.Name) as SatisfyingFunctions

// 8. Verify test coverage
MATCH (test:TEST {Name: "TestBackgroundMusic"})-[:verify]->(req:REQ)
RETURN test.Name as Test, req.Name as VerifiedRequirement

// 9. Check for any orphaned music-related nodes (should return empty)
MATCH (n)
WHERE n.Name IN ["PlayMusic", "PauseMusic", "SelectMusicTrack", "LoadMusicPreferences", "MusicService", "MusicTrack", "BackgroundMusicFlow", "PlayBackgroundMusic"]
AND NOT EXISTS((n)<-[:compose]-(:SYS))
AND NOT EXISTS((n)<-[:compose]-())
RETURN n.Name as OrphanedNode, labels(n) as NodeType

// 10. Display complete music architecture hierarchy
MATCH (sys:SYS {Name: "DailyWisdomApp"})-[:compose*]->(n)
WHERE n.Name IN ["PlayBackgroundMusic", "MusicService", "MusicTrack", "BackgroundMusicFlow", "MusicManager", "BackgroundMusicReq", "TestBackgroundMusic"]
OR n.ID STARTS WITH "FUNC_" AND n.Name IN ["PlayMusic", "PauseMusic", "SelectMusicTrack", "LoadMusicPreferences"]
RETURN n.Name as Component, labels(n)[0] as Type, n.Descr as Description
ORDER BY Type, n.Name