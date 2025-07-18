// Load Music Architecture into Neo4j
// Fixed version with proper WITH clauses

// First, check if DailyWisdomApp exists, if not create it
MERGE (sys:SYS {Name: "DailyWisdomApp"})
ON CREATE SET sys.ID = "SYS_DailyWisdomApp", 
              sys.Descr = "Daily Wisdom mobile application with music features"
WITH sys

// 1. Create PlayBackgroundMusic Use Case
MERGE (uc:UC {ID: "UC_PlayBackgroundMusic"})
ON CREATE SET uc.Name = "PlayBackgroundMusic",
              uc.Descr = "Enable calm background music during wisdom quote viewing for enhanced meditation experience"
WITH sys, uc
CREATE (sys)-[:compose]->(uc)
WITH sys

// 2. Create MusicService Actor
MERGE (actor:ACTOR {ID: "ACTOR_MusicService"})
ON CREATE SET actor.Name = "MusicService",
              actor.Descr = "External service responsible for audio playback and music streaming functionality"
WITH sys, actor
CREATE (sys)-[:compose]->(actor)
WITH sys

// 3. Create MusicTrack Schema
MERGE (schema:SCHEMA {ID: "SCHEMA_MusicTrack"})
ON CREATE SET schema.Name = "MusicTrack",
              schema.Descr = "Data structure for background music track information",
              schema.Struct = '{"trackId": "string", "title": "string", "artist": "string", "duration": "integer", "url": "string", "genre": "string", "isCalming": "boolean"}'
WITH sys, schema
CREATE (sys)-[:compose]->(schema)
WITH sys

// 4. Create Music Functions
MERGE (f1:FUNC {ID: "FUNC_PlayMusic"})
ON CREATE SET f1.Name = "PlayMusic",
              f1.Descr = "Start playing background music track with fade-in effect"
WITH sys

MERGE (f2:FUNC {ID: "FUNC_PauseMusic"})
ON CREATE SET f2.Name = "PauseMusic",
              f2.Descr = "Pause currently playing music with fade-out effect"
WITH sys

MERGE (f3:FUNC {ID: "FUNC_SelectMusicTrack"})
ON CREATE SET f3.Name = "SelectMusicTrack",
              f3.Descr = "Select appropriate calming music track based on user preferences"
WITH sys

MERGE (f4:FUNC {ID: "FUNC_LoadMusicPreferences"})
ON CREATE SET f4.Name = "LoadMusicPreferences",
              f4.Descr = "Load user's music preferences including volume and genre settings"
WITH sys

// 5. Create BackgroundMusicFlow Functional Chain
MERGE (fchain:FCHAIN {ID: "FCHAIN_BackgroundMusicFlow"})
ON CREATE SET fchain.Name = "BackgroundMusicFlow",
              fchain.Descr = "Function chain managing the complete background music lifecycle during wisdom viewing"
WITH sys, fchain
CREATE (sys)-[:compose]->(fchain)
WITH sys

// 6. Create MusicManager Module
MERGE (mod:MOD {ID: "MOD_MusicManager"})
ON CREATE SET mod.Name = "MusicManager",
              mod.Descr = "Module containing all music-related functionality for the Daily Wisdom app"
WITH sys, mod
CREATE (sys)-[:compose]->(mod)
WITH sys

// 7. Create BackgroundMusicReq Requirement
MERGE (req:REQ {ID: "REQ_BackgroundMusic"})
ON CREATE SET req.Name = "BackgroundMusicReq",
              req.Descr = "System shall provide optional calming background music during wisdom quote viewing"
WITH sys

// 8. Create TestBackgroundMusic Test
MERGE (test:TEST {ID: "TEST_BackgroundMusic"})
ON CREATE SET test.Name = "TestBackgroundMusic",
              test.Descr = "Verify background music plays correctly and enhances user experience"
WITH sys

// Now verify the music architecture is loaded
MATCH (n)
WHERE n.Name CONTAINS 'Music' OR n.Name CONTAINS 'music'
RETURN labels(n)[0] as NodeType, n.Name as Name, n.Descr as Description
ORDER BY NodeType, Name;