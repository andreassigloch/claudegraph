// Extend Daily Wisdom App Architecture with Background Music Functionality
// Following ontology v1.1.0 and ensuring all ACTOR and SCHEMA nodes are connected to system

// 1. Create PlayBackgroundMusic Use Case
CREATE (uc:UC {
    Name: "PlayBackgroundMusic",
    ID: "UC_PlayBackgroundMusic",
    Descr: "Enable calm background music during wisdom quote viewing for enhanced meditation experience"
})

// 2. Create MusicService Actor
CREATE (actor:ACTOR {
    Name: "MusicService",
    ID: "ACTOR_MusicService",
    Descr: "External service responsible for audio playback and music streaming functionality"
})

// 3. Create MusicTrack Schema
CREATE (schema:SCHEMA {
    Name: "MusicTrack",
    ID: "SCHEMA_MusicTrack",
    Descr: "Data structure for background music track information",
    SchemaSpec: '{
        "type": "object",
        "properties": {
            "trackId": {"type": "string", "description": "Unique identifier for the music track"},
            "title": {"type": "string", "description": "Name of the music track"},
            "artist": {"type": "string", "description": "Artist or composer name"},
            "duration": {"type": "integer", "description": "Track duration in seconds"},
            "url": {"type": "string", "description": "URL or path to the audio file"},
            "genre": {"type": "string", "description": "Music genre (e.g., ambient, meditation, nature)"},
            "volume": {"type": "number", "description": "Default volume level (0.0 to 1.0)"}
        },
        "required": ["trackId", "title", "url", "duration"]
    }'
})

// 4. Create Music-related Functions
CREATE (f1:FUNC {
    Name: "PlayMusic",
    ID: "FUNC_PlayMusic",
    Descr: "Start playing background music track with fade-in effect",
    FuncDef: "def play_music(track: MusicTrack, fade_duration: float = 2.0) -> bool"
})

CREATE (f2:FUNC {
    Name: "PauseMusic",
    ID: "FUNC_PauseMusic",
    Descr: "Pause currently playing music with fade-out effect",
    FuncDef: "def pause_music(fade_duration: float = 1.0) -> bool"
})

CREATE (f3:FUNC {
    Name: "SelectMusicTrack",
    ID: "FUNC_SelectMusicTrack",
    Descr: "Select appropriate music track based on user preferences and current wisdom theme",
    FuncDef: "def select_music_track(theme: str, preferences: dict) -> MusicTrack"
})

CREATE (f4:FUNC {
    Name: "LoadMusicPreferences",
    ID: "FUNC_LoadMusicPreferences",
    Descr: "Load user's music preferences including volume, genre, and enabled status",
    FuncDef: "def load_music_preferences(user_id: str) -> dict"
})

// 5. Create BackgroundMusicFlow FCHAIN
CREATE (fchain:FCHAIN {
    Name: "BackgroundMusicFlow",
    ID: "FCHAIN_BackgroundMusicFlow",
    Descr: "Function chain managing the complete background music lifecycle during wisdom viewing"
})

// Now create all the relationships

// Connect Use Case to System (assuming DailyWisdomApp system exists)
MATCH (sys:SYS {Name: "DailyWisdomApp"})
MATCH (uc:UC {ID: "UC_PlayBackgroundMusic"})
CREATE (sys)-[:compose]->(uc)

// CRITICAL: Connect ACTOR to System via compose relationship
MATCH (sys:SYS {Name: "DailyWisdomApp"})
MATCH (actor:ACTOR {ID: "ACTOR_MusicService"})
CREATE (sys)-[:compose]->(actor)

// CRITICAL: Connect SCHEMA to System via compose relationship
MATCH (sys:SYS {Name: "DailyWisdomApp"})
MATCH (schema:SCHEMA {ID: "SCHEMA_MusicTrack"})
CREATE (sys)-[:compose]->(schema)

// Connect FCHAIN to Use Case
MATCH (uc:UC {ID: "UC_PlayBackgroundMusic"})
MATCH (fchain:FCHAIN {ID: "FCHAIN_BackgroundMusicFlow"})
CREATE (uc)-[:compose]->(fchain)

// Connect Functions to FCHAIN
MATCH (fchain:FCHAIN {ID: "FCHAIN_BackgroundMusicFlow"})
MATCH (f1:FUNC {ID: "FUNC_LoadMusicPreferences"})
MATCH (f2:FUNC {ID: "FUNC_SelectMusicTrack"})
MATCH (f3:FUNC {ID: "FUNC_PlayMusic"})
MATCH (f4:FUNC {ID: "FUNC_PauseMusic"})
CREATE (fchain)-[:compose]->(f1)
CREATE (fchain)-[:compose]->(f2)
CREATE (fchain)-[:compose]->(f3)
CREATE (fchain)-[:compose]->(f4)

// Create flow relationships between functions
MATCH (f1:FUNC {ID: "FUNC_LoadMusicPreferences"})
MATCH (f2:FUNC {ID: "FUNC_SelectMusicTrack"})
CREATE (f1)-[:flow {
    FlowDescr: "User preferences flow to track selection",
    FlowDef: "preferences = load_music_preferences(user_id); track = select_music_track(theme, preferences)"
}]->(f2)

MATCH (f2:FUNC {ID: "FUNC_SelectMusicTrack"})
MATCH (f3:FUNC {ID: "FUNC_PlayMusic"})
CREATE (f2)-[:flow {
    FlowDescr: "Selected track flows to music player",
    FlowDef: "track = select_music_track(theme, preferences); play_music(track)"
}]->(f3)

// Connect MusicService Actor to music functions
MATCH (actor:ACTOR {ID: "ACTOR_MusicService"})
MATCH (f1:FUNC {ID: "FUNC_PlayMusic"})
MATCH (f2:FUNC {ID: "FUNC_PauseMusic"})
CREATE (f1)-[:flow {
    FlowDescr: "Play music command sent to music service",
    FlowDef: "music_service.play(track, fade_in)"
}]->(actor)
CREATE (f2)-[:flow {
    FlowDescr: "Pause music command sent to music service",
    FlowDef: "music_service.pause(fade_out)"
}]->(actor)

// Connect Schema to Functions that use it
MATCH (schema:SCHEMA {ID: "SCHEMA_MusicTrack"})
MATCH (f1:FUNC {ID: "FUNC_PlayMusic"})
MATCH (f2:FUNC {ID: "FUNC_SelectMusicTrack"})
CREATE (f1)-[:use]->(schema)
CREATE (f2)-[:use]->(schema)

// Connect to existing DisplayWisdom Use Case for integration
MATCH (uc1:UC {Name: "DisplayWisdom"})
MATCH (uc2:UC {ID: "UC_PlayBackgroundMusic"})
CREATE (uc1)-[:relation {
    RelDescr: "Background music enhances wisdom display experience",
    RelType: "enhances"
}]->(uc2)

// Connect LoadMusicPreferences to User actor (assuming it exists)
MATCH (user:ACTOR {Name: "User"})
MATCH (f:FUNC {ID: "FUNC_LoadMusicPreferences"})
CREATE (user)-[:flow {
    FlowDescr: "User preferences requested from user profile",
    FlowDef: "user.get_music_preferences()"
}]->(f)

// Create Module for music functionality
CREATE (mod:MOD {
    Name: "MusicManager",
    ID: "MOD_MusicManager",
    Descr: "Module handling all background music functionality",
    ModDef: "music_manager.py"
})

// Connect Module to System
MATCH (sys:SYS {Name: "DailyWisdomApp"})
MATCH (mod:MOD {ID: "MOD_MusicManager"})
CREATE (sys)-[:compose]->(mod)

// Allocate music functions to MusicManager module
MATCH (mod:MOD {ID: "MOD_MusicManager"})
MATCH (f1:FUNC {ID: "FUNC_PlayMusic"})
MATCH (f2:FUNC {ID: "FUNC_PauseMusic"})
MATCH (f3:FUNC {ID: "FUNC_SelectMusicTrack"})
MATCH (f4:FUNC {ID: "FUNC_LoadMusicPreferences"})
CREATE (f1)-[:allocate]->(mod)
CREATE (f2)-[:allocate]->(mod)
CREATE (f3)-[:allocate]->(mod)
CREATE (f4)-[:allocate]->(mod)

// Add requirements for the new functionality
CREATE (req:REQ {
    Name: "BackgroundMusicReq",
    ID: "REQ_BackgroundMusic",
    Descr: "System shall provide calm background music during wisdom viewing to enhance meditation experience"
})

// Connect requirement to system and functions
MATCH (sys:SYS {Name: "DailyWisdomApp"})
MATCH (req:REQ {ID: "REQ_BackgroundMusic"})
CREATE (sys)-[:compose]->(req)

MATCH (req:REQ {ID: "REQ_BackgroundMusic"})
MATCH (f1:FUNC {ID: "FUNC_PlayMusic"})
MATCH (f2:FUNC {ID: "FUNC_PauseMusic"})
MATCH (f3:FUNC {ID: "FUNC_SelectMusicTrack"})
CREATE (f1)-[:satisfy]->(req)
CREATE (f2)-[:satisfy]->(req)
CREATE (f3)-[:satisfy]->(req)

// Add test case for music functionality
CREATE (test:TEST {
    Name: "TestBackgroundMusic",
    ID: "TEST_BackgroundMusic",
    Descr: "Verify background music plays correctly during wisdom viewing and respects user preferences",
    TestDef: "test_background_music_integration()"
})

// Connect test to system and requirement
MATCH (sys:SYS {Name: "DailyWisdomApp"})
MATCH (test:TEST {ID: "TEST_BackgroundMusic"})
CREATE (sys)-[:compose]->(test)

MATCH (req:REQ {ID: "REQ_BackgroundMusic"})
MATCH (test:TEST {ID: "TEST_BackgroundMusic"})
CREATE (test)-[:verify]->(req)