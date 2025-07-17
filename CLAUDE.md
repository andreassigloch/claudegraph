# CLAUDE.md - ClaudeGraph

## Project Overview

ClaudeGraph enables graph-based architecture intelligence for Claude Code development. It provides bidirectional synchronization between architecture design and code implementation using Neo4j graph database and the ontology schema v1.1.0.

## Architecture Philosophy

1. **Design First**: Create architecture graph before coding
2. **Extract Always**: Keep architecture synchronized with code changes
3. **Query Everything**: Use Cypher for impact analysis and architectural insights
4. **Leverage Neo4j MCP**: Use existing Neo4j MCP connector, not custom MCP server

## Core Components

### grphzer Integration
- **Purpose**: AST parsing and architecture extraction from existing code
- **Location**: `analyzer/` - Adapted from `/Users/andreas/Documents/Projekte/grphzer`
- **Features**: Python AST parsing, flow analysis, FCHAIN detection, actor boundary detection
- **Output**: JSON graph format conforming to ontology schema

### Neo4j MCP Client
- **Purpose**: Direct graph database operations via existing Neo4j MCP
- **Location**: `neo4j_client/`
- **Features**: Cypher query execution, graph storage, impact analysis queries
- **Configuration**: Docker Neo4j instance with MCP server

### Claude Code Integration
- **Purpose**: /command GrphArchitect implementation
- **Location**: `commands/`
- **Features**: Architecture design, code analysis, impact queries, consistency checks

## Key Commands

### `/command GrphArchitect`
Main entry point for architecture operations:

```bash
# Start new project with architecture design
/command GrphArchitect design

# Extract architecture from existing code
/command GrphArchitect analyze [path]

# Query current architecture
/command GrphArchitect query [cypher|pattern]

# Analyze change impacts
/command GrphArchitect impact [component]

# Check architectural consistency
/command GrphArchitect check
```

## Ontology Schema (v1.1.0)

### Primary Node Types
- **SYS**: System container
- **UC**: Use cases
- **ACTOR**: External entities (users, databases, APIs)
- **FCHAIN**: Function chains (execution flows)
- **FUNC**: Individual functions
- **REQ**: Requirements
- **TEST**: Test cases
- **MOD**: Code modules/files
- **SCHEMA**: Data structures and interfaces

### Key Relationships
- **compose**: Hierarchical composition
- **flow**: Functional dependencies with FlowDescr/FlowDef
- **satisfy**: Requirements satisfaction
- **verify**: Test verification
- **allocate**: Function to module allocation
- **relation**: Generic relationships

## Development Workflows

### 1. New Project (Forward Engineering)
```
1. /command GrphArchitect design
   → Interactive architecture design
   → Define SYS, UC, ACTOR, FCHAIN, FUNC nodes
   → Define flow relationships
   → Generate code scaffolding

2. Code development
   → Implement functions from FUNC nodes
   → Follow flow relationships
   → Maintain architecture sync

3. Continuous validation
   → /command GrphArchitect check
   → Auto-update graph from code changes
```

### 2. Existing Project (Reverse Engineering)
```
1. /command GrphArchitect analyze
   → Parse code with grphzer
   → Extract functions, flows, actors
   → Build architecture graph
   → Store in Neo4j

2. Architecture review
   → /command GrphArchitect query
   → Identify patterns and issues
   → Add missing requirements/tests

3. Maintain synchronization
   → Auto-update on code changes
   → Consistency checks
```

### 3. During Development
```
Before changes:
/command GrphArchitect impact [component]
→ See what will be affected

After changes:
→ Auto-update graph
→ Validate consistency
→ Update tests if needed
```

## Docker Setup

### Neo4j Database
```yaml
# docker-compose.yml
services:
  neo4j:
    image: neo4j:latest
    ports:
      - "7475:7474"  # Web UI (shifted to avoid conflict)
      - "7688:7687"  # Bolt protocol (shifted to avoid conflict)
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["graph-data-science"]
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - ./ontology:/import
```

### Connection Configuration
```json
{
  "neo4j": {
    "uri": "bolt://localhost:7688",
    "username": "neo4j",
    "password": "claudegraph",
    "database": "neo4j"
  }
}
```

## Configuration

### Project Configuration (.claudegraph/config.yaml)
```yaml
project:
  name: "MyProject"
  ontology_version: "1.1.0"
  auto_sync: true

neo4j:
  uri: "bolt://localhost:7688"
  username: "neo4j"
  password: "claudegraph"

grphzer:
  max_loc: 25000
  exclude_patterns:
    - "__pycache__/**"
    - ".git/**"
    - "venv/**"

analysis:
  flow_detection: true
  actor_detection: true
  fchain_generation: true
  llm_enhancement: false  # Optional
```

## Common Queries

### Impact Analysis
```cypher
// What components are affected by changing a function?
MATCH (f:FUNC {Name: $function_name})
MATCH (f)-[*1..3]-(affected)
RETURN affected, relationships(path)

// Which tests need updating?
MATCH (f:FUNC {Name: $function_name})
MATCH (f)-[:satisfy]->(r:REQ)-[:verify]->(t:TEST)
RETURN t.Name, t.Descr
```

### Architecture Validation
```cypher
// Functions without requirements
MATCH (f:FUNC)
WHERE NOT EXISTS((f)-[:satisfy]->(:REQ))
RETURN f.Name

// Requirements without tests
MATCH (r:REQ)
WHERE NOT EXISTS((r)-[:verify]->(:TEST))
RETURN r.Name
```

### Pattern Detection
```cypher
// Find functional chains
MATCH path = (a:ACTOR)-[:flow*]->(b:ACTOR)
WHERE length(path) > 2
RETURN path

// Identify system boundaries
MATCH (a:ACTOR)-[:flow]->(f:FUNC)
RETURN a.Name, count(f) as interactions
ORDER BY interactions DESC
```

## Reference Models

### Test Projects
1. **AiSE_Test**: `/Users/andreas/Documents/Tools/VSCode/AssistantTestClaude`
   - Flask web application with Neo4j
   - WebSocket communications
   - LLM tool integration

2. **grphzer**: `/Users/andreas/Documents/Projekte/grphzer`
   - Code analysis tool
   - AST parsing and flow detection
   - Architecture extraction

## Memory Items

- **ontology_schema.json**: Located in `/Users/andreas/Documents/Tools/VSCode/AssistantTestClaude/static/config/schema/ontology_schema.json`
- **grphzer capabilities**: Full AST parsing, flow analysis, actor detection, FCHAIN generation
- **Neo4j MCP**: Already exists, use directly instead of custom MCP server
- **Reference projects**: AiSE_Test and grphzer both use similar architecture patterns

## Development Guidelines

### Code Style
- Follow ontology schema v1.1.0 strictly
- Use PascalCase for node names (max 25 chars)
- Provide meaningful descriptions for all nodes
- Document flow relationships with FlowDescr and FlowDef

### Testing Strategy
- Unit tests for grphzer integration
- Integration tests with Neo4j MCP
- End-to-end tests for Claude Code commands
- Reference model validation

### Error Handling
- Graceful degradation when Neo4j unavailable
- Validation of ontology compliance
- Clear error messages for users
- Fallback to local JSON if needed

## Getting Started

1. **Setup Environment**
   ```bash
   cd /Users/andreas/Documents/Projekte/ClaudeGraph
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Start Neo4j**
   ```bash
   docker-compose up -d neo4j
   ```

3. **Load Ontology**
   ```bash
   python scripts/load_ontology.py
   ```

4. **Test with Reference Project**
   ```bash
   /command GrphArchitect analyze /Users/andreas/Documents/Tools/VSCode/AssistantTestClaude
   ```

This architecture enables Claude Code to understand and reason about code architecture through graph queries, providing intelligent suggestions and impact analysis for development decisions.