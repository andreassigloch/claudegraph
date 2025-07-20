# ClaudeGraph

**Graph-based Architecture Intelligence for Claude Code**

ClaudeGraph enables Claude Code to understand and reason about software architecture through graph databases, providing intelligent suggestions and impact analysis for development decisions.

## Overview

ClaudeGraph bridges the gap between code and architecture by:
- **Extracting architecture** from existing codebases using grphzer
- **Storing structure** in Neo4j graph database
- **Providing intelligent queries** for impact analysis
- **Enabling design-first** development workflows

## Key Features

### 🔍 Reverse Engineering
- Parse existing Python codebases
- Extract functions, flows, and dependencies
- Identify actors and system boundaries
- Generate architecture graphs automatically

### 🎨 Forward Engineering
- Design architecture before coding
- Create ontology-compliant structures
- Generate code scaffolding from graphs
- Maintain architecture-code synchronization

### 📊 Impact Analysis
- Analyze change impacts before implementation
- Trace dependencies across components
- Identify affected tests and requirements
- Assess architectural risks

### 🔎 Architecture Queries
- Natural language to Cypher translation
- Pattern-based architecture exploration
- Consistency validation and health checks
- Real-time architectural insights

## Architecture

```
Claude Code ↔ Neo4j MCP ↔ Neo4j Database
     ↕
ClaudeGraph Tools:
- grphzer (code analysis)
- Prompt templates
- Query patterns
- Ontology validation
```

## Quick Start

### 1. Setup Environment
```bash
# Clone and setup
cd /Users/andreas/Documents/Projekte/ClaudeGraph
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start Neo4j
docker-compose up -d
```

### 2. Load Ontology
```bash
# Load schema into Neo4j
docker exec -it claudegraph-neo4j cypher-shell -u neo4j -p claudegraph -f /import/ontology/load_ontology.cypher
```

### 3. First Analysis
```bash
# Analyze existing project
python -m commands.grph_architect analyze /path/to/project --store-neo4j

# Or use Claude Code command
/command GrphArchitect analyze /path/to/project
```

## Claude Code Integration

### Commands Available

#### `/command GrphArchitect design`
Create architecture for new projects:
- Interactive design process
- Ontology-compliant structure
- Code scaffolding generation

#### `/command GrphArchitect analyze [path]`
Extract architecture from existing code:
- grphzer-powered analysis
- Automatic graph generation
- Neo4j storage

#### `/command GrphArchitect query [pattern]`
Query architecture:
- Natural language patterns
- Direct Cypher queries
- Structured insights

#### `/command GrphArchitect impact [component]`
Analyze change impacts:
- Direct and transitive dependencies
- Test implications
- Risk assessment

#### `/command GrphArchitect check`
Validate architecture:
- Ontology compliance
- Structural integrity
- Quality metrics

## Ontology Schema

Based on v1.1.0 with these key node types:

- **SYS**: System containers
- **UC**: Use cases
- **ACTOR**: External entities
- **FCHAIN**: Function chains
- **FUNC**: Individual functions
- **REQ**: Requirements
- **TEST**: Test cases
- **MOD**: Code modules
- **SCHEMA**: Data structures

## Use Cases

### 1. New Project Development
```
1. /command GrphArchitect design
   → Interactive architecture design
   → Define system structure
   → Generate code scaffolding

2. Implement functions
   → Follow architecture blueprint
   → Maintain graph synchronization

3. Continuous validation
   → /command GrphArchitect check
   → Ensure ontology compliance
```

### 2. Legacy Code Understanding
```
1. /command GrphArchitect analyze
   → Extract current architecture
   → Identify patterns and issues

2. /command GrphArchitect query
   → Explore system structure
   → Understand dependencies

3. Plan improvements
   → /command GrphArchitect impact
   → Safe refactoring decisions
```

### 3. Change Impact Analysis
```
Before changes:
→ /command GrphArchitect impact [component]
→ Understand what will be affected
→ Plan testing strategy

After changes:
→ Auto-update architecture graph
→ Validate consistency
→ Update documentation
```

## Configuration

### Docker Neo4j
```yaml
# docker-compose.yml
services:
  neo4j:
    image: neo4j:5.15
    ports:
      - "7475:7474"  # Web UI (shifted to avoid conflict)
      - "7688:7687"  # Bolt (shifted to avoid conflict)
    environment:
      - NEO4J_AUTH=neo4j/claudegraph
```

### Project Configuration
```yaml
# .claudegraph/config.yaml
project:
  name: "MyProject"
  ontology_version: "1.1.0"
  auto_sync: true

neo4j:
  uri: "bolt://localhost:7688"
  username: "neo4j"
  password: "claudegraph"

analysis:
  flow_detection: true
  actor_detection: true
  fchain_generation: true
```

## Example Queries

### Architecture Overview
```cypher
MATCH (s:SYS)-[:compose*]->(n)
RETURN s.Name as system, 
       labels(n) as components,
       count(n) as count
```

### Impact Analysis
```cypher
MATCH (c:FUNC {Name: 'SearchAgent'})
MATCH (c)-[*1..3]-(affected)
RETURN affected.Name as impacted_component,
       labels(affected) as types
```

### Test Coverage
```cypher
MATCH (r:REQ)
OPTIONAL MATCH (r)-[:verify]->(t:TEST)
RETURN r.Name as requirement,
       CASE WHEN t IS NULL THEN 'No Test' ELSE t.Name END as test_status
```

## Reference Projects

ClaudeGraph has been tested with:
- **AiSE_Test**: Flask application with Neo4j and WebSocket
- **grphzer**: Code analysis tool with AST parsing
- **ClaudeGraph**: Self-analysis and architecture validation

## Development

### Project Structure
```
ClaudeGraph/
├── analyzer/           # grphzer integration
├── commands/           # /command GrphArchitect
├── neo4j_client/       # Neo4j operations
├── ontology/           # Schema and constraints
├── prompts/            # Claude Code templates
├── docker-compose.yml  # Neo4j setup
├── requirements.txt    # Dependencies
└── CLAUDE.md          # Claude Code instructions
```

### Contributing
1. Follow ontology schema v1.1.0 strictly
2. Use prompt templates for consistent behavior
3. Test with reference projects
4. Validate against architectural rules

## License

MIT License - See LICENSE file for details.

## Citation

If you use ClaudeGraph in your development workflow:

```bibtex
@software{claudegraph,
  title={ClaudeGraph: Graph-based Architecture Intelligence},
  author={Andreas Sigloch},
  year={2025},
  url={https://github.com/andreas/claudegraph}
}
```

---

**ClaudeGraph makes architecture visible, queryable, and actionable for intelligent development decisions.**