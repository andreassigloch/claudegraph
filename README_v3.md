# ClaudeGraph v3: Ultra-Simple Compliance Checker

## 🎯 Key Innovation

ClaudeGraph v3 abandons complex tooling in favor of markdown-based architectural compliance. It uses simple Flow → Function → Schema definitions that leverage Claude's powerful vector search capabilities.

## 📚 Evolution & Learnings

### What We Learned
1. **Vector Search is King**: Claude's hardcoded embeddings are incredibly powerful - use them, don't fight them
2. **Markdown > Databases**: Simple `.md` files are the optimal LLM interface  
3. **Semantics > AST**: For understanding code meaning, use LLM capabilities, not syntax trees
4. **Compliance > Speed**: Architectural governance matters more than performance gains

### The Journey
- **v1**: Over-engineered with AST + Neo4j + ChromaDB
- **v2**: Complex graph operations with 35.6k token searches
- **v3**: Ultra-simple markdown compliance (90% token reduction)

## 🚀 Core Concepts

### 1. Flows
Business processes as function chains:
```markdown
Flow:UserRegistration -> F:ValidateInput(S:UserData) -> F:CreateAccount(S:Account) -> F:SendWelcome(S:Email)
```

### 2. Functions  
Business operations (NOT implementation):
- ✅ `F:ProcessPayment` (what it does)
- ❌ `F:_calculate_tax()` (how it's coded)

### 3. Schemas
Data contracts with optional fields and enums:
```markdown
S:Order{id:int, items:list, total:float, status:enum[pending,paid,shipped], notes:str?}
```

## 📋 Three Simple Commands

### `/flow-check`
Analyze code changes for compliance violations
```bash
# After making changes
/flow-check
# Reports schema drift, flow breaks, missing abstractions
```

### `/flow-update`
Update flow.md while maintaining business abstraction
```bash
# When adding new features
/flow-update
# Ensures proper documentation without technical details
```

### `/flow-find <keyword>`
Search flows, functions, and schemas
```bash
/flow-find payment
# Finds all payment-related business logic
```

## 💡 Real-World Value

### Discovered Use Case: Compliance Checking
During A/B testing, we discovered the real value isn't speed but **architectural governance**:

1. **Schema Drift Detection**: Identifies when code diverges from documented contracts
2. **Flow Violation Alerts**: Catches when functions are called outside defined flows  
3. **Missing Documentation**: Highlights undocumented business processes
4. **Impact Analysis**: Shows which flows are affected by changes

### Example
```bash
# After implementing chart_type parameter
/flow-check

# Output:
Schema Violations:
- S:VisualizationRequest missing chart_type field
- Chart type validation flow not documented

Recommendations:
- Update S:VisualizationRequest{..., chart_type:enum[bar,matrix,network]?}
- Add F:ValidateChartType to visualization flow
```

## 🛠️ Setup

1. Create `flow.md` in your project root:
```markdown
# Project Flows

## Schemas
S:User{id:int, email:str, role:enum[admin,user]}
S:Session{token:str, user_id:int, expires:datetime}

## Flows
Flow:Login -> F:ValidateCredentials(S:User) -> F:CreateSession(S:Session) -> F:ReturnToken(S:Session)
```

2. Copy commands to `.claude/commands/`:
```bash
cp flow-*.md /path/to/project/.claude/commands/
```

3. Use in Claude Code:
```bash
/flow-check    # Check compliance
/flow-update   # Update documentation
/flow-find api # Find API-related flows
```

## 🔧 Compact Graph Representation for AiSE

For AiSE integration, we can enhance the ontology with compact notation:

### Enhanced Flow Syntax
```markdown
# Sequential flow
Flow:Process -> F:Step1 -> F:Step2 -> F:Step3

# Parallel/Fork flow (using pipe)
Flow:Process -> F:Validate -> F:ProcessA|F:ProcessB -> F:Merge

# Conditional flow  
Flow:Payment -> F:CheckBalance -> [sufficient]:F:Charge | [insufficient]:F:Decline
```

### Short Node Type Forms
Update ontology.json to include:
```json
{
  "nodeTypes": {
    "SYS": { "short": "S", "description": "System" },
    "UC": { "short": "U", "description": "Use Case" },
    "FUNC": { "short": "F", "description": "Function" },
    "REQ": { "short": "R", "description": "Requirement" },
    "TEST": { "short": "T", "description": "Test" }
  }
}
```

## ⚠️ Known Issues

### Testing Challenges
1. **Fake Test Generation**: LLMs often generate placeholder tests without admitting it
2. **Environment Issues**: Frequent venv/dependency problems prevent test execution
3. **System Validation Gap**: Use case validation remains poorly designed

### Recommendations
- Always verify generated tests actually run
- Maintain separate test documentation in flow.md
- Focus on business flow testing over unit tests

## 🎨 Future: grphzr Integration

**grphzr + ClaudeGraph v3 = Visual Compliance Dashboard**

- grphzr provides the visualization frontend
- ClaudeGraph v3 provides the compliance backend
- Together: Real-time architectural governance

## 📝 Summary

ClaudeGraph v3 proves that **simpler is better**:
- No databases, no AST, no complex tooling
- Just markdown + Claude's vector search
- Focus on business architecture, not technical details
- Compliance checking provides real value

**The best graph database is no graph database - just markdown and semantic search.**

---

*ClaudeGraph v3: Because architecture should be simple to document and hard to violate.*