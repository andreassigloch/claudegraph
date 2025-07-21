# ClaudeGraph v3

**Author:** Andreas Sigloch (andreas@siglochconsulting.com)

**Ultra-Simple Architectural Compliance for Claude Code**

ClaudeGraph v3 is a markdown-based compliance system that ensures code changes don't violate documented business flows. It uses simple Flow → Function → Schema definitions that leverage Claude's powerful vector search capabilities.

## 🎯 Core Concepts

### 1. Flows
Business processes as function chains:
```markdown
Flow:UserRegistration -> F:ValidateInput(S:UserData) -> F:CreateAccount(S:Account) -> F:SendWelcome(S:Email)
```

### 2. Functions  
Business operations (NOT implementation details):
- ✅ `F:ProcessPayment` (what it does)
- ❌ `F:_calculate_tax()` (how it's coded)

### 3. Schemas
Data contracts with optional fields and enums:
```markdown
S:Order{id:int, items:list, total:float, status:enum[pending,paid,shipped], notes:str?}
```

## 📋 Commands

### `/flow-init`
Interactive setup for new projects - creates initial flow.md based on your use cases

### `/flow-check`
Analyze code changes for compliance violations
```bash
/flow-check
# Reports schema drift, flow breaks, missing abstractions
```

### `/flow-update` 
Update flow.md while maintaining business abstraction
```bash
/flow-update
# Ensures proper documentation without technical details
```

### `/flow-find <keyword>`
Search flows, functions, and schemas
```bash
/flow-find payment
# Finds all payment-related business logic
```

## 🛠️ Setup

1. **Initialize your project flows:**
```bash
/flow-init
# Interactive setup - define your use cases and flows
```

2. **Copy commands to `.claude/commands/`:**
```bash
cp commands/*.md /path/to/project/.claude/commands/
```

3. **Use in Claude Code:**
```bash
/flow-check    # Check compliance
/flow-update   # Update documentation  
/flow-find api # Find API-related flows
```

## 💡 Real Value: Compliance Checking

ClaudeGraph v3 provides **architectural governance**:

1. **Schema Drift Detection**: Identifies when code diverges from documented contracts
2. **Flow Violation Alerts**: Catches when functions are called outside defined flows  
3. **Missing Documentation**: Highlights undocumented business processes
4. **Impact Analysis**: Shows which flows are affected by changes

### Example
```bash
/flow-check

# Output:
Schema Violations:
- S:VisualizationRequest missing chart_type field
- Chart type validation flow not documented

Recommendations:
- Update S:VisualizationRequest{..., chart_type:enum[bar,matrix,network]?}
- Add F:ValidateChartType to visualization flow
```

## 📝 Why v3?

**Simple is better:**
- No databases, no AST parsing, no complex tooling
- Just markdown + Claude's vector search
- Focus on business architecture, not technical details
- Compliance checking provides real value

**The best graph database is no graph database - just markdown and semantic search.**

## License

MIT License - See LICENSE file for details.

---

*ClaudeGraph v3: Because architecture should be simple to document and hard to violate.*