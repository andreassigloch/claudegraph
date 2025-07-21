# Flow Update Command

**Author:** Andreas Sigloch (andreas@siglochconsulting.com)

You are helping a user update their flow.md documentation while maintaining proper business-level abstraction. Focus on capturing new business processes without getting into technical implementation details.

## Your Task

1. **Understand the changes** - Ask user what new functionality they've added
2. **Read current flow.md** to understand existing architecture
3. **Update flows and schemas** to reflect new business capabilities
4. **Maintain abstraction level** - keep focus on business processes, not code structure

## Update Guidelines

### When to Add New Flows
- New user-facing features or capabilities
- New business processes or workflows  
- Changes in how users accomplish their goals
- New integrations that affect business logic

### When to Update Schemas
- New data fields that matter to business logic
- Changes in data validation rules
- New status values or enums
- Modified data relationships

### When NOT to Update
- Pure code refactoring without business impact
- Technical infrastructure changes
- Performance optimizations
- Bug fixes that don't change business logic

## Abstraction Rules

### Keep Business Focus
- ✅ `F:ProcessPayment` → Business operation
- ❌ `F:validate_credit_card_luhn` → Technical implementation
- ✅ `F:NotifyUser` → Business outcome  
- ❌ `F:send_smtp_email` → Technical detail

### Schema Abstraction
- ✅ `S:Order{total:float, status:enum[pending,paid,shipped]}` → Business data
- ❌ `S:OrderTable{id:uuid, created_at:timestamp, updated_at:timestamp}` → Technical storage
- ✅ `S:User{role:enum[admin,user]}` → Business roles
- ❌ `S:UserSession{jwt_token:str, expiry:int}` → Technical session details

## Update Process

1. **Ask clarifying questions:**
   - "What new capabilities can users now do?"
   - "How does this change the business process?"
   - "What new data do you need to track?"
   - "Does this affect any existing flows?"

2. **Propose specific updates:**
   - Show exact text additions/changes to flow.md
   - Explain why each change maintains business abstraction
   - Highlight any flows that might be affected

3. **Validate business logic:**
   - Ensure new flows connect logically to existing ones
   - Check that schemas support the new business processes
   - Confirm abstractions remain meaningful to business stakeholders

## Update Format

Present changes clearly:

```markdown
## Proposed Updates to flow.md

### New Schema Additions:
```
S:NewEntity{field:type, field:type}
```

### Modified Schemas:
```
S:ExistingEntity{existing_fields, new_field:type}
```

### New Flows:
```
Flow:NewProcess -> F:Step1(S:Schema) -> F:Step2(S:Schema)
```

### Modified Flows:
```
Flow:ExistingProcess -> F:Step1(S:Schema) -> F:NewStep(S:NewSchema) -> F:Step3(S:Schema)
```
```

## Success Criteria

- Documentation stays current with business capabilities
- Abstraction level remains consistent
- New flows integrate logically with existing architecture
- Business stakeholders can understand the documented processes
- Future `/flow-check` commands will validate against updated architecture

Ask the user what new functionality they've implemented and how it affects their business processes.