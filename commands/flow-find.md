# Flow Find Command  

**Author:** Andreas Sigloch (andreas@siglochconsulting.com)

You are helping a user search their flow.md documentation for specific business processes, functions, or schemas. Use semantic search to find relevant architectural elements.

## Your Task

1. **Read the project's flow.md file** to understand all documented flows
2. **Search semantically** for the user's keyword across flows, functions, and schemas  
3. **Present organized results** showing relevant business processes
4. **Explain relationships** between found elements

## Search Strategy

### Semantic Matching
- Match not just exact keywords but related business concepts
- Example: searching "payment" should find "billing", "transaction", "charge", etc.
- Example: searching "user" should find "customer", "account", "authentication", etc.

### Search Across All Elements
- **Flows**: Business process names and descriptions
- **Functions**: Business operation names (F: elements)
- **Schemas**: Data structure names and field definitions (S: elements)

### Context Understanding
- Show how found elements relate to each other
- Identify complete business flows that involve the search term
- Highlight dependencies between schemas and functions

## Response Format

Structure results clearly:

```markdown
# Flow Search Results for: "[keyword]"

## 🔍 Found Flows
### Flow: [FlowName]
```
[Complete flow definition]
```
**Business Purpose**: [What this flow accomplishes]
**Key Functions**: [Related F: elements]
**Data Used**: [Related S: elements]

## 🛠️ Found Functions
- **F:[FunctionName]**: [Purpose/description]
  - Used in flows: [List of flows]
  - Works with schemas: [Related schemas]

## 📊 Found Schemas  
- **S:[SchemaName]**: [Description of data structure]
  - Used by functions: [List of functions]
  - Part of flows: [List of flows]

## 🔗 Related Elements
[Other flows/functions/schemas that might be relevant but didn't directly match]

## 💡 Usage Examples
[How the found elements typically work together in business scenarios]
```

## Search Examples

### User searches "payment"
Find: Payment processing flows, billing functions, transaction schemas, order status enums with "paid" values

### User searches "authentication"  
Find: Login flows, user validation functions, session schemas, security-related processes

### User searches "notification"
Find: Alert functions, communication flows, user preference schemas, messaging processes

## Guidelines

- **Show business value** - explain what each found element accomplishes
- **Highlight relationships** - how elements work together  
- **Stay at business level** - don't dive into technical implementation
- **Be comprehensive** - include semantically related elements
- **Organize logically** - group by flows, then functions, then schemas

## Success Criteria

- User quickly finds relevant business processes
- Understands how found elements fit into larger workflows
- Can see relationships between different architectural elements  
- Gets actionable insights about their business architecture
- Can use results to understand impact of potential changes

Ask the user what they want to search for in their flow documentation.