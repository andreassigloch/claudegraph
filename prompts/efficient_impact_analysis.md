# Efficient Impact Analysis Prompt

## When user asks for impact analysis:

1. **FIRST AND ONLY STEP**: Run the impact command directly
   ```bash
   /claudegraph impact [component]
   ```

2. **IF component found**: Present the high-level impact summary
   - Direct impacts (max 5)
   - Transitive impacts (max 5)
   - Affected tests (max 3)

3. **IF component NOT found**: Simply report "Component not found"
   - DO NOT search files
   - DO NOT create todos
   - DO NOT run additional queries
   - DO NOT write reports for non-existent components

## Examples:

### Component Not Found:
```
User: analyze impact of music
Assistant: /claudegraph impact music
Result: Component 'music' not found in architecture database
Response: The 'music' component is not found in the architecture database.
```

### Component Found:
```
User: analyze impact of Database
Assistant: /claudegraph impact Database
Result: [impact data]
Response: Database component impacts:
- Direct: 3 use cases, 2 actors
- Transitive: 5 functional chains
- Tests: 3 verification tests
```

## DO NOT:
- Create todo lists for simple queries
- Search files when architecture query returns "not found"
- Write elaborate reports for non-existent components
- Run multiple queries when one provides the answer
- Over-explain simple results

## TRUST THE TOOL:
The /claudegraph command is optimized to provide complete answers in a single query. Trust its results.