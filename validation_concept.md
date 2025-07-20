# ClaudeGraph v3 A/B Validation Concept

## Objective
Validate that ClaudeGraph v3's Flow-based Context Management delivers measurable improvements for UC1-4 through controlled A/B testing.

## Methodology

### Test Setup
- **Group A (Control)**: Claude instance without flow.md context
- **Group B (Treatment)**: Claude instance with flow.md context pre-loaded
- **Test Project**: AssistantTestClaude (real-world Flask/Neo4j application)

### Metrics to Measure
1. **Token Usage**: Total tokens consumed per task
2. **Accuracy**: Correctness of implementation/analysis
3. **Time to Solution**: Steps/iterations needed
4. **Error Rate**: Schema mismatches, wrong function calls
5. **Context Switches**: How often Claude needs to search/read files

## Test Scenarios

### UC1: Data Format Forgotten
**Task**: "Update the visualization_assistant to accept a new field 'chart_type' in its input"

**Expected Outcomes**:
- Group A: Searches multiple files to find schema, might introduce inconsistencies
- Group B: Immediately sees VisualizationRequest schema, maintains consistency

**Measurements**:
- Token count for schema discovery
- Schema consistency in final implementation
- Number of files searched

### UC2: System Validation Missed  
**Task**: "Add a new function to process user feedback in the GraphQuery flow"

**Expected Outcomes**:
- Group A: Adds function but might miss flow integration points
- Group B: Sees complete GraphQuery flow, properly integrates

**Measurements**:
- Completeness of flow integration
- Tests added for flow validation
- Edge cases handled

### UC3: Token-Intensive Searches
**Task**: "Find all places where email notifications are sent and add logging"

**Expected Outcomes**:
- Group A: Searches entire codebase with grep/find
- Group B: Uses `flow find email` to locate instantly

**Measurements**:
- Total tokens used
- Number of search operations
- Time to locate all instances

### UC4: Function Names Forgotten
**Task**: "Update the function that handles Neo4j query execution to add caching"

**Expected Outcomes**:
- Group A: Searches for various query-related terms
- Group B: Finds ExecuteCypher in GraphQuery flow immediately

**Measurements**:
- Search attempts before finding correct function
- Token usage for discovery
- Accuracy of function identification

## Execution Plan

### 1. Test Script Structure
```python
# validation_runner.py
class ValidationTest:
    def __init__(self, test_name, task_prompt):
        self.test_name = test_name
        self.task_prompt = task_prompt
        self.metrics = {
            "tokens": 0,
            "files_searched": 0,
            "time_seconds": 0,
            "errors": [],
            "accuracy_score": 0
        }
```

### 2. Prompts for Each Group

**Group A Prompt Template**:
```
You are working on the AssistantTestClaude project at /Users/andreas/Documents/Tools/VSCode/AssistantTestClaude.
Task: {task_description}
Please complete this task.
```

**Group B Prompt Template**:
```
You are working on the AssistantTestClaude project at /Users/andreas/Documents/Tools/VSCode/AssistantTestClaude.

Here is the project's flow context:
{flow_md_content}

Task: {task_description}
Please complete this task using the flow information provided.
```

### 3. Success Criteria

**UC1 Success**: 
- B uses 80% fewer tokens for schema discovery
- B maintains 100% schema consistency

**UC2 Success**:
- B identifies all integration points correctly
- B adds proper flow validation

**UC3 Success**:
- B uses 90% fewer tokens (matches our 35.6k→3.5k claim)
- B finds all instances faster

**UC4 Success**:
- B finds function in 1-2 attempts vs A's 5+ attempts
- B uses 70% fewer search tokens

## Implementation Steps

1. Create `validation_runner.py` to automate tests
2. Design controlled prompts for each UC
3. Create evaluation rubrics for accuracy scoring
4. Run each test scenario 3 times for statistical validity
5. Generate comparison report with visualizations

## Expected Results Summary

| Use Case | Metric | Group A (No Flow) | Group B (With Flow) | Improvement |
|----------|--------|-------------------|---------------------|-------------|
| UC1 | Token Usage | ~5,000 | ~1,000 | 80% |
| UC1 | Schema Errors | 2-3 | 0 | 100% |
| UC2 | Integration Completeness | 60% | 100% | 40pp |
| UC3 | Search Tokens | ~35,000 | ~3,500 | 90% |
| UC4 | Search Attempts | 5-8 | 1-2 | 75% |

## Next Steps
1. Implement validation_runner.py
2. Create standardized evaluation rubrics
3. Execute tests with real Claude instances
4. Document results with screenshots/logs
5. Create visualization of A/B results