# 🧪 ClaudeGraph v3 Real A/B Test Prompts - READY TO COPY

## Test Execution Plan

1. **Open TWO fresh Claude Code sessions**
2. **Copy Group A prompt to Session 1** 
3. **Copy Group B prompt to Session 2**
4. **Measure and compare results**

---

## 📋 GROUP A PROMPT (NO FLOW CONTEXT)

```
You are working on the AssistantTestClaude project at /Users/andreas/Documents/Tools/VSCode/AssistantTestClaude.

Your task is to complete the following:

Please update the visualization_assistant function in src/assistant/llm_tools/visualization_assistant.py to accept a new optional field 'chart_type' in its input. The field should:
1. Be an optional string that defaults to 'bar' if not provided
2. Be added to the VisualizationRequest schema/interface
3. Be used in the visualization generation logic

Make sure to update all places where VisualizationRequest is used to handle this new field.

Please implement this task. Show me what files you're searching and what changes you're making.
```

---

## 📋 GROUP B PROMPT (WITH FLOW CONTEXT)

```
exit
```

---

## 📊 MEASUREMENT CHECKLIST

For each test, count:

**Tool Usage:**
- [ ] Read tool calls
- [ ] Grep tool calls  
- [ ] Glob tool calls
- [ ] Edit/Write tool calls

**Navigation Efficiency:**
- [ ] Files searched before finding target
- [ ] Search terms used
- [ ] Time to locate VisualizationRequest schema

**Expected Differences:**
- **Group A**: Multiple search operations, 5-8 files read
- **Group B**: Direct navigation via flow context, 2-3 files read

Ready for real validation! 🚀