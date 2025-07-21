# Flow Check Command

**Author:** Andreas Sigloch (andreas@siglochconsulting.com)

You are analyzing code changes for architectural compliance violations against the documented business flows. Focus on detecting schema drift, flow violations, and missing documentation.

## Your Task

1. **Read the flow.md file** in the project root to understand documented flows, functions, and schemas
2. **Analyze recent code changes** to identify compliance issues
3. **Report violations** with specific recommendations
4. **Suggest updates** to maintain architectural integrity

## Analysis Areas

### Schema Drift Detection
- Compare code data structures with documented schemas
- Identify missing fields, type mismatches, or new fields not in schemas
- Check for enum values used in code but not documented

### Flow Violation Detection  
- Look for function calls that bypass documented business flows
- Identify direct database access that should go through business functions
- Find business logic scattered outside of documented function boundaries

### Missing Documentation
- Identify new business processes that aren't documented in flows
- Find functions that handle business logic but aren't in flow.md
- Detect new data structures that need schema definitions

### Implementation Drift
- Flag when code uses technical implementation details instead of business abstractions
- Identify when business functions are broken down into too-technical subfunctions

## Report Format

Structure your findings as:

```markdown
# Flow Compliance Report

## ✅ Compliant Areas
- [List areas that follow documented flows correctly]

## ⚠️ Schema Violations
- **Issue**: [Specific problem]
- **Location**: [File/function where found]  
- **Recommendation**: [Specific fix needed]

## ⚠️ Flow Violations
- **Issue**: [Business process bypassed]
- **Location**: [Where the violation occurs]
- **Recommendation**: [How to fix]

## 📝 Missing Documentation
- **New Pattern**: [Undocumented business process]
- **Recommendation**: [Suggested flow/schema additions]

## 🔧 Quick Fixes
1. [Immediate actionable items]
2. [Priority order for fixes]
```

## Guidelines

- **Focus on business impact**, not code style
- **Be specific** - provide exact file locations and line numbers when possible
- **Suggest concrete fixes** - don't just identify problems
- **Prioritize** - distinguish between critical violations and minor drift
- **Stay business-focused** - ignore purely technical implementation details

## Success Criteria

- User understands which changes violate their architectural intentions
- Clear path forward for maintaining compliance
- Balance between flexibility and governance
- Actionable recommendations that preserve business abstraction

Start by reading the project's flow.md file to understand the documented architecture.