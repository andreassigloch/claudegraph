# ClaudeGraph v3: Flow Definitions

## Schemas

S:UserData{username:str, email:str, password_hash:str}
S:EmailTemplate{to:str, subject:str, body:str, template_id:str}
S:AuthToken{user_id:int, token:str, expires_at:datetime}
S:GraphData{nodes:list, edges:list, metadata:dict}
S:QueryRequest{pattern:str, filters:dict, limit:int}
S:QueryResult{data:list, count:int, execution_time:float}

## Flows

Flow:UserRegistration -> F:ValidateInput(S:UserData) -> F:CreateUser(S:UserData) -> F:SendEmail(S:EmailTemplate)

Flow:UserAuthentication -> F:ValidateCredentials(S:UserData) -> F:GenerateToken(S:AuthToken) -> F:StoreSession(S:AuthToken)

Flow:GraphQuery -> F:ParseQuery(S:QueryRequest) -> F:ExecuteCypher(S:QueryRequest) -> F:FormatResults(S:QueryResult)

Flow:GraphVisualization -> F:LoadGraph(S:GraphData) -> F:ApplyLayout(S:GraphData) -> F:RenderNodes(S:GraphData)

Flow:ImpactAnalysis -> F:FindComponent(S:QueryRequest) -> F:TraceConnections(S:GraphData) -> F:GenerateReport(S:QueryResult)

## Example Usage

```bash
# Check for schema conflicts
python flow_manager.py check

# Find functions by keyword  
python flow_manager.py find "email"

# Generate compact context for Claude
python flow_manager.py context current_file.py
```

This ultra-simple approach replaces:
- ❌ Complex ontology with 9 node types
- ❌ Docker + Neo4j + ChromaDB setup  
- ❌ 35.6k token searches

With:
- ✅ 3 concepts: Flow → Function → Schema
- ✅ One markdown file
- ✅ ~10k token context generation
- ✅ Instant keyword search
                 