# 🧪 Real ClaudeGraph v3 A/B Test - Execution Instructions

## Test Setup

You'll run **two separate Claude Code sessions** to compare performance with and without flow context.

### Project Location
```
/Users/andreas/Documents/Tools/VSCode/AssistantTestClaude
```

### Test Scenario: UC1 - Add chart_type field to visualization_assistant

## Group A Test (WITHOUT Flow Context)

### Group A Prompt:
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

### What to Measure for Group A:
- [ ] Number of tool calls (Read, Grep, Glob)
- [ ] Files searched before finding target
- [ ] Total response tokens
- [ ] Time to completion
- [ ] Search attempts to locate VisualizationRequest schema

## Group B Test (WITH Flow Context)

### Group B Prompt:
```
You are working on the AssistantTestClaude project at /Users/andreas/Documents/Tools/VSCode/AssistantTestClaude.

Here is the project's flow documentation that shows the architecture:

# AiSE Application Flow - ClaudeGraph v3 Format

## User Interface Flows

Flow:WebRequest -> F:index(S:HttpRequest) -> F:render_template(S:HtmlResponse)
Flow:StaticAssets -> F:send_static(S:FilePath) -> F:send_from_directory(S:FileResponse)

## System Management Flows

Flow:ListSystems -> F:get_systems(S:HttpRequest) -> F:get_available_systems(S:Query) -> F:cypher_write(S:CypherQuery) -> F:json_response(S:SystemList)
Flow:CreateSystem -> F:create_system_route(S:SystemData) -> F:create_system(S:SystemNode) -> F:cypher_write(S:CypherQuery) -> F:json_response(S:SystemNode)
Flow:DeleteSystem -> F:delete_system_route(S:SystemUuid) -> F:delete_system(S:SystemUuid) -> F:cypher_write(S:CypherQuery) -> F:json_response(S:Success)

## View Management Flows

Flow:GetViews -> F:get_views(S:SystemUuid) -> F:get_system_views(S:SystemUuid) -> F:cypher_write(S:CypherQuery) -> F:json_response(S:ViewList)
Flow:CreateView -> F:create_view_route(S:ViewData) -> F:create_view(S:ViewNode) -> F:cypher_write(S:CypherQuery) -> F:json_response(S:ViewNode)
Flow:UpdateView -> F:update_view(S:ViewLayout) -> F:update_view_layout(S:LayoutData) -> F:cypher_write(S:CypherQuery) -> F:json_response(S:Success)

## LLM Tool Processing Flows

Flow:QueryTool -> F:query(S:QueryRequest) -> F:store_conversation_message(S:UserMessage) -> F:tool_service.use_tool(S:ToolConfig) -> F:ki_tool_function(S:ToolParams) -> F:standardize_response(S:ToolResponse) -> F:validate_and_log_result(S:GraphData) -> F:store_conversation_message(S:AssistantMessage) -> F:json_response(S:GraphResponse)

Flow:EngineeringAssistant -> F:engineering_assistant(S:QueryContent) -> F:get_provider(S:ProviderConfig) -> F:decide_tool(S:LlmResponse) -> F:create_system_node|batch_generate_use_cases|batch_generate_requirements|batch_generate_functions|search_agent(S:ToolParams) -> F:format_response(S:GraphData)

Flow:SearchAgent -> F:search_agent.execute(S:QueryContent) -> F:extract_system_uuid(S:Content) -> F:get_conversation_summary(S:SystemUuid) -> F:generate_cypher_query(S:NaturalQuery) -> F:neo4j_service.execute_cypher(S:CypherQuery) -> F:format_results(S:Neo4jData) -> F:check_visualization_request(S:Query) -> F:visualization_assistant(S:VisualizationRequest) -> F:format_response(S:SearchResults)

## Graph Operations Flows

Flow:GetGraph -> F:get_graph(S:SystemUuid) -> F:get_entire_graph(S:SystemUuid) -> F:cypher_write(S:CypherQuery) -> F:json_response(S:GraphData)
Flow:UpdateGraph -> F:update_graph_route(S:GraphData) -> F:validate_graph_data(S:GraphData) -> F:update_graph(S:GraphUpdate) -> F:cypher_write(S:CypherQuery) -> F:json_response(S:UpdatedGraph)
Flow:CreateNode -> F:create_node_route(S:NodeData) -> F:create_[type]_node(S:NodeProps) -> F:cypher_write(S:CypherQuery) -> F:validate_graph_data(S:NodeData) -> F:create_node_embedding(S:NodeDict) -> F:json_response(S:NodeDict)
Flow:CreateRelationship -> F:create_relationship_route(S:RelData) -> F:create_[type]_relation(S:RelProps) -> F:cypher_write(S:CypherQuery) -> F:validate_graph_data(S:RelData) -> F:create_relationship_embedding(S:RelDict) -> F:json_response(S:RelDict)

## Conversation Management Flows

Flow:StoreConversation -> F:store_conversation(S:ConversationData) -> F:store_conversation_message(S:MessageData) -> F:cypher_write(S:CypherQuery) -> F:json_response(S:MessageUuid)
Flow:GetConversations -> F:get_system_conversations(S:SystemUuid) -> F:get_conversation_history(S:SystemUuid) -> F:cypher_write(S:CypherQuery) -> F:process_neo4j_datetime(S:ConversationData) -> F:json_response(S:ConversationList)

## Visualization Flows

Flow:CreateVisualization -> F:create_visualization(S:VisualizationRequest) -> F:visualization_assistant(S:QueryData) -> F:generate_vis_id(S:Timestamp) -> F:store_conversation_message(S:PendingVis) -> F:json_response(S:PendingResponse) -> F:async_generate_visualization(S:Neo4jData) -> F:validate_visualization(S:HtmlContent) -> F:update_conversation_visualization(S:VisualizationData)

Flow:ServeVisualization -> F:serve_visualization(S:VisId) -> F:cypher_write(S:FindVisQuery) -> F:render_template|return_html(S:VisualizationHtml)
Flow:CheckVisualizationStatus -> F:get_visualization_status(S:VisId) -> F:get_conversation_visualization(S:MessageUuid) -> F:json_response(S:StatusData)

## WebSocket Communication Flows

Flow:WebSocketConnect -> F:echo_socket(S:WebSocket) -> F:clients.add(S:WebSocket) -> F:ws.receive(S:Message) -> F:clients.remove(S:WebSocket)
Flow:BroadcastUpdate -> F:create_node|create_relationship(S:EntityData) -> F:broadcast_message(S:UpdateMessage) -> F:clients.send(S:JsonMessage)

## Validation Flows

Flow:ValidateOntology -> F:validate_ontology_route(S:ValidationRequest) -> F:validator.validate_ontology(S:GraphData) -> F:check_node_rules(S:NodeData) -> F:check_relationship_rules(S:RelData) -> F:calculate_stats(S:Findings) -> F:json_response(S:ValidationResults)

## Audit and Logging Flows

Flow:AuditLog -> F:process_audit_log(S:AuditEvent) -> F:AuditLogger.log_event(S:EventData) -> F:write_audit_file(S:LogEntry) -> F:json_response(S:Success)
Flow:LogActivity -> F:log_activity(S:ActivityData) -> F:cypher_write(S:ActivityQuery) -> F:broadcast_activity(S:ActivityMessage)

## Database Query Flows

Flow:DirectNeo4jQuery -> F:neo4j_query(S:CypherRequest) -> F:driver.session.run(S:CypherQuery) -> F:process_neo4j_records(S:QueryResult) -> F:jsonify(S:RawResult)

## Tool Configuration Flow

Flow:GetTools -> F:get_tools(S:HttpRequest) -> F:list(TOOL_CONFIG.keys()) -> F:json_response(S:ToolList)

## Health Check Flow

Flow:HealthCheck -> F:health_check(S:HttpRequest) -> F:json_response(S:HealthStatus)

## Index Management Flow

Flow:CheckIndexes -> F:check_database_indexes(S:HttpRequest) -> F:check_indexes(S:Neo4jSession) -> F:json_response(S:IndexList)

## Background Processing Flows

Flow:UpdateEmbeddings -> F:update_embeddings_background(S:Thread) -> F:vector_store.update_all_embeddings(S:AllNodes) -> F:create_embedding(S:NodeText) -> F:store_embedding(S:VectorData)

## Key Data Schemas

S:HttpRequest = {method, path, headers, body}
S:QueryRequest = {query: str, stage: str, content: dict}
S:GraphData = {nodes: [], relationships: []}
S:NodeData = {uuid: str, type: str, Name: str, Descr: str, ...props}
S:RelData = {uuid: str, source: str, target: str, type: str, ...props}
S:SystemNode = {uuid: str, type: "SYS", Name: str, Descr: str}
S:ToolConfig = {funcName: str, parameters: [], system: str, user: str, temperature: float, provider_config: dict}
S:CypherQuery = {query: str, parameters: dict}
S:ValidationResults = {stats: dict, findings: [], schema_info: dict}
S:VisualizationRequest = {query: str, data: dict, system_uuid: str}
S:ConversationData = {uuid: str, content: str, role: str, timestamp: datetime}
S:AuditEvent = {event_type: str, system_id: str, payload: str}

Your task is to complete the following:

Please update the visualization_assistant function in src/assistant/llm_tools/visualization_assistant.py to accept a new optional field 'chart_type' in its input. The field should:
1. Be an optional string that defaults to 'bar' if not provided
2. Be added to the VisualizationRequest schema/interface
3. Be used in the visualization generation logic

Make sure to update all places where VisualizationRequest is used to handle this new field.

Please implement this task using the flow information provided above to help you navigate the codebase efficiently.
```

### What to Measure for Group B:
- [ ] Number of tool calls (Read, Grep, Glob)
- [ ] Files searched before finding target
- [ ] Total response tokens
- [ ] Time to completion
- [ ] How quickly VisualizationRequest schema is located

## Expected Results

### Group A (No Flow) Predictions:
- **Search Pattern**: Multiple Grep searches for "visualization", "VisualizationRequest"
- **Files Read**: 5-8 files to locate schema and function
- **Tool Calls**: 8-12 total
- **Tokens**: 4000-6000
- **Time**: 2-3 minutes

### Group B (With Flow) Predictions:
- **Direct Access**: Uses flow info to find F:visualization_assistant(S:VisualizationRequest)
- **Files Read**: 2-3 files maximum
- **Tool Calls**: 3-5 total
- **Tokens**: 1500-2500 (including flow context)
- **Time**: 1 minute or less

## Measurement Template

After each test, record:

```
## Group [A/B] Results
- Tool Calls: [count]
- Files Read: [list]
- Search Operations: [count]
- Total Tokens: [estimate from response length]
- Time: [minutes:seconds]
- Schema Discovery: [how many attempts to find VisualizationRequest]
- Notes: [observations about efficiency]
```

## Key Validation Points

1. **Token Efficiency**: Does Group B use significantly fewer tokens?
2. **Search Reduction**: Does flow context eliminate multiple search attempts?
3. **Schema Discovery**: How quickly is VisualizationRequest schema found?
4. **Direct Navigation**: Does Group B go straight to target files?

Ready to run the real tests? Copy the prompts above into fresh Claude Code sessions and let's get empirical data!