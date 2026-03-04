# Service Plan — MCP Server

**Service**: MCP Server
**Type**: stdio / SSE
**Port**: — (stdio) or 8001 (SSE)
**Build Phase**: Phase 4 (Week 4)
**Parent Plan**: [implementation_plan.md](implementation_plan.md)

---

## 1. Purpose

The MCP (Model Context Protocol) Server exposes external tool integrations — Jira, GitHub, PagerDuty, and Knowledge Base search — as structured tools that the LangGraph Agent can discover and invoke. It follows the MCP specification, allowing the agent to dynamically list available tools and call them with validated inputs.

---

## 2. Responsibilities

- Expose tools following the MCP specification (tool name, description, JSON Schema input)
- Handle tool invocations from the LangGraph Agent
- Interact with external APIs (Jira, GitHub, PagerDuty)
- Wrap the Retrieval Service as an MCP tool for agent self-querying
- Return structured results (or structured errors) to the agent
- Validate tool inputs against defined schemas

---

## 3. Transport Options

| Transport | How It Works                                     | When to Use                        |
|-----------|--------------------------------------------------|------------------------------------|
| **stdio** | Agent spawns MCP server as subprocess, communicates via stdin/stdout | Development, single-container deploy |
| **SSE**   | MCP server runs as HTTP service, agent connects via SSE | Multi-container Docker Compose deploy |

**Default**: stdio for simplicity. Switch to SSE when deploying as a separate container.

---

## 4. Tools

### 4.1 Jira Tools

#### `jira_search`
Search Jira tickets by JQL query.

```json
{
  "name": "jira_search",
  "description": "Search Jira tickets using JQL query syntax",
  "inputSchema": {
    "type": "object",
    "properties": {
      "jql": { "type": "string", "description": "JQL query (e.g., 'project = PLAT AND status = Open')" },
      "max_results": { "type": "integer", "default": 10 }
    },
    "required": ["jql"]
  }
}
```

**Returns**: List of tickets with key, summary, status, assignee, priority.

#### `jira_create`
Create a new Jira ticket.

```json
{
  "name": "jira_create",
  "description": "Create a new Jira ticket",
  "inputSchema": {
    "type": "object",
    "properties": {
      "project": { "type": "string", "description": "Project key (e.g., 'PLAT')" },
      "summary": { "type": "string" },
      "description": { "type": "string" },
      "issue_type": { "type": "string", "default": "Task" },
      "priority": { "type": "string", "default": "Medium" }
    },
    "required": ["project", "summary"]
  }
}
```

**Returns**: Created ticket key and URL. **Requires human-in-the-loop confirmation.**

### 4.2 GitHub Tools

#### `github_get_pr`
Get pull request details.

```json
{
  "name": "github_get_pr",
  "description": "Get details of a GitHub pull request",
  "inputSchema": {
    "type": "object",
    "properties": {
      "repo": { "type": "string", "description": "Repository in 'owner/repo' format" },
      "pr_number": { "type": "integer" }
    },
    "required": ["repo", "pr_number"]
  }
}
```

**Returns**: PR title, author, status, description, changed files, review status.

#### `github_get_file`
Read a file from a GitHub repository.

```json
{
  "name": "github_get_file",
  "description": "Read a file's contents from a GitHub repository",
  "inputSchema": {
    "type": "object",
    "properties": {
      "repo": { "type": "string", "description": "Repository in 'owner/repo' format" },
      "path": { "type": "string", "description": "File path within the repo" },
      "ref": { "type": "string", "default": "main", "description": "Branch or commit SHA" }
    },
    "required": ["repo", "path"]
  }
}
```

**Returns**: File contents as text.

### 4.3 PagerDuty Tools

#### `pagerduty_incidents`
List current open incidents.

```json
{
  "name": "pagerduty_incidents",
  "description": "List current open PagerDuty incidents",
  "inputSchema": {
    "type": "object",
    "properties": {
      "status": { "type": "string", "enum": ["triggered", "acknowledged", "resolved"], "default": "triggered" },
      "service_name": { "type": "string", "description": "Filter by service name (optional)" }
    }
  }
}
```

**Returns**: List of incidents with ID, title, status, service, urgency, created_at.

#### `pagerduty_ack`
Acknowledge a PagerDuty incident.

```json
{
  "name": "pagerduty_ack",
  "description": "Acknowledge a PagerDuty incident",
  "inputSchema": {
    "type": "object",
    "properties": {
      "incident_id": { "type": "string" }
    },
    "required": ["incident_id"]
  }
}
```

**Returns**: Confirmation of acknowledgment. **Requires human-in-the-loop confirmation.**

### 4.4 Knowledge Base Tool

#### `knowledge_search`
Search the internal knowledge base (wraps the Retrieval Service).

```json
{
  "name": "knowledge_search",
  "description": "Search the internal knowledge base for relevant documents",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": { "type": "string" },
      "team": { "type": "string", "description": "Team collection to search" },
      "top_k": { "type": "integer", "default": 5 }
    },
    "required": ["query"]
  }
}
```

**Returns**: Ranked list of document chunks with text, source, and relevance score.

---

## 5. Tool Annotations

Each tool declares metadata to help the agent decide when human confirmation is needed:

```python
@server.tool(
    name="jira_create",
    description="Create a new Jira ticket",
    annotations={
        "requires_confirmation": True,     # Agent should trigger human-in-the-loop
        "side_effects": "write",           # "read" or "write"
    }
)
```

The agent reads `requires_confirmation` and routes through the confirm node when `True`.

---

## 6. Error Handling

All tool errors are returned as structured MCP error responses, not exceptions:

```python
# Success
return [TextContent(type="text", text=json.dumps(result))]

# Error
return [TextContent(type="text", text=json.dumps({
    "error": True,
    "code": "JIRA_AUTH_FAILED",
    "message": "Failed to authenticate with Jira API"
}))]
```

The agent receives errors as tool results and can decide to retry, use a fallback, or report the error to the user.

---

## 7. Directory Structure

```
services/mcp_server/
├── Dockerfile
├── requirements.txt
└── src/
    ├── main.py                   # MCP server entry point
    ├── config.py                 # Configuration
    └── tools/
        ├── jira.py               # jira_search, jira_create
        ├── github.py             # github_get_pr, github_get_file
        ├── pagerduty.py          # pagerduty_incidents, pagerduty_ack
        └── knowledge.py          # knowledge_search (calls Retrieval Service)
```

---

## 8. Dependencies

```
mcp>=1.0.0
httpx>=0.27.0
grpcio>=1.60.0                    # For knowledge_search tool calling Retrieval Service
pydantic>=2.0
pydantic-settings>=2.0
structlog>=24.0
```

---

## 9. Configuration

| Variable              | Default     | Description                            |
|-----------------------|-------------|----------------------------------------|
| `MCP_TRANSPORT`       | `stdio`     | `stdio` or `sse`                       |
| `MCP_SSE_PORT`        | `8001`      | Port when using SSE transport          |
| `JIRA_BASE_URL`       | —           | Jira instance URL                      |
| `JIRA_EMAIL`          | —           | Jira account email                     |
| `JIRA_API_TOKEN`      | —           | Jira API token                         |
| `GITHUB_TOKEN`        | —           | GitHub personal access token           |
| `PAGERDUTY_API_KEY`   | —           | PagerDuty API key                      |
| `RETRIEVAL_SERVICE_HOST` | `localhost:50051` | Retrieval service gRPC address |

---

## 10. Implementation Steps

1. [ ] Scaffold MCP server with `mcp` library and stdio transport
2. [ ] Implement `jira_search` tool — JQL query via Jira REST API
3. [ ] Implement `jira_create` tool — create ticket via Jira REST API
4. [ ] Implement `github_get_pr` tool — fetch PR via GitHub REST API
5. [ ] Implement `github_get_file` tool — fetch file contents via GitHub API
6. [ ] Implement `pagerduty_incidents` tool — list incidents via PagerDuty API
7. [ ] Implement `pagerduty_ack` tool — acknowledge incident via PagerDuty API
8. [ ] Implement `knowledge_search` tool — call Retrieval Service via gRPC
9. [ ] Add tool annotations (`requires_confirmation`, `side_effects`)
10. [ ] Add structured error handling for all tools
11. [ ] Add SSE transport option for Docker deployment
12. [ ] Write unit tests (mock external APIs)
13. [ ] Write integration test with MCP client
14. [ ] Create Dockerfile

---

## 11. Key Design Decisions

| Decision                          | Choice              | Rationale                                              |
|-----------------------------------|----------------------|--------------------------------------------------------|
| MCP library                       | Official `mcp` SDK   | Standard implementation, well-supported                |
| Default transport                 | stdio                | Zero-config for development, agent spawns server       |
| External API client               | `httpx`              | Async, modern, well-maintained HTTP client             |
| Confirmation annotations          | Tool-level metadata  | Agent reads metadata to decide on human-in-the-loop    |
| Knowledge search as MCP tool      | Yes                  | Agent can self-query the KB as part of multi-step plans |
