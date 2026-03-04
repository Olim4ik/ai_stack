# Backlog — MCP Server

**Phase**: 3
**Service**: MCP Server (stdio / SSE)
**Plan**: [plan_mcp_server.md](../plan_mcp_server.md)

---

## Tasks

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Scaffold MCP server with `mcp` library and stdio transport | DONE | `services/mcp_server/src/main.py` — Server with list_tools + call_tool |
| 2 | Implement `jira_search` tool — JQL query via Jira REST API | DONE | `tools/jira.py` — httpx async client, basic auth |
| 3 | Implement `jira_create` tool — create ticket via Jira REST API | DONE | `tools/jira.py` — Jira v3 API doc format |
| 4 | Implement `github_get_pr` tool — fetch PR via GitHub REST API | DONE | `tools/github.py` — Bearer auth, full PR details |
| 5 | Implement `github_get_file` tool — fetch file contents via GitHub API | DONE | `tools/github.py` — base64 decode, 10KB limit |
| 6 | Implement `pagerduty_incidents` tool — list incidents via PagerDuty API | DONE | `tools/pagerduty.py` — token auth, service filter |
| 7 | Implement `pagerduty_ack` tool — acknowledge incident via PagerDuty API | DONE | `tools/pagerduty.py` — PUT status update |
| 8 | Implement `knowledge_search` tool — call Retrieval Service via gRPC | DONE | `tools/knowledge.py` — lazy gRPC stub |
| 9 | Add tool annotations (requires_confirmation, side_effects) | DONE | `CONFIRMATION_REQUIRED` set + `_requires_confirmation` flag in results |
| 10 | Add structured error handling for all tools | DONE | All tools return JSON `{"error": true, "code": ..., "message": ...}` |
| 11 | Implement pydantic-settings configuration | DONE | `src/config.py` |
| 12 | Add SSE transport option for Docker deployment | DONE | `main.py` — Starlette SSE server via `MCP_TRANSPORT=sse` |
| 13 | Write unit tests (mock external APIs) | DONE | `tests/unit/test_mcp_tools.py` — 7 tests passing |
| 14 | Create Dockerfile | DONE | `services/mcp_server/Dockerfile` |

---

## Progress Log

| Date | Update |
|------|--------|
| 2026-03-04 | All 14 tasks complete. MCP server with 7 tools (jira_search, jira_create, github_get_pr, github_get_file, pagerduty_incidents, pagerduty_ack, knowledge_search), stdio + SSE transports, confirmation annotations, structured error handling, 7 unit tests passing. |
