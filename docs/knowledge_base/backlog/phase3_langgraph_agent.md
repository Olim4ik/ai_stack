# Backlog — LangGraph Agent

**Phase**: 3
**Service**: LangGraph Agent (gRPC, port 50053)
**Plan**: [plan_langgraph_agent.md](../plan_langgraph_agent.md)

---

## Tasks

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Define `agent.proto` with `Chat` and `ResumeChat` RPCs | DONE | `proto/agent.proto` — streaming responses |
| 2 | Generate Python gRPC stubs | DONE | `services/agent/src/generated/` (agent + retrieval + embedding) |
| 3 | Define `AgentState` TypedDict and graph state schema | DONE | `src/state.py` — 10 fields incl. messages, plan, trace |
| 4 | Implement `classify` node — LLM-based query classification | DONE | `nodes/classify.py` — Claude structured output, fallback parsing |
| 5 | Implement `retrieve` node — call Retrieval Service via gRPC | DONE | `nodes/retrieve.py` — calls RetrievalClient |
| 6 | Implement `plan` node — LLM decomposes query into sub-tasks | DONE | `nodes/plan.py` — SEARCH: and TOOL: prefixed steps |
| 7 | Implement `execute` node — loop over plan, call retrieve/tools | DONE | `nodes/execute.py` — iterates plan, confirmation check |
| 8 | Implement `synthesize` node — generate cited answer from context | DONE | `nodes/synthesize.py` — Claude with context + source citations |
| 9 | Build LangGraph graph with conditional routing | DONE | `graph/builder.py` — StateGraph with 6 nodes, conditional edges |
| 10 | Integrate MCP client — connect to MCP Server, discover tools | DONE | `mcp_client/client.py` — stdio_client wrapper, tool discovery |
| 11 | Implement `confirm` node with interrupt_before for human-in-the-loop | DONE | `nodes/confirm.py` + `interrupt_before=["confirm"]` in builder |
| 12 | Implement AgentService gRPC servicer with streaming | DONE | `src/service.py` — Chat + ResumeChat with event streaming |
| 13 | Add session memory management | DONE | In-memory `_sessions` dict, LangGraph message accumulator |
| 14 | Add reasoning trace emission at each node | DONE | Each node appends to `reasoning_trace`, streamed as events |
| 15 | Implement pydantic-settings configuration | DONE | `src/config.py` |
| 16 | Write unit tests for individual nodes | DONE | `tests/unit/test_agent_router.py` — 8 routing tests passing |
| 17 | Create Dockerfile | DONE | `services/agent/Dockerfile` |
| 18 | Add service to `docker-compose.yml` | DONE | depends_on: retrieval, mcp-server |

---

## Progress Log

| Date | Update |
|------|--------|
| 2026-03-04 | All 18 tasks complete. LangGraph agent with 6 nodes (classify → route → retrieve/plan/execute → confirm → synthesize), conditional routing, MCP client integration, human-in-the-loop via interrupt_before, session memory, reasoning trace, gRPC streaming service, 8 unit tests passing, Dockerfile, docker-compose entry. |
