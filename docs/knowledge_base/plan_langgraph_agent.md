# Service Plan — LangGraph Agent

**Service**: LangGraph Agent
**Type**: gRPC Service
**Port**: 50053
**Build Phase**: Phase 3 (Week 3-4)
**Parent Plan**: [implementation_plan.md](implementation_plan.md)

---

## 1. Purpose

The LangGraph Agent is the core intelligence layer. It receives user queries via gRPC, classifies them, orchestrates multi-step reasoning using a stateful graph, invokes tools via MCP, retrieves context from the Retrieval Service, and produces cited, streamed answers.

---

## 2. Responsibilities

- Classify incoming queries (simple lookup / multi-step reasoning / action request)
- Execute a LangGraph state machine to plan and run retrieval + tool workflows
- Call the Retrieval Service (gRPC) for RAG context
- Call MCP tools (Jira, GitHub, PagerDuty) for external actions
- Maintain session memory (short-term within conversation, optional long-term)
- Support human-in-the-loop by pausing the graph and waiting for confirmation
- Stream reasoning trace and response tokens back to the gateway

---

## 3. LangGraph State Machine

### 3.1 Graph Overview

```
                    ┌─────────┐
                    │  START   │
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │ Classify │
                    └────┬────┘
                         │
            ┌────────────┼────────────┐
            │            │            │
       ┌────▼───┐  ┌─────▼─────┐ ┌───▼────┐
       │ Simple │  │ Multi-Step│ │ Action │
       │ Lookup │  │ Reasoning │ │Request │
       └────┬───┘  └─────┬─────┘ └───┬────┘
            │            │            │
            │      ┌─────▼─────┐     │
            │      │   Plan    │     │
            │      └─────┬─────┘     │
            │            │           │
            │      ┌─────▼─────┐    │
            │      │  Execute  │◄───┘
            │      │  (loop)   │
            │      └─────┬─────┘
            │            │
            │      ┌─────▼─────┐
            │      │  Confirm? │ (human-in-the-loop)
            │      └─────┬─────┘
            │            │
            ▼            ▼
       ┌─────────────────────┐
       │     Synthesize      │
       └──────────┬──────────┘
                  │
             ┌────▼────┐
             │   END   │
             └─────────┘
```

### 3.2 Graph State

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]     # Conversation history
    query: str                                   # Current user query
    query_type: str                              # "simple" | "multi_step" | "action"
    plan: list[str]                              # Steps the agent plans to take
    retrieved_chunks: list[dict]                 # RAG results with scores
    tool_results: list[dict]                     # Results from MCP tool calls
    pending_confirmation: dict | None            # Action awaiting user approval
    final_answer: str                            # Synthesized response
    reasoning_trace: list[dict]                  # Step-by-step execution log
```

### 3.3 Node Descriptions

| Node        | Description                                                                                      |
|-------------|--------------------------------------------------------------------------------------------------|
| **Classify**    | Use LLM to categorize the query. Output: `query_type` field updated.                        |
| **Retrieve**    | Call Retrieval Service (gRPC) with the query. Store results in `retrieved_chunks`.           |
| **Plan**        | For multi-step queries, use LLM to decompose into sub-tasks. Store in `plan`.               |
| **Execute**     | Loop: for each step in `plan`, call Retrieve or MCP tools. Accumulate results.              |
| **Confirm**     | If an action is destructive or ambiguous, set `pending_confirmation` and pause the graph.    |
| **Synthesize**  | Generate final answer from all gathered context, with source citations.                      |

### 3.4 Conditional Routing

```python
def route_by_query_type(state: AgentState) -> str:
    match state["query_type"]:
        case "simple":
            return "retrieve"        # → Retrieve → Synthesize
        case "multi_step":
            return "plan"            # → Plan → Execute (loop) → Synthesize
        case "action":
            return "execute"         # → Execute → Confirm → Synthesize
```

---

## 4. MCP Integration

The agent connects to the MCP Server as a client to invoke tools.

### 4.1 Connection

- Transport: **stdio** (agent spawns MCP server as a subprocess)
- Alternative: **SSE** transport if running as separate container
- The agent discovers available tools at startup via `tools/list`

### 4.2 Tool Invocation Flow

```
Agent decides to use a tool
    → Calls MCP client.call_tool(name, arguments)
    → MCP Server executes the tool
    → Returns structured result (or error)
    → Agent incorporates result into state
```

### 4.3 Available Tools

| Tool                  | MCP Name              | Description                              |
|-----------------------|-----------------------|------------------------------------------|
| Search Jira           | `jira_search`         | Search tickets by JQL query              |
| Create Jira Ticket    | `jira_create`         | Create a new Jira ticket                 |
| GitHub PR Lookup      | `github_get_pr`       | Get PR details by repo + number          |
| GitHub File Contents  | `github_get_file`     | Read a file from a GitHub repository     |
| PagerDuty Incidents   | `pagerduty_incidents` | List current open incidents              |
| PagerDuty Acknowledge | `pagerduty_ack`       | Acknowledge an incident                  |
| Knowledge Search      | `knowledge_search`    | Search the internal knowledge base (RAG) |

---

## 5. Human-in-the-Loop

### 5.1 When to Pause

The agent pauses and requests confirmation when:
- Creating or modifying external resources (Jira tickets, PagerDuty actions)
- The query is ambiguous and the agent is unsure of the intended action
- A destructive action is detected (e.g., escalating an incident)

### 5.2 Mechanism

1. Agent sets `pending_confirmation` in state with action details
2. Graph execution pauses (LangGraph `interrupt_before` on the confirm node)
3. Gateway receives the pause event, sends `confirm_required` SSE to frontend
4. User approves or rejects via `POST /api/chat/{id}/confirm`
5. Gateway calls Agent's `ResumeChat` gRPC method with the confirmation
6. Graph execution resumes from the confirm node

---

## 6. Memory

### 6.1 Short-Term (Session)

- Full conversation history stored in `AgentState.messages`
- Passed through the graph on every turn
- Automatically managed by LangGraph's message accumulator

### 6.2 Long-Term (Cross-Session) — Optional

- After each session, extract key facts (e.g., "Auth service runs on port 8443")
- Store in a dedicated Qdrant collection `long_term_memory`
- Retrieve relevant memories at the start of each session using semantic search

---

## 7. gRPC Interface

### 7.1 Proto Definition

```protobuf
syntax = "proto3";
package agent;

service AgentService {
  rpc Chat(ChatRequest) returns (stream ChatResponse);
  rpc ResumeChat(ResumeRequest) returns (stream ChatResponse);
}

message ChatRequest {
  string session_id = 1;
  string message = 2;
  string team = 3;
}

message ResumeRequest {
  string session_id = 1;
  string action_id = 2;
  bool approved = 3;
}

message ChatResponse {
  string event_type = 1;    // "token", "source", "reasoning", "confirm_required", "done"
  string data = 2;          // JSON payload
}
```

---

## 8. Directory Structure

```
services/agent/
├── Dockerfile
├── requirements.txt
└── src/
    ├── main.py               # gRPC server entry point
    ├── config.py              # Configuration
    ├── service.py             # AgentService gRPC implementation
    ├── state.py               # AgentState TypedDict definition
    ├── graph/
    │   ├── builder.py         # Build and compile the LangGraph graph
    │   └── router.py          # Conditional routing logic
    ├── nodes/
    │   ├── classify.py        # Query classification node
    │   ├── retrieve.py        # RAG retrieval node
    │   ├── plan.py            # Multi-step planning node
    │   ├── execute.py         # Tool execution loop node
    │   ├── confirm.py         # Human-in-the-loop node
    │   └── synthesize.py      # Final answer generation node
    ├── mcp_client/
    │   └── client.py          # MCP client wrapper
    └── grpc_clients/
        └── retrieval.py       # Retrieval service gRPC client
```

---

## 9. Dependencies

```
langgraph>=0.2.0
langchain-anthropic>=0.3.0
langchain-core>=0.3.0
grpcio>=1.60.0
grpcio-tools>=1.60.0
mcp>=1.0.0
anthropic>=0.40.0
structlog>=24.0
pydantic>=2.0
pydantic-settings>=2.0
```

---

## 10. Implementation Steps

1. [ ] Define `AgentState` TypedDict and graph state schema
2. [ ] Implement `classify` node — LLM-based query classification
3. [ ] Implement `retrieve` node — call Retrieval Service via gRPC
4. [ ] Implement `plan` node — LLM decomposes query into sub-tasks
5. [ ] Implement `execute` node — loop over plan, call retrieve/tools
6. [ ] Implement `synthesize` node — generate cited answer from context
7. [ ] Build LangGraph graph with conditional routing
8. [ ] Integrate MCP client — connect to MCP Server, discover tools
9. [ ] Implement `confirm` node with `interrupt_before` for human-in-the-loop
10. [ ] Define `agent.proto` and implement gRPC service with streaming
11. [ ] Add session memory management
12. [ ] Add reasoning trace emission at each node
13. [ ] Write unit tests for individual nodes
14. [ ] Write integration test for full graph execution
15. [ ] Create Dockerfile

---

## 11. Key Design Decisions

| Decision                          | Choice                   | Rationale                                                 |
|-----------------------------------|--------------------------|-----------------------------------------------------------|
| Agent as gRPC service vs library  | gRPC service             | Separate scaling, clear boundary, independent deployment  |
| MCP transport                     | stdio (default)          | Simplest for single-container; switch to SSE for multi-container |
| LLM for classification           | Claude with structured output | Consistent with the rest of the agent's LLM usage    |
| Graph persistence                 | In-memory per session    | Simplicity; upgrade to Redis/Postgres for production      |
