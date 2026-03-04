# Internal Knowledge Base / DevOps Assistant

## Functional Requirements Document

### 1. Overview

An AI-powered internal knowledge base assistant that helps engineering teams find answers from internal documentation, runbooks, and project artifacts. The system uses RAG (Retrieval-Augmented Generation) with an agentic workflow to retrieve context, reason through multi-step queries, and execute actions via external tool integrations.

### 2. Tech Stack

| Layer              | Technology         |
|--------------------|--------------------|
| Orchestration      | LangGraph          |
| Vector Store       | Qdrant             |
| Tool Protocol      | MCP (Model Context Protocol) |
| API Gateway        | FastAPI (REST)     |
| Internal Services  | gRPC               |
| Frontend           | Vue.js             |
| Containerization   | Docker / Docker Compose |
| LLM               | Claude (Anthropic API) |
| Embeddings         | OpenAI or Sentence Transformers |

---

### 3. Core Functional Requirements

#### 3.1 Document Ingestion

- **FR-1.1**: The system shall accept documents in Markdown, plain text, PDF, and HTML formats.
- **FR-1.2**: The system shall chunk documents using configurable strategies (fixed-size with overlap, semantic chunking).
- **FR-1.3**: The system shall generate vector embeddings for each chunk and store them in Qdrant.
- **FR-1.4**: Each chunk shall retain metadata: source file, document title, section heading, ingestion timestamp, and document type.
- **FR-1.5**: The system shall support re-ingestion of updated documents, replacing stale chunks while preserving document identity.
- **FR-1.6**: The system shall support multi-tenant collections in Qdrant, isolating data by team or project.

#### 3.2 Retrieval & Search

- **FR-2.1**: The system shall perform semantic search over ingested documents using dense vector similarity.
- **FR-2.2**: The system shall support hybrid search combining dense vectors with keyword/sparse vector matching.
- **FR-2.3**: The system shall support metadata filtering (by document type, team, date range, tags).
- **FR-2.4**: The system shall return ranked results with relevance scores and source attribution.
- **FR-2.5**: The system shall expose retrieval as a gRPC microservice consumed by the agent.

#### 3.3 Agent Workflow (LangGraph)

- **FR-3.1**: The agent shall classify incoming queries into categories: simple lookup, multi-step reasoning, or action request.
- **FR-3.2**: **Simple lookup**: Query goes directly to RAG retrieval and generates an answer with citations.
- **FR-3.3**: **Multi-step reasoning**: Agent plans a sequence of retrieval and tool calls, executes them iteratively, and synthesizes a final answer.
- **FR-3.4**: **Action request**: Agent identifies the required action (e.g., create a ticket) and routes to the appropriate MCP tool.
- **FR-3.5**: The agent shall support conditional routing — choosing different graph paths based on query classification and intermediate results.
- **FR-3.6**: The agent shall maintain conversational memory within a session (short-term) and optionally persist key facts across sessions (long-term).
- **FR-3.7**: The agent shall support human-in-the-loop — pausing execution to request user confirmation before taking destructive or ambiguous actions.
- **FR-3.8**: The agent shall expose its reasoning trace (which nodes were visited, what tools were called, what was retrieved) for transparency.

#### 3.4 MCP Tool Server

- **FR-4.1**: The system shall run an MCP server exposing tools that the LangGraph agent can invoke.
- **FR-4.2**: **Jira Tool**: Create, update, and search Jira tickets.
- **FR-4.3**: **GitHub Tool**: Look up PRs, commits, and file contents from GitHub repositories.
- **FR-4.4**: **PagerDuty Tool**: Check current incidents, acknowledge, and escalate.
- **FR-4.5**: **Document Search Tool**: Expose the RAG retrieval pipeline as an MCP tool so the agent can call it as a step in its workflow.
- **FR-4.6**: Each tool shall define its schema (name, description, input parameters, output format) following the MCP specification.
- **FR-4.7**: Tool execution errors shall be returned to the agent as structured error responses, not exceptions.

#### 3.5 API Layer

##### 3.5.1 REST API (FastAPI)

- **FR-5.1**: `POST /api/chat` — Accept a user message and return the agent's response (supports streaming via SSE).
- **FR-5.2**: `POST /api/documents/ingest` — Upload and ingest documents into the knowledge base.
- **FR-5.3**: `GET /api/documents` — List ingested documents with pagination and filtering.
- **FR-5.4**: `DELETE /api/documents/{id}` — Remove a document and its associated chunks from the vector store.
- **FR-5.5**: `GET /api/chat/history` — Retrieve conversation history for a session.
- **FR-5.6**: `POST /api/chat/{id}/confirm` — Submit human-in-the-loop confirmation for a pending agent action.

##### 3.5.2 gRPC Internal Services

- **FR-5.7**: **Embedding Service** — Accepts text, returns vector embeddings. Used by both ingestion and retrieval.
- **FR-5.8**: **Retrieval Service** — Accepts a query vector and filters, returns ranked document chunks from Qdrant.
- **FR-5.9**: Services shall use Protocol Buffers for message definitions and support health checking.

#### 3.6 Frontend (Vue.js)

- **FR-6.1**: Chat interface for conversing with the assistant, with streaming response display.
- **FR-6.2**: Source panel showing retrieved documents and relevance scores for each response.
- **FR-6.3**: Agent reasoning trace viewer — collapsible view of the agent's step-by-step execution.
- **FR-6.4**: Document management page — upload, list, and delete knowledge base documents.
- **FR-6.5**: Human-in-the-loop confirmation dialog — when the agent requests approval, display the proposed action and allow approve/reject.
- **FR-6.6**: Session history sidebar — browse and resume past conversations.

---

### 4. Non-Functional Requirements

- **NFR-1**: All services shall be containerized with Docker and orchestrated via Docker Compose.
- **NFR-2**: The system shall use structured logging (JSON format) across all services.
- **NFR-3**: API responses for chat queries shall stream the first token within 2 seconds under normal load.
- **NFR-4**: The system shall include RAG evaluation metrics: retrieval precision, answer faithfulness, and answer relevance.
- **NFR-5**: The system shall handle concurrent users (target: 10 simultaneous sessions under load test).
- **NFR-6**: Environment configuration shall be managed via environment variables (12-factor app).
- **NFR-7**: Sensitive credentials (API keys, tokens) shall never be logged or exposed in API responses.

---

### 5. Architecture Overview

```
                         +------------------+
                         |   Vue.js Client  |
                         +--------+---------+
                                  |
                            REST / SSE
                                  |
                         +--------+---------+
                         |  FastAPI Gateway  |
                         +--------+---------+
                                  |
                    +-------------+-------------+
                    |                           |
              +-----+------+          +--------+--------+
              |  LangGraph |          | Document Ingest |
              |   Agent    |          |    Pipeline     |
              +-----+------+          +--------+--------+
                    |                           |
          +---------+---------+                 |
          |         |         |                 |
     +----+---+ +---+----+ +--+---+    +--------+--------+
     |  MCP   | | gRPC   | | gRPC |    |  gRPC Embedding |
     | Server | | Retriev | | Embed|    |     Service     |
     +----+---+ +---+----+ +--+---+    +--------+--------+
          |         |         |                  |
     External   +---+---------+------------------+
     APIs       |      Qdrant Vector Store       |
     (Jira,     +--------------------------------+
      GitHub,
      PagerDuty)
```

---

### 6. Build Phases

| Phase   | Scope                                                        |
|---------|--------------------------------------------------------------|
| Phase 1 | FastAPI skeleton + Qdrant ingestion pipeline (chunk, embed, store) |
| Phase 2 | gRPC embedding & retrieval microservices + RAG query endpoint |
| Phase 3 | LangGraph agent with tool calling + conditional routing      |
| Phase 4 | MCP server with Jira/GitHub/PagerDuty tools + human-in-the-loop |
| Phase 5 | Vue.js frontend + Docker Compose orchestration               |
| Phase 6 | Evaluation metrics, structured logging, load testing         |

---

### 7. Success Criteria

1. A user can upload internal docs and get accurate, cited answers from the assistant.
2. The agent correctly routes simple vs. multi-step vs. action queries.
3. MCP tools successfully interact with external services (Jira, GitHub, PagerDuty).
4. Human-in-the-loop pauses execution and resumes after user confirmation.
5. The full stack runs via `docker compose up` with no manual setup beyond environment variables.
6. RAG evaluation shows measurable retrieval precision and answer quality scores.
