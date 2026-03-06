# Internal Knowledge Base / DevOps Assistant

An AI-powered internal knowledge base assistant that helps engineering teams find answers from internal documentation, runbooks, and project artifacts. The system uses RAG (Retrieval-Augmented Generation) with an agentic workflow to retrieve context, reason through multi-step queries, and execute actions via external tool integrations.

## Architecture

```
 ┌──────────┐    REST/SSE    ┌───────────────┐
 │  Vue.js  │ ─────────────► │   FastAPI     │
 │  :8080   │                │   Gateway     │
 └──────────┘                │   :8000       │
                             └───────┬───────┘
                                     │
                        ┌────────────┼────────────┐
                        │            │            │
                        ▼            │            ▼
              ┌──────────────┐       │   ┌──────────────┐
              │  LangGraph   │       │   │   Document   │
              │    Agent     │       │   │   Ingestion  │
              │   :50053     │       │   │  (in gateway)│
              └──────┬───┬──┘       │   └──────┬───────┘
                     │   │          │          │
             ┌───────┘   └──────┐   │          │
             ▼                  ▼   │          │
    ┌──────────────┐  ┌──────────┐ │          │
    │  MCP Server  │  │ Retrieval│ │          │
    │   (stdio)    │  │ Service  │◄┘          │
    └──────┬───────┘  │  :50051  │            │
           │          └────┬─────┘            │
       External            │                  │
        APIs               ▼                  ▼
    (Jira, GitHub,   ┌──────────┐      ┌──────────┐
     PagerDuty)      │  Qdrant  │      │ Embedding│
                     │  :6333   │      │ Service  │
                     └──────────┘      │  :50052  │
                                       └──────────┘
```

## Services

### Qdrant Vector Store
Vector database that stores document chunk embeddings and supports semantic search. Deployed as a pre-built Docker container.
- **Ports**: 6333 (REST), 6334 (gRPC)
- **Multi-tenancy**: One collection per team (`team_{name}`)
- **Storage**: Persistent Docker volume

### Embedding Service
A gRPC microservice that accepts text input and returns vector embeddings. Used by both the ingestion pipeline and the Retrieval Service.
- **Port**: 50052
- **Providers**: OpenAI `text-embedding-3-small` (default) or Sentence Transformers (local)
- **Supports**: Single and batch embedding (up to 64 texts per request)

### Retrieval Service
Handles all vector search operations against Qdrant. Accepts a query, embeds it via the Embedding Service, performs semantic search, and returns ranked document chunks.
- **Port**: 50051
- **Search modes**: Dense (default) and Hybrid (dense + BM25 sparse with RRF fusion)
- **Filtering**: By team, document type, tags, and date range

### LangGraph Agent
The core intelligence layer. Classifies queries, orchestrates multi-step reasoning using a stateful graph, invokes tools via MCP, and produces cited answers.
- **Port**: 50053
- **Query types**: Simple lookup, multi-step reasoning, action requests
- **LLM**: Claude (Anthropic API)
- **Features**: Conditional routing, session memory, human-in-the-loop confirmation, reasoning trace

### MCP Server
Exposes external tool integrations as structured tools that the LangGraph Agent can discover and invoke, following the Model Context Protocol specification.
- **Transport**: stdio (default) or SSE
- **Tools**: `jira_search`, `jira_create`, `github_get_pr`, `github_get_file`, `pagerduty_incidents`, `pagerduty_ack`, `knowledge_search`

### FastAPI Gateway
Single entry point for all client requests. Exposes REST endpoints, proxies chat to the Agent via gRPC, handles document ingestion, and streams responses via SSE.
- **Port**: 8000
- **Endpoints**:
  - `POST /api/chat` — Chat with the assistant (SSE streaming)
  - `POST /api/documents/ingest` — Upload and ingest documents
  - `GET /api/documents` — List ingested documents
  - `DELETE /api/documents/{id}` — Remove a document
  - `GET /api/chat/history` — Conversation history
  - `POST /api/chat/{id}/confirm` — Human-in-the-loop confirmation

### Vue.js Frontend
Single-page application with chat interface, document management, agent reasoning trace viewer, and human-in-the-loop confirmation dialogs.
- **Port**: 8080 (served via nginx)
- **Stack**: Vue 3, Vite, Pinia, Vue Router, Tailwind CSS, TypeScript

## Tech Stack

| Layer              | Technology                     |
|--------------------|--------------------------------|
| Orchestration      | LangGraph                      |
| Vector Store       | Qdrant                         |
| Tool Protocol      | MCP (Model Context Protocol)   |
| API Gateway        | FastAPI (REST / SSE)           |
| Internal Services  | gRPC + Protocol Buffers        |
| Frontend           | Vue.js 3 + Vite + Tailwind    |
| Containerization   | Docker / Docker Compose        |
| LLM                | Claude (Anthropic API)         |
| Embeddings         | OpenAI / Sentence Transformers |
| Language           | Python 3.12 (backend)         |

## Getting Started

### Prerequisites

- Docker and Docker Compose
- An OpenAI API key (for embeddings)
- An Anthropic API key (for the LLM agent)
- Optional: Jira, GitHub, and PagerDuty API tokens (for MCP tool integrations)

### 1. Clone the repository

```bash
git clone <repository-url>
cd ai_complete
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys:

```env
# Required
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional (for MCP tool integrations)
JIRA_BASE_URL=https://yourcompany.atlassian.net
JIRA_EMAIL=you@company.com
JIRA_API_TOKEN=...
GITHUB_TOKEN=ghp_...
PAGERDUTY_API_KEY=...
```

### 3. Start all services

```bash
docker compose up --build
```

This starts all services in dependency order:

```
qdrant → embedding → retrieval → mcp-server → agent → gateway → frontend
```

### 4. Access the application

| Interface          | URL                     |
|--------------------|-------------------------|
| Frontend (UI)      | http://localhost:8080   |
| FastAPI Gateway    | http://localhost:8000   |
| API Docs (Swagger) | http://localhost:8000/docs |
| Qdrant Dashboard   | http://localhost:6333/dashboard |

### 5. Upload documents and chat

1. Open http://localhost:8080
2. Navigate to the **Documents** page and upload Markdown, PDF, HTML, or plain text files
3. Go to the **Chat** page and ask questions about your uploaded documents
4. The assistant will retrieve relevant context, provide cited answers, and can execute actions via Jira/GitHub/PagerDuty when needed

## Running Individual Services

For development, you can run services individually:

```bash
# Start only infrastructure
docker compose up qdrant

# Start backend services
docker compose up qdrant embedding retrieval

# Start the full backend (no frontend)
docker compose up qdrant embedding retrieval agent mcp-server gateway
```

## Project Structure

```
ai_complete/
├── docker-compose.yml
├── .env.example
├── proto/                          # Shared protobuf definitions
│   ├── embedding.proto
│   ├── retrieval.proto
│   └── agent.proto
├── services/
│   ├── gateway/                    # FastAPI Gateway
│   ├── agent/                      # LangGraph Agent
│   ├── embedding/                  # Embedding gRPC Service
│   ├── retrieval/                  # Retrieval gRPC Service
│   └── mcp_server/                 # MCP Tool Server
├── frontend/                       # Vue.js Client
├── tests/
│   ├── integration/
│   └── load/
└── docs/
    └── knowledge_base/             # Requirements and service plans
```

## Documentation

Detailed service plans and requirements are available in [docs/knowledge_base/](docs/knowledge_base/):

- [Functional Requirements](docs/knowledge_base/functional_requirements.md)
- [Implementation Plan](docs/knowledge_base/implementation_plan.md)
- [FastAPI Gateway Plan](docs/knowledge_base/plan_fastapi_gateway.md)
- [LangGraph Agent Plan](docs/knowledge_base/plan_langgraph_agent.md)
- [Embedding Service Plan](docs/knowledge_base/plan_embedding_service.md)
- [Retrieval Service Plan](docs/knowledge_base/plan_retrieval_service.md)
- [MCP Server Plan](docs/knowledge_base/plan_mcp_server.md)
- [Infrastructure Plan](docs/knowledge_base/plan_infrastructure.md)
