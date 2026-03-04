# Service Plan — Infrastructure (Qdrant + Vue.js Frontend)

**Components**: Qdrant Vector Store, Vue.js Frontend (nginx)
**Type**: Infrastructure / Static Frontend
**Ports**: 6333 (Qdrant REST), 6334 (Qdrant gRPC), 8080 (Vue.js)
**Build Phase**: Phase 1 (Qdrant), Phase 5 (Vue.js)
**Parent Plan**: [implementation_plan.md](implementation_plan.md)

---

## Part A: Qdrant Vector Store

### 1. Purpose

Qdrant is the vector database that stores document chunk embeddings and supports semantic search. It is deployed as a pre-built Docker container with no custom code — only configuration.

### 2. Deployment

```yaml
# docker-compose.yml (excerpt)
services:
  qdrant:
    image: qdrant/qdrant:v1.12.0
    ports:
      - "6333:6333"     # REST API
      - "6334:6334"     # gRPC API
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      QDRANT__SERVICE__GRPC_PORT: 6334
    restart: unless-stopped

volumes:
  qdrant_data:
```

### 3. Collection Strategy

| Aspect              | Configuration                                    |
|---------------------|--------------------------------------------------|
| Collection naming   | `team_{team_name}` (e.g., `team_platform`)       |
| Vector size         | Matches embedding model (1536 for OpenAI)        |
| Distance metric     | Cosine                                           |
| Sparse vectors      | Enabled for hybrid search support                |
| On-disk storage     | Enabled — keeps memory usage manageable          |

### 4. Payload Indexes

Create payload indexes for commonly filtered fields to speed up filtered search:

```python
client.create_payload_index(
    collection_name="team_platform",
    field_name="tags",
    field_schema=PayloadSchemaType.KEYWORD,
)
client.create_payload_index(
    collection_name="team_platform",
    field_name="doc_id",
    field_schema=PayloadSchemaType.KEYWORD,
)
client.create_payload_index(
    collection_name="team_platform",
    field_name="ingested_at",
    field_schema=PayloadSchemaType.DATETIME,
)
```

### 5. Backup & Persistence

- Data persisted via Docker volume `qdrant_data`
- For production: use Qdrant's snapshot API for backups
- Snapshots can be triggered via `POST /collections/{name}/snapshots`

### 6. Implementation Steps

1. [ ] Add Qdrant service to `docker-compose.yml`
2. [ ] Verify Qdrant starts and is accessible on ports 6333/6334
3. [ ] Create initialization script that sets up default collections and payload indexes
4. [ ] Document collection naming convention and payload schema
5. [ ] Test data persistence across container restarts

---

## Part B: Vue.js Frontend

### 1. Purpose

A single-page application that provides the chat interface, document management, agent reasoning trace viewer, and human-in-the-loop confirmation dialogs.

### 2. Pages & Components

#### 2.1 Pages

| Page             | Route               | Description                                |
|------------------|----------------------|--------------------------------------------|
| Chat             | `/`                  | Main chat interface with the assistant     |
| Documents        | `/documents`         | Upload, list, and delete knowledge base docs|
| History          | `/history`           | Browse and resume past conversations        |

#### 2.2 Core Components

```
src/
├── App.vue
├── router/
│   └── index.ts
├── views/
│   ├── ChatView.vue            # Main chat page
│   ├── DocumentsView.vue       # Document management page
│   └── HistoryView.vue         # Session history page
├── components/
│   ├── chat/
│   │   ├── ChatInput.vue       # Message input with send button
│   │   ├── ChatMessage.vue     # Single message bubble (user or assistant)
│   │   ├── StreamingMessage.vue # Displays tokens as they stream in
│   │   ├── SourcePanel.vue     # Shows retrieved sources with scores
│   │   ├── ReasoningTrace.vue  # Collapsible agent step-by-step trace
│   │   └── ConfirmDialog.vue   # Human-in-the-loop approve/reject dialog
│   ├── documents/
│   │   ├── DocumentUpload.vue  # Drag-and-drop file upload
│   │   ├── DocumentList.vue    # Paginated document table
│   │   └── DocumentFilters.vue # Filter by team, tags, date
│   └── layout/
│       ├── Sidebar.vue         # Navigation + session history
│       └── Header.vue          # Top bar with team selector
├── composables/
│   ├── useChat.ts              # Chat logic, SSE connection, state
│   ├── useDocuments.ts         # Document CRUD operations
│   └── useEventSource.ts       # SSE stream handler
├── stores/
│   ├── chatStore.ts            # Pinia store for chat state
│   └── documentStore.ts        # Pinia store for documents
├── api/
│   └── client.ts               # Axios/fetch API client for FastAPI gateway
└── types/
    └── index.ts                # TypeScript interfaces
```

### 3. Key Interactions

#### 3.1 Chat Flow (SSE)

```
User types message
    → POST /api/chat { session_id, message, team }
    → Open SSE connection to response stream
    → Receive events:
        "token"            → Append to streaming message
        "source"           → Add to source panel
        "reasoning"        → Add step to reasoning trace
        "confirm_required" → Show confirmation dialog
        "done"             → Finalize message
```

#### 3.2 Human-in-the-Loop

```
Agent sends "confirm_required" event
    → ConfirmDialog.vue shows proposed action details
    → User clicks Approve or Reject
    → POST /api/chat/{id}/confirm { action_id, approved }
    → SSE stream resumes with result
```

#### 3.3 Document Upload

```
User drags file onto DocumentUpload.vue
    → POST /api/documents/ingest (multipart form)
    → Show progress indicator
    → On success, refresh document list
```

### 4. Tech Stack

| Concern          | Library              | Rationale                          |
|------------------|----------------------|------------------------------------|
| Framework        | Vue 3 (Composition API) | Specified in requirements       |
| Build tool       | Vite                 | Fast builds, Vue official tooling  |
| State management | Pinia                | Official Vue state management      |
| HTTP client      | Axios                | Mature, interceptors, good DX      |
| UI framework     | Tailwind CSS         | Utility-first, rapid prototyping   |
| Router           | Vue Router 4         | Standard Vue routing               |
| TypeScript       | Yes                  | Type safety across the frontend    |

### 5. Deployment (nginx)

```dockerfile
# frontend/Dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 8080
```

```nginx
# frontend/nginx.conf
server {
    listen 8080;
    root /usr/share/nginx/html;
    index index.html;

    # SPA fallback
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy API requests to FastAPI gateway
    location /api/ {
        proxy_pass http://gateway:8000;
        proxy_set_header Host $host;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
        proxy_buffering off;           # Required for SSE
    }
}
```

### 6. Docker Compose Entry

```yaml
# docker-compose.yml (excerpt)
services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    depends_on:
      - gateway
    restart: unless-stopped
```

### 7. Implementation Steps

1. [ ] Scaffold Vue 3 project with Vite, TypeScript, Pinia, Vue Router, Tailwind
2. [ ] Implement layout (Sidebar, Header) with navigation
3. [ ] Build `ChatInput.vue` and `ChatMessage.vue` components
4. [ ] Implement `useEventSource.ts` composable for SSE stream handling
5. [ ] Build `StreamingMessage.vue` — render tokens as they arrive
6. [ ] Build `SourcePanel.vue` — display retrieved sources with relevance scores
7. [ ] Build `ReasoningTrace.vue` — collapsible step-by-step agent trace
8. [ ] Build `ConfirmDialog.vue` — human-in-the-loop approve/reject
9. [ ] Implement `ChatView.vue` integrating all chat components
10. [ ] Build `DocumentUpload.vue` with drag-and-drop
11. [ ] Build `DocumentList.vue` with pagination and filters
12. [ ] Implement `DocumentsView.vue`
13. [ ] Build `HistoryView.vue` with session list
14. [ ] Create API client (`api/client.ts`) for all FastAPI endpoints
15. [ ] Set up Pinia stores for chat and document state
16. [ ] Create nginx config with API proxy and SSE support
17. [ ] Create Dockerfile (multi-stage: build + nginx)
18. [ ] Test all views and interactions end-to-end

---

## Part C: Docker Compose (Full Orchestration)

### 1. Complete docker-compose.yml

```yaml
version: "3.8"

services:
  # Infrastructure
  qdrant:
    image: qdrant/qdrant:v1.12.0
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped

  # Backend Services
  embedding:
    build: ./services/embedding
    ports:
      - "50052:50052"
    env_file: .env
    restart: unless-stopped

  retrieval:
    build: ./services/retrieval
    ports:
      - "50051:50051"
    env_file: .env
    depends_on:
      - qdrant
      - embedding
    restart: unless-stopped

  agent:
    build: ./services/agent
    ports:
      - "50053:50053"
    env_file: .env
    depends_on:
      - retrieval
    restart: unless-stopped

  mcp-server:
    build: ./services/mcp_server
    env_file: .env
    depends_on:
      - retrieval
    restart: unless-stopped

  gateway:
    build: ./services/gateway
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - embedding
      - retrieval
      - agent
    restart: unless-stopped

  # Frontend
  frontend:
    build: ./frontend
    ports:
      - "8080:8080"
    depends_on:
      - gateway
    restart: unless-stopped

volumes:
  qdrant_data:
```

### 2. Startup Order

```
qdrant → embedding → retrieval → mcp-server → agent → gateway → frontend
```

### 3. Implementation Steps

1. [ ] Create `docker-compose.yml` with all services
2. [ ] Create `.env.example` with all required environment variables
3. [ ] Verify `docker compose up` starts all services
4. [ ] Verify inter-service gRPC connectivity
5. [ ] Verify frontend can reach gateway through nginx proxy
6. [ ] Test full end-to-end flow: upload doc → chat → get cited answer
7. [ ] Add healthcheck directives to each service in compose
