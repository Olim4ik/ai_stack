# Backlog — Vue.js Frontend

**Phase**: 4
**Service**: Vue.js Frontend (nginx, port 8080)
**Plan**: [plan_infrastructure.md](../plan_infrastructure.md) — Part B

---

## Tasks

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Scaffold Vue 3 project with Vite, TypeScript, Pinia, Vue Router, Tailwind | DONE | `frontend/` — package.json, vite.config.ts, tailwind.config.js, tsconfig |
| 2 | Implement layout components (Sidebar, Header) with navigation | DONE | `components/layout/Sidebar.vue`, `Header.vue` — team selector, new chat, recent sessions |
| 3 | Set up Vue Router with Chat, Documents, History routes | DONE | `router/index.ts` — 3 routes |
| 4 | Build `ChatInput.vue` — message input with send button | DONE | `components/chat/ChatInput.vue` — Enter to send, Shift+Enter newline |
| 5 | Build `ChatMessage.vue` — single message bubble (user/assistant) | DONE | `components/chat/ChatMessage.vue` — avatar, role label, content, sources, reasoning |
| 6 | Implement `useEventSource.ts` composable for SSE stream handling | DONE | `composables/useEventSource.ts` — ReadableStream reader, event dispatching |
| 7 | Build `StreamingMessage.vue` — render tokens as they arrive | DONE | `components/chat/StreamingMessage.vue` — cursor animation, loading dots |
| 8 | Build `SourcePanel.vue` — display retrieved sources with scores | DONE | `components/chat/SourcePanel.vue` — collapsible, relevance %, section metadata |
| 9 | Build `ReasoningTrace.vue` — collapsible agent step trace | DONE | `components/chat/ReasoningTrace.vue` — per-node icons, step timeline |
| 10 | Build `ConfirmDialog.vue` — human-in-the-loop approve/reject | DONE | `components/chat/ConfirmDialog.vue` — tool name, description, approve/reject buttons |
| 11 | Implement `ChatView.vue` integrating all chat components | DONE | `views/ChatView.vue` — message list, streaming, confirm, auto-scroll |
| 12 | Create API client (`api/client.ts`) for all FastAPI endpoints | DONE | `api/client.ts` — fetch for SSE, axios for REST |
| 13 | Set up Pinia stores for chat and document state | DONE | `stores/chatStore.ts`, `documentStore.ts` |
| 14 | Build `DocumentUpload.vue` with drag-and-drop | DONE | `components/documents/DocumentUpload.vue` — drag/drop, file picker, team/tags |
| 15 | Build `DocumentList.vue` with pagination and filters | DONE | `components/documents/DocumentList.vue` — table, pagination, delete |
| 16 | Implement `DocumentsView.vue` | DONE | `views/DocumentsView.vue` — upload + filters + list |
| 17 | Build `HistoryView.vue` with session list | DONE | `views/HistoryView.vue` — session cards, resume navigation |
| 18 | Define TypeScript interfaces | DONE | `types/index.ts` — ChatMessage, Source, Document, SSEEvent, etc. |

---

## Progress Log

| Date | Update |
|------|--------|
| 2026-03-04 | All 18 tasks complete. Full Vue.js SPA with Vite + TypeScript + Tailwind + Pinia. Chat with SSE streaming, source panel, reasoning trace, human-in-the-loop confirmation dialog. Document management with drag-and-drop upload, pagination, filters. Session history with resume. API client using fetch (SSE) + axios (REST). |
