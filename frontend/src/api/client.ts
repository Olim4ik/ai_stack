import axios from 'axios'
import type {
  DocumentListResponse,
  IngestResponse,
  HealthResponse,
  ChatSession,
} from '@/types'

const api = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
})

// ── Chat ───────────────────────────────────────────────────────────
export async function sendChatMessage(
  sessionId: string,
  message: string,
  team: string
): Promise<Response> {
  // Use fetch for SSE streaming (not axios)
  return fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message, team }),
  })
}

export async function confirmAction(
  sessionId: string,
  actionId: string,
  approved: boolean
): Promise<Response> {
  return fetch(`/api/chat/${sessionId}/confirm`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action_id: actionId, approved }),
  })
}

export async function getChatHistory(
  sessionId: string
): Promise<ChatSession[]> {
  const { data } = await api.get('/chat/history', {
    params: { session_id: sessionId },
  })
  return data
}

// ── Documents ──────────────────────────────────────────────────────
export async function ingestDocument(
  file: File,
  team: string,
  tags: string[]
): Promise<IngestResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('team', team)
  tags.forEach((t) => form.append('tags', t))
  const { data } = await api.post('/documents/ingest', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function listDocuments(
  params: { team?: string; page?: number; page_size?: number } = {}
): Promise<DocumentListResponse> {
  const { data } = await api.get('/documents', { params })
  return data
}

export async function deleteDocument(docId: string): Promise<void> {
  await api.delete(`/documents/${docId}`)
}

// ── Health ─────────────────────────────────────────────────────────
export async function getHealth(): Promise<HealthResponse> {
  const { data } = await axios.get('/health')
  return data
}
