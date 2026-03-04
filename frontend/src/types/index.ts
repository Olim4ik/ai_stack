export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  reasoning?: ReasoningStep[]
  confirmAction?: ConfirmAction
  timestamp: number
}

export interface Source {
  chunk_id: string
  title: string
  score: number
  metadata: Record<string, string>
}

export interface ReasoningStep {
  node: string
  action: string
  timestamp?: number
}

export interface ConfirmAction {
  action_id: string
  tool: string
  description: string
}

export interface ChatSession {
  session_id: string
  title: string
  team: string
  message_count: number
  last_message_at: number
  preview: string
}

export interface Document {
  doc_id: string
  title: string
  source_file: string
  team: string
  tags: string[]
  chunk_count: number
  ingested_at: string
}

export interface DocumentListResponse {
  documents: Document[]
  total: number
  page: number
  page_size: number
}

export interface IngestResponse {
  doc_id: string
  title: string
  chunks_created: number
  message: string
}

export interface HealthResponse {
  status: string
  services: Record<string, string>
}

export type SSEEventType = 'token' | 'source' | 'reasoning' | 'confirm_required' | 'done' | 'error'

export interface SSEEvent {
  event: SSEEventType
  data: string
}
