import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { ChatMessage, ChatSession, Source, ReasoningStep, ConfirmAction } from '@/types'

export const useChatStore = defineStore('chat', () => {
  const messages = ref<ChatMessage[]>([])
  const sessions = ref<ChatSession[]>([])
  const currentSessionId = ref<string>(crypto.randomUUID())
  const currentTeam = ref<string>('platform')
  const isStreaming = ref(false)
  const streamingContent = ref('')
  const streamingSources = ref<Source[]>([])
  const streamingReasoning = ref<ReasoningStep[]>([])
  const pendingConfirm = ref<ConfirmAction | null>(null)

  const hasMessages = computed(() => messages.value.length > 0)

  function addUserMessage(content: string) {
    messages.value.push({
      id: crypto.randomUUID(),
      role: 'user',
      content,
      timestamp: Date.now(),
    })
  }

  function startStreaming() {
    isStreaming.value = true
    streamingContent.value = ''
    streamingSources.value = []
    streamingReasoning.value = []
    pendingConfirm.value = null
  }

  function appendToken(token: string) {
    streamingContent.value += token
  }

  function addSource(source: Source) {
    streamingSources.value.push(source)
  }

  function addReasoningStep(step: ReasoningStep) {
    streamingReasoning.value.push({ ...step, timestamp: Date.now() })
  }

  function setConfirmRequired(action: ConfirmAction) {
    pendingConfirm.value = action
  }

  function finalizeMessage() {
    if (streamingContent.value) {
      messages.value.push({
        id: crypto.randomUUID(),
        role: 'assistant',
        content: streamingContent.value,
        sources: [...streamingSources.value],
        reasoning: [...streamingReasoning.value],
        timestamp: Date.now(),
      })
    }
    isStreaming.value = false
    streamingContent.value = ''
    streamingSources.value = []
    streamingReasoning.value = []
    pendingConfirm.value = null
  }

  function newSession() {
    if (messages.value.length > 0) {
      sessions.value.unshift({
        session_id: currentSessionId.value,
        title: messages.value[0].content.slice(0, 60),
        team: currentTeam.value,
        message_count: messages.value.length,
        last_message_at: Date.now(),
        preview: messages.value[messages.value.length - 1].content.slice(0, 100),
      })
    }
    currentSessionId.value = crypto.randomUUID()
    messages.value = []
    streamingContent.value = ''
    isStreaming.value = false
  }

  function loadSession(session: ChatSession) {
    currentSessionId.value = session.session_id
    currentTeam.value = session.team
  }

  return {
    messages,
    sessions,
    currentSessionId,
    currentTeam,
    isStreaming,
    streamingContent,
    streamingSources,
    streamingReasoning,
    pendingConfirm,
    hasMessages,
    addUserMessage,
    startStreaming,
    appendToken,
    addSource,
    addReasoningStep,
    setConfirmRequired,
    finalizeMessage,
    newSession,
    loadSession,
  }
})
