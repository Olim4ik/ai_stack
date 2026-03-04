import { useChatStore } from '@/stores/chatStore'
import { useEventSource } from './useEventSource'
import { sendChatMessage, confirmAction } from '@/api/client'

export function useChat() {
  const store = useChatStore()
  const { streamResponse } = useEventSource()

  async function send(message: string) {
    if (!message.trim() || store.isStreaming) return

    store.addUserMessage(message)
    store.startStreaming()

    try {
      const response = await sendChatMessage(
        store.currentSessionId,
        message,
        store.currentTeam
      )

      await streamResponse(response, {
        onToken: (content) => store.appendToken(content),
        onSource: (data) => store.addSource(data),
        onReasoning: (data) => store.addReasoningStep(data),
        onConfirmRequired: (data) => store.setConfirmRequired(data),
        onDone: () => store.finalizeMessage(),
        onError: (error) => {
          store.appendToken(`\n\n[Error: ${error}]`)
          store.finalizeMessage()
        },
      })
    } catch (e: any) {
      store.appendToken(`\n\n[Connection error: ${e.message}]`)
      store.finalizeMessage()
    }
  }

  async function confirm(actionId: string, approved: boolean) {
    try {
      const response = await confirmAction(
        store.currentSessionId,
        actionId,
        approved
      )
      store.pendingConfirm = null
      store.startStreaming()

      await streamResponse(response, {
        onToken: (content) => store.appendToken(content),
        onSource: (data) => store.addSource(data),
        onReasoning: (data) => store.addReasoningStep(data),
        onConfirmRequired: (data) => store.setConfirmRequired(data),
        onDone: () => store.finalizeMessage(),
        onError: (error) => {
          store.appendToken(`\n\n[Error: ${error}]`)
          store.finalizeMessage()
        },
      })
    } catch (e: any) {
      store.appendToken(`\n\n[Confirmation error: ${e.message}]`)
      store.finalizeMessage()
    }
  }

  return { send, confirm }
}
