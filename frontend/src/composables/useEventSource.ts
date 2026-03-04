import { ref } from 'vue'
import type { SSEEventType } from '@/types'

export interface SSECallbacks {
  onToken: (content: string) => void
  onSource: (data: any) => void
  onReasoning: (data: any) => void
  onConfirmRequired: (data: any) => void
  onDone: (data: any) => void
  onError: (error: string) => void
}

export function useEventSource() {
  const connected = ref(false)

  async function streamResponse(response: Response, callbacks: SSECallbacks) {
    if (!response.ok) {
      callbacks.onError(`HTTP ${response.status}: ${response.statusText}`)
      return
    }

    const reader = response.body?.getReader()
    if (!reader) {
      callbacks.onError('No response body')
      return
    }

    connected.value = true
    const decoder = new TextDecoder()
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        let currentEvent: SSEEventType = 'token'

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim() as SSEEventType
          } else if (line.startsWith('data: ')) {
            const rawData = line.slice(6)
            try {
              const data = JSON.parse(rawData)
              dispatchEvent(currentEvent, data, callbacks)
            } catch {
              // Plain text data (token content)
              dispatchEvent(currentEvent, rawData, callbacks)
            }
          }
        }
      }
    } finally {
      connected.value = false
      reader.releaseLock()
    }
  }

  function dispatchEvent(event: SSEEventType, data: any, callbacks: SSECallbacks) {
    switch (event) {
      case 'token':
        callbacks.onToken(typeof data === 'string' ? data : data.content || '')
        break
      case 'source':
        callbacks.onSource(data)
        break
      case 'reasoning':
        callbacks.onReasoning(data)
        break
      case 'confirm_required':
        callbacks.onConfirmRequired(data)
        break
      case 'done':
        callbacks.onDone(data)
        break
      case 'error':
        callbacks.onError(typeof data === 'string' ? data : data.message || 'Unknown error')
        break
    }
  }

  return { connected, streamResponse }
}
