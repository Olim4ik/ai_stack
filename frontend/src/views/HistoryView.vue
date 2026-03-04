<script setup lang="ts">
import { useRouter } from 'vue-router'
import { useChatStore } from '@/stores/chatStore'

const chatStore = useChatStore()
const router = useRouter()

function resumeSession(sessionId: string) {
  const session = chatStore.sessions.find((s) => s.session_id === sessionId)
  if (session) {
    chatStore.loadSession(session)
    router.push('/')
  }
}

function formatTime(ts: number): string {
  return new Date(ts).toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}
</script>

<template>
  <div class="mx-auto max-w-3xl p-6">
    <h2 class="text-xl font-semibold text-gray-900">Conversation History</h2>

    <!-- Empty -->
    <div v-if="chatStore.sessions.length === 0" class="mt-12 text-center">
      <svg class="mx-auto h-12 w-12 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1">
        <path stroke-linecap="round" stroke-linejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      <p class="mt-3 text-sm text-gray-500">No conversation history yet</p>
    </div>

    <!-- Session list -->
    <div v-else class="mt-4 space-y-2">
      <button
        v-for="session in chatStore.sessions"
        :key="session.session_id"
        @click="resumeSession(session.session_id)"
        class="flex w-full items-start gap-4 rounded-xl border border-gray-200 bg-white p-4 text-left hover:border-blue-300 hover:shadow-sm transition-all"
      >
        <div class="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-blue-50 text-blue-600">
          <svg class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
            <path stroke-linecap="round" stroke-linejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
        </div>
        <div class="min-w-0 flex-1">
          <p class="text-sm font-medium text-gray-900 truncate">{{ session.title }}</p>
          <p class="mt-0.5 text-xs text-gray-400">
            {{ session.team }} &middot; {{ session.message_count }} messages &middot; {{ formatTime(session.last_message_at) }}
          </p>
          <p class="mt-1 text-sm text-gray-500 truncate">{{ session.preview }}</p>
        </div>
      </button>
    </div>
  </div>
</template>
