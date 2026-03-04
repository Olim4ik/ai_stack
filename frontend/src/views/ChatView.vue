<script setup lang="ts">
import { ref, nextTick, watch } from 'vue'
import { useChatStore } from '@/stores/chatStore'
import { useChat } from '@/composables/useChat'
import ChatMessage from '@/components/chat/ChatMessage.vue'
import ChatInput from '@/components/chat/ChatInput.vue'
import StreamingMessage from '@/components/chat/StreamingMessage.vue'
import ConfirmDialog from '@/components/chat/ConfirmDialog.vue'

const chatStore = useChatStore()
const { send, confirm } = useChat()
const messagesContainer = ref<HTMLElement>()

function scrollToBottom() {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

watch(() => chatStore.messages.length, scrollToBottom)
watch(() => chatStore.streamingContent, scrollToBottom)

async function handleSend(message: string) {
  await send(message)
}

function handleConfirm(actionId: string, approved: boolean) {
  confirm(actionId, approved)
}
</script>

<template>
  <div class="flex h-full flex-col">
    <!-- Messages -->
    <div ref="messagesContainer" class="flex-1 overflow-y-auto">
      <!-- Empty state -->
      <div
        v-if="!chatStore.hasMessages && !chatStore.isStreaming"
        class="flex h-full items-center justify-center"
      >
        <div class="text-center">
          <svg class="mx-auto h-12 w-12 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1">
            <path stroke-linecap="round" stroke-linejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
          <h3 class="mt-3 text-lg font-medium text-gray-700">How can I help?</h3>
          <p class="mt-1 text-sm text-gray-500">
            Ask questions about your team's knowledge base, runbooks, or documentation.
          </p>
        </div>
      </div>

      <!-- Message list -->
      <div v-else class="mx-auto max-w-3xl divide-y divide-gray-100">
        <ChatMessage
          v-for="msg in chatStore.messages"
          :key="msg.id"
          :message="msg"
        />
        <StreamingMessage />

        <!-- Confirm dialog -->
        <ConfirmDialog
          v-if="chatStore.pendingConfirm"
          :action="chatStore.pendingConfirm"
          @confirm="handleConfirm"
        />
      </div>
    </div>

    <!-- Input -->
    <ChatInput
      :disabled="chatStore.isStreaming"
      @send="handleSend"
    />
  </div>
</template>
