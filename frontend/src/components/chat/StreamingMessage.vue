<script setup lang="ts">
import { computed } from 'vue'
import { useChatStore } from '@/stores/chatStore'
import ReasoningTrace from './ReasoningTrace.vue'
import SourcePanel from './SourcePanel.vue'

const store = useChatStore()

const hasContent = computed(() => store.streamingContent.length > 0)
</script>

<template>
  <div v-if="store.isStreaming" class="flex gap-3 bg-gray-50 px-4 py-4">
    <!-- Avatar -->
    <div class="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-emerald-100 text-sm font-medium text-emerald-700">
      A
    </div>

    <!-- Content -->
    <div class="min-w-0 flex-1">
      <p class="text-xs font-medium uppercase tracking-wide text-gray-400 mb-1">Assistant</p>

      <!-- Reasoning (shown while streaming) -->
      <ReasoningTrace
        v-if="store.streamingReasoning.length > 0"
        :steps="store.streamingReasoning"
        :open="true"
      />

      <!-- Streaming text -->
      <div v-if="hasContent" class="prose prose-sm max-w-none text-gray-800 whitespace-pre-wrap">
        {{ store.streamingContent }}<span class="inline-block h-4 w-1.5 animate-pulse bg-gray-400 ml-0.5 align-middle" />
      </div>

      <!-- Loading indicator -->
      <div v-else class="flex items-center gap-2 text-sm text-gray-400">
        <div class="flex gap-1">
          <span class="h-2 w-2 rounded-full bg-gray-300 animate-bounce" style="animation-delay: 0ms" />
          <span class="h-2 w-2 rounded-full bg-gray-300 animate-bounce" style="animation-delay: 150ms" />
          <span class="h-2 w-2 rounded-full bg-gray-300 animate-bounce" style="animation-delay: 300ms" />
        </div>
        Thinking...
      </div>

      <!-- Sources (shown as they arrive) -->
      <SourcePanel
        v-if="store.streamingSources.length > 0"
        :sources="store.streamingSources"
      />
    </div>
  </div>
</template>
