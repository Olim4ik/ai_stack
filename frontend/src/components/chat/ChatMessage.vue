<script setup lang="ts">
import type { ChatMessage } from '@/types'
import SourcePanel from './SourcePanel.vue'
import ReasoningTrace from './ReasoningTrace.vue'

defineProps<{
  message: ChatMessage
}>()
</script>

<template>
  <div
    :class="[
      'flex gap-3 px-4 py-4',
      message.role === 'user' ? 'bg-white' : 'bg-gray-50'
    ]"
  >
    <!-- Avatar -->
    <div
      :class="[
        'flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full text-sm font-medium',
        message.role === 'user'
          ? 'bg-blue-100 text-blue-700'
          : 'bg-emerald-100 text-emerald-700'
      ]"
    >
      {{ message.role === 'user' ? 'U' : 'A' }}
    </div>

    <!-- Content -->
    <div class="min-w-0 flex-1">
      <p class="text-xs font-medium uppercase tracking-wide text-gray-400 mb-1">
        {{ message.role === 'user' ? 'You' : 'Assistant' }}
      </p>
      <div class="prose prose-sm max-w-none text-gray-800 whitespace-pre-wrap">
        {{ message.content }}
      </div>

      <!-- Reasoning Trace -->
      <ReasoningTrace
        v-if="message.reasoning && message.reasoning.length > 0"
        :steps="message.reasoning"
      />

      <!-- Sources -->
      <SourcePanel
        v-if="message.sources && message.sources.length > 0"
        :sources="message.sources"
      />
    </div>
  </div>
</template>
