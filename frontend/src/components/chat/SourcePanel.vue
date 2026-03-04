<script setup lang="ts">
import { ref } from 'vue'
import type { Source } from '@/types'

defineProps<{
  sources: Source[]
}>()

const expanded = ref(false)
</script>

<template>
  <div class="mt-3 rounded-lg border border-gray-200 bg-white">
    <button
      @click="expanded = !expanded"
      class="flex w-full items-center justify-between px-3 py-2 text-sm font-medium text-gray-600 hover:bg-gray-50"
    >
      <span class="flex items-center gap-1.5">
        <svg class="h-4 w-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
          <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
        </svg>
        {{ sources.length }} source{{ sources.length !== 1 ? 's' : '' }}
      </span>
      <svg
        :class="['h-4 w-4 text-gray-400 transition-transform', expanded ? 'rotate-180' : '']"
        fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"
      >
        <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
      </svg>
    </button>

    <div v-if="expanded" class="border-t border-gray-100">
      <div
        v-for="(source, i) in sources"
        :key="source.chunk_id"
        :class="['px-3 py-2 text-sm', i > 0 ? 'border-t border-gray-100' : '']"
      >
        <div class="flex items-center justify-between">
          <span class="font-medium text-gray-700">
            {{ source.title || source.metadata?.source_file || source.chunk_id }}
          </span>
          <span class="rounded-full bg-blue-50 px-2 py-0.5 text-xs font-medium text-blue-600">
            {{ (source.score * 100).toFixed(0) }}%
          </span>
        </div>
        <p v-if="source.metadata?.section" class="mt-0.5 text-xs text-gray-400">
          Section: {{ source.metadata.section }}
        </p>
      </div>
    </div>
  </div>
</template>
