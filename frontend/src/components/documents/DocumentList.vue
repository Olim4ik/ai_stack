<script setup lang="ts">
import type { Document } from '@/types'

defineProps<{
  documents: Document[]
  total: number
  page: number
  pageSize: number
  loading: boolean
}>()

const emit = defineEmits<{
  delete: [docId: string]
  pageChange: [page: number]
}>()

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}
</script>

<template>
  <div class="rounded-xl border border-gray-200 bg-white">
    <!-- Header -->
    <div class="flex items-center justify-between border-b border-gray-200 px-6 py-3">
      <h3 class="text-sm font-semibold text-gray-700">Documents ({{ total }})</h3>
    </div>

    <!-- Loading -->
    <div v-if="loading" class="flex items-center justify-center py-12">
      <div class="h-6 w-6 animate-spin rounded-full border-2 border-gray-300 border-t-blue-600" />
    </div>

    <!-- Empty -->
    <div v-else-if="documents.length === 0" class="py-12 text-center text-sm text-gray-400">
      No documents found
    </div>

    <!-- Table -->
    <table v-else class="w-full">
      <thead>
        <tr class="border-b border-gray-100 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
          <th class="px-6 py-3">Title</th>
          <th class="px-6 py-3">Team</th>
          <th class="px-6 py-3">Tags</th>
          <th class="px-6 py-3">Chunks</th>
          <th class="px-6 py-3">Ingested</th>
          <th class="px-6 py-3 text-right">Actions</th>
        </tr>
      </thead>
      <tbody class="divide-y divide-gray-50">
        <tr v-for="doc in documents" :key="doc.doc_id" class="hover:bg-gray-50">
          <td class="px-6 py-3">
            <p class="text-sm font-medium text-gray-900">{{ doc.title }}</p>
            <p class="text-xs text-gray-400">{{ doc.source_file }}</p>
          </td>
          <td class="px-6 py-3">
            <span class="rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-600">
              {{ doc.team }}
            </span>
          </td>
          <td class="px-6 py-3">
            <div class="flex flex-wrap gap-1">
              <span
                v-for="tag in doc.tags"
                :key="tag"
                class="rounded-full bg-blue-50 px-2 py-0.5 text-xs text-blue-600"
              >
                {{ tag }}
              </span>
            </div>
          </td>
          <td class="px-6 py-3 text-sm text-gray-600">{{ doc.chunk_count }}</td>
          <td class="px-6 py-3 text-sm text-gray-500">{{ formatDate(doc.ingested_at) }}</td>
          <td class="px-6 py-3 text-right">
            <button
              @click="emit('delete', doc.doc_id)"
              class="text-sm text-red-500 hover:text-red-700"
            >
              Delete
            </button>
          </td>
        </tr>
      </tbody>
    </table>

    <!-- Pagination -->
    <div v-if="total > pageSize" class="flex items-center justify-between border-t border-gray-200 px-6 py-3">
      <p class="text-sm text-gray-500">
        Page {{ page }} of {{ Math.ceil(total / pageSize) }}
      </p>
      <div class="flex gap-2">
        <button
          @click="emit('pageChange', page - 1)"
          :disabled="page <= 1"
          class="rounded-md border border-gray-300 px-3 py-1 text-sm disabled:opacity-50"
        >
          Previous
        </button>
        <button
          @click="emit('pageChange', page + 1)"
          :disabled="page >= Math.ceil(total / pageSize)"
          class="rounded-md border border-gray-300 px-3 py-1 text-sm disabled:opacity-50"
        >
          Next
        </button>
      </div>
    </div>
  </div>
</template>
