<script setup lang="ts">
import type { ConfirmAction } from '@/types'

defineProps<{
  action: ConfirmAction
}>()

const emit = defineEmits<{
  confirm: [actionId: string, approved: boolean]
}>()
</script>

<template>
  <div class="mx-auto my-4 max-w-md rounded-xl border-2 border-amber-300 bg-amber-50 p-4 shadow-sm">
    <div class="flex items-start gap-3">
      <div class="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-amber-100">
        <svg class="h-5 w-5 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
          <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
        </svg>
      </div>
      <div class="flex-1">
        <h3 class="text-sm font-semibold text-amber-900">Action Requires Approval</h3>
        <p class="mt-1 text-sm text-amber-700">
          The assistant wants to execute:
        </p>
        <div class="mt-2 rounded-lg bg-white/60 px-3 py-2 text-sm">
          <p class="font-medium text-gray-800">{{ action.tool }}</p>
          <p class="mt-0.5 text-gray-600">{{ action.description }}</p>
        </div>
        <div class="mt-3 flex gap-2">
          <button
            @click="emit('confirm', action.action_id, true)"
            class="rounded-lg bg-emerald-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-emerald-700 transition-colors"
          >
            Approve
          </button>
          <button
            @click="emit('confirm', action.action_id, false)"
            class="rounded-lg border border-gray-300 bg-white px-4 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
          >
            Reject
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
