<script setup lang="ts">
import { useChatStore } from '@/stores/chatStore'

const chatStore = useChatStore()
const teams = ['platform', 'backend', 'frontend', 'devops', 'data']
</script>

<template>
  <header class="flex h-14 items-center justify-between border-b border-gray-200 bg-white px-6">
    <h1 class="text-lg font-semibold text-gray-900">Knowledge Base Assistant</h1>
    <div class="flex items-center gap-3">
      <label class="text-sm text-gray-500">Team:</label>
      <select
        v-model="chatStore.currentTeam"
        class="rounded-md border border-gray-300 bg-white px-3 py-1.5 text-sm text-gray-700 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
      >
        <option v-for="team in teams" :key="team" :value="team">
          {{ team }}
        </option>
      </select>
    </div>
  </header>
</template>
