<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useChatStore } from '@/stores/chatStore'

const route = useRoute()
const chatStore = useChatStore()

const navItems = [
  { name: 'Chat', path: '/', icon: 'M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z' },
  { name: 'Documents', path: '/documents', icon: 'M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z' },
  { name: 'History', path: '/history', icon: 'M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z' },
]

const recentSessions = computed(() => chatStore.sessions.slice(0, 5))
</script>

<template>
  <aside class="flex w-64 flex-col border-r border-gray-200 bg-white">
    <!-- Logo -->
    <div class="flex h-14 items-center gap-2 border-b border-gray-200 px-4">
      <svg class="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
        <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
      </svg>
      <span class="text-sm font-semibold text-gray-900">KB Assistant</span>
    </div>

    <!-- New Chat Button -->
    <div class="p-3">
      <button
        @click="chatStore.newSession()"
        class="flex w-full items-center gap-2 rounded-lg border border-gray-300 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
      >
        <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
          <path stroke-linecap="round" stroke-linejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
        </svg>
        New Chat
      </button>
    </div>

    <!-- Navigation -->
    <nav class="flex-1 space-y-1 px-3">
      <router-link
        v-for="item in navItems"
        :key="item.path"
        :to="item.path"
        :class="[
          'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
          route.path === item.path
            ? 'bg-blue-50 text-blue-700'
            : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
        ]"
      >
        <svg class="h-5 w-5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
          <path stroke-linecap="round" stroke-linejoin="round" :d="item.icon" />
        </svg>
        {{ item.name }}
      </router-link>

      <!-- Recent Sessions -->
      <div v-if="recentSessions.length > 0" class="mt-6">
        <p class="px-3 text-xs font-semibold uppercase tracking-wider text-gray-400">Recent</p>
        <button
          v-for="session in recentSessions"
          :key="session.session_id"
          @click="chatStore.loadSession(session)"
          class="mt-1 flex w-full items-center gap-2 rounded-lg px-3 py-1.5 text-left text-sm text-gray-600 hover:bg-gray-50"
        >
          <span class="truncate">{{ session.title }}</span>
        </button>
      </div>
    </nav>
  </aside>
</template>
