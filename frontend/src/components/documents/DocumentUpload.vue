<script setup lang="ts">
import { ref } from 'vue'

const emit = defineEmits<{
  upload: [file: File, team: string, tags: string[]]
}>()

const dragOver = ref(false)
const selectedFile = ref<File | null>(null)
const team = ref('platform')
const tagInput = ref('')
const tags = ref<string[]>([])

const teams = ['platform', 'backend', 'frontend', 'devops', 'data']

function onDrop(e: DragEvent) {
  dragOver.value = false
  const file = e.dataTransfer?.files[0]
  if (file) selectedFile.value = file
}

function onFileSelect(e: Event) {
  const input = e.target as HTMLInputElement
  if (input.files?.[0]) selectedFile.value = input.files[0]
}

function addTag() {
  const tag = tagInput.value.trim()
  if (tag && !tags.value.includes(tag)) {
    tags.value.push(tag)
    tagInput.value = ''
  }
}

function removeTag(tag: string) {
  tags.value = tags.value.filter((t) => t !== tag)
}

function handleUpload() {
  if (!selectedFile.value) return
  emit('upload', selectedFile.value, team.value, tags.value)
  selectedFile.value = null
  tags.value = []
}
</script>

<template>
  <div class="rounded-xl border border-gray-200 bg-white p-6">
    <!-- Drop zone -->
    <div
      @dragover.prevent="dragOver = true"
      @dragleave="dragOver = false"
      @drop.prevent="onDrop"
      :class="[
        'flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 transition-colors',
        dragOver ? 'border-blue-400 bg-blue-50' : 'border-gray-300',
        selectedFile ? 'border-emerald-400 bg-emerald-50' : ''
      ]"
    >
      <svg v-if="!selectedFile" class="h-10 w-10 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
        <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
      </svg>
      <svg v-else class="h-10 w-10 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
        <path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      <p class="mt-2 text-sm text-gray-600">
        <template v-if="selectedFile">{{ selectedFile.name }}</template>
        <template v-else>
          Drop a file here or
          <label class="cursor-pointer font-medium text-blue-600 hover:text-blue-500">
            browse
            <input type="file" class="hidden" accept=".md,.txt,.pdf,.html" @change="onFileSelect" />
          </label>
        </template>
      </p>
      <p class="mt-1 text-xs text-gray-400">Markdown, PDF, HTML, or plain text</p>
    </div>

    <!-- Options -->
    <div v-if="selectedFile" class="mt-4 space-y-3">
      <div>
        <label class="block text-sm font-medium text-gray-700">Team</label>
        <select v-model="team" class="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none">
          <option v-for="t in teams" :key="t" :value="t">{{ t }}</option>
        </select>
      </div>

      <div>
        <label class="block text-sm font-medium text-gray-700">Tags</label>
        <div class="mt-1 flex gap-2">
          <input
            v-model="tagInput"
            @keydown.enter.prevent="addTag"
            placeholder="Add tag..."
            class="flex-1 rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
          />
          <button @click="addTag" class="rounded-md bg-gray-100 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-200">Add</button>
        </div>
        <div v-if="tags.length > 0" class="mt-2 flex flex-wrap gap-1">
          <span
            v-for="tag in tags"
            :key="tag"
            class="inline-flex items-center gap-1 rounded-full bg-blue-50 px-2.5 py-0.5 text-xs font-medium text-blue-700"
          >
            {{ tag }}
            <button @click="removeTag(tag)" class="text-blue-400 hover:text-blue-600">&times;</button>
          </span>
        </div>
      </div>

      <button
        @click="handleUpload"
        class="w-full rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 transition-colors"
      >
        Upload & Ingest
      </button>
    </div>
  </div>
</template>
