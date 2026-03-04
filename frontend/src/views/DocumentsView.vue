<script setup lang="ts">
import { onMounted } from 'vue'
import { useDocumentStore } from '@/stores/documentStore'
import { useDocuments } from '@/composables/useDocuments'
import DocumentUpload from '@/components/documents/DocumentUpload.vue'
import DocumentList from '@/components/documents/DocumentList.vue'
import DocumentFilters from '@/components/documents/DocumentFilters.vue'

const store = useDocumentStore()
const { loadDocuments, uploadFile, removeDocument, changePage } = useDocuments()

onMounted(() => loadDocuments())

async function handleUpload(file: File, team: string, tags: string[]) {
  await uploadFile(file, team, tags)
}

function handleFilter(team: string) {
  store.filterTeam = team
  loadDocuments()
}
</script>

<template>
  <div class="mx-auto max-w-5xl space-y-6 p-6">
    <h2 class="text-xl font-semibold text-gray-900">Document Management</h2>

    <!-- Upload -->
    <DocumentUpload @upload="handleUpload" />

    <!-- Error -->
    <div v-if="store.error" class="rounded-lg bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-600">
      {{ store.error }}
    </div>

    <!-- Filters -->
    <DocumentFilters :selected-team="store.filterTeam" @filter="handleFilter" />

    <!-- List -->
    <DocumentList
      :documents="store.documents"
      :total="store.total"
      :page="store.page"
      :page-size="store.pageSize"
      :loading="store.loading"
      @delete="removeDocument"
      @page-change="changePage"
    />
  </div>
</template>
