import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { Document } from '@/types'
import { listDocuments, ingestDocument, deleteDocument } from '@/api/client'

export const useDocumentStore = defineStore('documents', () => {
  const documents = ref<Document[]>([])
  const total = ref(0)
  const page = ref(1)
  const pageSize = ref(20)
  const loading = ref(false)
  const uploading = ref(false)
  const error = ref<string | null>(null)
  const filterTeam = ref<string>('')

  async function fetchDocuments() {
    loading.value = true
    error.value = null
    try {
      const params: Record<string, any> = {
        page: page.value,
        page_size: pageSize.value,
      }
      if (filterTeam.value) params.team = filterTeam.value
      const result = await listDocuments(params)
      documents.value = result.documents
      total.value = result.total
    } catch (e: any) {
      error.value = e.message || 'Failed to load documents'
    } finally {
      loading.value = false
    }
  }

  async function upload(file: File, team: string, tags: string[]) {
    uploading.value = true
    error.value = null
    try {
      const result = await ingestDocument(file, team, tags)
      await fetchDocuments()
      return result
    } catch (e: any) {
      error.value = e.message || 'Failed to upload document'
      throw e
    } finally {
      uploading.value = false
    }
  }

  async function remove(docId: string) {
    error.value = null
    try {
      await deleteDocument(docId)
      documents.value = documents.value.filter((d) => d.doc_id !== docId)
      total.value--
    } catch (e: any) {
      error.value = e.message || 'Failed to delete document'
    }
  }

  function setPage(p: number) {
    page.value = p
    fetchDocuments()
  }

  return {
    documents,
    total,
    page,
    pageSize,
    loading,
    uploading,
    error,
    filterTeam,
    fetchDocuments,
    upload,
    remove,
    setPage,
  }
})
