import { useDocumentStore } from '@/stores/documentStore'

export function useDocuments() {
  const store = useDocumentStore()

  async function loadDocuments() {
    await store.fetchDocuments()
  }

  async function uploadFile(file: File, team: string, tags: string[]) {
    return store.upload(file, team, tags)
  }

  async function removeDocument(docId: string) {
    await store.remove(docId)
  }

  function changePage(page: number) {
    store.setPage(page)
  }

  return { loadDocuments, uploadFile, removeDocument, changePage }
}
