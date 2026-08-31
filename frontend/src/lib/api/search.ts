import apiClient from './client'
import { getAuthToken } from '@/lib/auth-token'
import { SearchRequest, SearchResponse, AskRequest } from '@/lib/types/search'

export const searchApi = {
  // Standard search (non-streaming)
  search: async (params: SearchRequest) => {
    const response = await apiClient.post<SearchResponse>('/search', params)
    return response.data
  },

  // Resolve a source_embedding chunk id to its parent source (for citation clicks)
  resolveChunk: async (chunkId: string) => {
    const response = await apiClient.get(`/search/chunk/${chunkId}`)
    return response.data as {
      id: string
      source_id: string
      source_title: string
      content: string
      order: number
    }
  },

  // Ask with streaming (uses relative URL for Docker compatibility)
  askKnowledgeBase: async (params: AskRequest, signal?: AbortSignal) => {
    // Get auth token using the same logic as apiClient interceptor
    const token = getAuthToken()

    // Use relative URL to leverage Next.js rewrites
    // This works both in dev (Next.js proxy) and production (Docker network)
    const url = '/api/search/ask'

    // Use fetch with ReadableStream for SSE
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token && { Authorization: `Bearer ${token}` })
      },
      body: JSON.stringify(params),
      signal
    })

    if (!response.ok) {
      // Try to extract error message from response
      let errorMessage = `HTTP error! status: ${response.status}`
      try {
        const errorData = await response.json()
        errorMessage = errorData.detail || errorData.message || errorMessage
      } catch {
        // If response isn't JSON, use status text
        errorMessage = response.statusText || errorMessage
      }
      throw new Error(errorMessage)
    }

    if (!response.body) {
      throw new Error('No response body received')
    }

    return response.body
  }
}
