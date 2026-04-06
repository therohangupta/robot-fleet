export interface GatewayFetchOptions extends RequestInit {
  baseUrl?: string
}

export async function fetchApi<T>(endpoint: string, options?: GatewayFetchOptions): Promise<T> {
  const baseUrl = options?.baseUrl ?? ''
  const { headers: optHeaders, baseUrl: _ignored, ...restOptions } = options ?? {}
  const response = await fetch(`${baseUrl}${endpoint}`, {
    ...restOptions,
    headers: {
      'Content-Type': 'application/json',
      ...(optHeaders || {}),
    },
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }

  return response.json()
}

