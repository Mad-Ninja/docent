export interface SearchHit {
  chunk_id: number
  path: string
  heading: string
  content: string
  score: number
}

export interface Source {
  n: number
  path: string
  heading: string
  score: number
}

/** Events of the /chat Server-Sent Events stream. */
export type ChatEvent =
  | { type: 'sources'; items: Source[] }
  | { type: 'token'; text: string }
  | { type: 'done' }

export async function searchDocs(q: string, k: number, signal?: AbortSignal): Promise<SearchHit[]> {
  const res = await fetch(`/api/search?${new URLSearchParams({ q, k: String(k) })}`, { signal })
  if (!res.ok) throw new Error(`search failed: ${res.status}`)
  return res.json()
}

/** Reads an SSE response body and yields parsed events; a network chunk may hold several events or half of one. */
export async function* readEvents(body: ReadableStream<Uint8Array>): AsyncGenerator<ChatEvent> {
  const reader = body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) return
    buffer += value
    let end: number
    while ((end = buffer.indexOf('\n\n')) >= 0) {
      const frame = buffer.slice(0, end)
      buffer = buffer.slice(end + 2)
      if (frame.startsWith('data: ')) yield JSON.parse(frame.slice(6)) as ChatEvent
    }
  }
}
