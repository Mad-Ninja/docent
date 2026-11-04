import { useCallback, useEffect, useReducer, useRef } from 'react'
import { readEvents, type Source } from './api'

export interface Message {
  id: number
  role: 'user' | 'assistant'
  text: string
  sources: Source[]
  /** Which generator answered: anthropic, claude-code or extractive. */
  backend?: string
  status: 'streaming' | 'done' | 'stopped' | 'error'
}

type Action =
  | { type: 'ask'; id: number; question: string }
  | { type: 'sources'; id: number; sources: Source[]; backend: string }
  | { type: 'token'; id: number; text: string }
  | { type: 'finish'; id: number; status: Message['status'] }

function reducer(messages: Message[], action: Action): Message[] {
  switch (action.type) {
    case 'ask':
      return [...messages,
        { id: action.id - 1, role: 'user', text: action.question, sources: [], status: 'done' },
        { id: action.id, role: 'assistant', text: '', sources: [], status: 'streaming' }]
    case 'sources':
      return messages.map((m) => (m.id === action.id ? { ...m, sources: action.sources, backend: action.backend } : m))
    case 'token':
      return messages.map((m) => (m.id === action.id ? { ...m, text: m.text + action.text } : m))
    case 'finish':
      return messages.map((m) => (m.id === action.id ? { ...m, status: action.status } : m))
  }
}

/** Chat state + a streaming request to /chat. Only one answer streams at a time; `stop` cancels it. */
export function useChatStream(k: number) {
  const [messages, dispatch] = useReducer(reducer, [])
  const controller = useRef<AbortController | null>(null)
  const nextId = useRef(2)

  const stop = useCallback(() => controller.current?.abort(), [])

  // Leaving the page must not leave a request streaming into an unmounted component.
  useEffect(() => stop, [stop])

  const ask = useCallback(async (question: string) => {
    controller.current?.abort()
    const ctrl = new AbortController()
    controller.current = ctrl
    const id = nextId.current
    nextId.current += 2
    dispatch({ type: 'ask', id, question })
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, k }),
        signal: ctrl.signal,
      })
      if (!res.ok || !res.body) throw new Error(`chat failed: ${res.status}`)
      let status: Message['status'] = 'done'
      for await (const event of readEvents(res.body)) {
        if (event.type === 'sources') dispatch({ type: 'sources', id, sources: event.items, backend: event.backend })
        else if (event.type === 'token') dispatch({ type: 'token', id, text: event.text })
        else if (event.type === 'error') status = 'error'  // generation failed after the stream had started
      }
      dispatch({ type: 'finish', id, status })
    } catch (e) {
      dispatch({ type: 'finish', id, status: ctrl.signal.aborted ? 'stopped' : 'error' })
      if (!ctrl.signal.aborted) console.error(e)
    }
  }, [k])

  const streaming = messages.some((m) => m.status === 'streaming')
  return { messages, ask, stop, streaming }
}
