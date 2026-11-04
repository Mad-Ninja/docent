import { useEffect, useRef, useState, type FormEvent } from 'react'
import { useSettings } from './settings'
import { useChatStream, type Message } from './useChatStream'

const EXAMPLES = ['How do I enable CORS?', 'Run a task after returning a response', 'How do I upload a file?']

export function Chat() {
  const { k, showScores } = useSettings()
  const { messages, ask, stop, streaming } = useChatStream(k)
  const [draft, setDraft] = useState('')
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => bottom.current?.scrollIntoView({ block: 'end' }), [messages])

  function submit(e: FormEvent) {
    e.preventDefault()
    const q = draft.trim()
    if (q.length < 2 || streaming) return
    setDraft('')
    void ask(q)
  }

  return (
    <section className="chat">
      <div className="messages">
        {messages.length === 0 && (
          <div className="empty">
            <p>Ask anything about FastAPI. Answers cite the documentation.</p>
            {EXAMPLES.map((q) => <button key={q} className="chip" onClick={() => void ask(q)}>{q}</button>)}
          </div>
        )}
        {messages.map((m) => <Bubble key={m.id} message={m} showScores={showScores} />)}
        <div ref={bottom} />
      </div>
      <form className="composer" onSubmit={submit}>
        <input value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="Ask a question…" />
        {streaming
          ? <button type="button" onClick={stop}>Stop</button>
          : <button type="submit" disabled={draft.trim().length < 2}>Send</button>}
      </form>
    </section>
  )
}

function Bubble({ message, showScores }: { message: Message; showScores: boolean }) {
  return (
    <div className={`bubble ${message.role}`}>
      <div className="text">{message.text || (message.status === 'streaming' ? '…' : '')}</div>
      {message.backend && message.status !== 'streaming' && <div className="note">answered by: {message.backend}</div>}
      {message.status === 'stopped' && <div className="note">Stopped.</div>}
      {message.status === 'error' && <div className="note error">Something went wrong. Try again.</div>}
      {message.sources.length > 0 && (
        <ol className="sources">
          {message.sources.map((s) => (
            <li key={s.n}>
              <span className="path">{s.path}</span> — {s.heading}
              {showScores && <span className="score"> {s.score.toFixed(3)}</span>}
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}
