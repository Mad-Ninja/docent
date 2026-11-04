import { useDeferredValue, useState } from 'react'
import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { searchDocs } from './api'
import { useSettings } from './settings'

/** Raw retrieval, without an answer: what the model would get as context. */
export function SearchPanel() {
  const { k, showScores } = useSettings()
  const [q, setQ] = useState('')
  // Typing stays responsive; the query follows the deferred value instead of every keystroke.
  const query = useDeferredValue(q.trim())
  const { data, isFetching, error } = useQuery({
    queryKey: ['search', query, k],
    queryFn: ({ signal }) => searchDocs(query, k, signal),
    enabled: query.length >= 2,
    staleTime: 60_000,
    placeholderData: keepPreviousData,
  })

  return (
    <aside className="search">
      <h2>Retrieval</h2>
      <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search the docs…" />
      {isFetching && <div className="note">Searching…</div>}
      {error && <div className="note error">{(error as Error).message}</div>}
      <ol className="hits">
        {data?.map((h) => (
          <li key={h.chunk_id}>
            <div className="heading">{h.heading}{showScores && <span className="score"> {h.score.toFixed(3)}</span>}</div>
            <div className="path">{h.path}</div>
            <p>{h.content.slice(0, 220)}…</p>
          </li>
        ))}
      </ol>
    </aside>
  )
}
