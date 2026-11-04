import { Chat } from './Chat'
import { SearchPanel } from './SearchPanel'
import { useSettings } from './settings'

export default function App() {
  const { k, setK, showScores, setShowScores } = useSettings()
  return (
    <div className="app">
      <header>
        <h1>Docent</h1>
        <span className="tagline">AI support agent over the FastAPI docs</span>
        <label>
          Sources
          <select value={k} onChange={(e) => setK(Number(e.target.value))}>
            {[3, 6, 10].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>
        <label>
          <input type="checkbox" checked={showScores} onChange={(e) => setShowScores(e.target.checked)} /> Scores
        </label>
      </header>
      <main>
        <Chat />
        <SearchPanel />
      </main>
    </div>
  )
}
