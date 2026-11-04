import { createContext, useContext, useMemo, useState, type ReactNode } from 'react'

interface Settings {
  /** How many passages the backend retrieves for an answer. */
  k: number
  showScores: boolean
  setK: (k: number) => void
  setShowScores: (v: boolean) => void
}

const SettingsContext = createContext<Settings | null>(null)

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [k, setK] = useState(6)
  const [showScores, setShowScores] = useState(false)
  // A new object on every render would re-render every consumer; memoise on the values only.
  const value = useMemo(() => ({ k, showScores, setK, setShowScores }), [k, showScores])
  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>
}

export function useSettings(): Settings {
  const ctx = useContext(SettingsContext)
  if (!ctx) throw new Error('useSettings must be used inside <SettingsProvider>')
  return ctx
}
