import { useEffect, useState } from 'react'
import './App.css'
import { getHealth, getSettings } from './api'
import type { Health } from './types'
import { Sidebar, type Page } from './components/Sidebar'
import { StatusBar } from './components/StatusBar'
import { SettingsModal } from './components/SettingsModal'
import { MatchAnalysisPage } from './pages/MatchAnalysisPage'
import { CaptainPage } from './pages/CaptainPage'

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [myTeam, setMyTeam] = useState<number | null>(null)
  const [showSettings, setShowSettings] = useState(false)
  const [page, setPage] = useState<Page>('match')

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null))
    getSettings().then((s) => setMyTeam(s.my_team)).catch(() => {})
  }, [])

  return (
    <div className="app-shell">
      <Sidebar
        health={health}
        myTeam={myTeam}
        page={page}
        onNavigate={setPage}
        onOpenSettings={() => setShowSettings(true)}
      />

      <div className="main-col">
        <StatusBar
          health={health}
          myTeam={myTeam}
          page={page}
          onNavigate={setPage}
          onOpenSettings={() => setShowSettings(true)}
        />

        {showSettings && (
          <SettingsModal
            health={health}
            myTeam={myTeam}
            onClose={() => setShowSettings(false)}
            onSaved={setMyTeam}
          />
        )}

        <main className="content">
          {page === 'match' ? (
            <MatchAnalysisPage myTeam={myTeam} />
          ) : (
            <CaptainPage myTeam={myTeam} />
          )}
        </main>

        <footer className="foot">
          FRC Scout AI · Tactical Intelligence System · data {health?.data_provider?.toUpperCase() ?? '—'} ·{' '}
          {health?.synthesis_model ?? ''}
        </footer>
      </div>
    </div>
  )
}
