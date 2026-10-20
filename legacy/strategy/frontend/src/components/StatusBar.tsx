import type { Health } from '../types'
import type { Page } from './Sidebar'

interface Props {
  health: Health | null
  myTeam: number | null
  page: Page
  onNavigate: (p: Page) => void
  onOpenSettings: () => void
}

const CRUMB: Record<Page, string> = {
  match: 'Match Analysis',
  captain: "Captain's Briefing",
}

function Pill({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className={`pill ${ok ? 'ok' : 'bad'}`}>
      <span className="dot" />
      {label}
    </span>
  )
}

export function StatusBar({ health, myTeam, page, onNavigate, onOpenSettings }: Props) {
  return (
    <header className="topbar">
      <div className="topbar-mobile-brand">
        <span className="material-symbols-outlined" style={{ color: 'var(--primary-tint)' }}>robot_2</span>
        SCOUT AI
      </div>

      <div className="topbar-crumb">{CRUMB[page]}</div>

      <nav className="topbar-nav">
        <button className={`nav-item ${page === 'match' ? 'active' : ''}`} onClick={() => onNavigate('match')}>
          Analysis
        </button>
        <button className={`nav-item ${page === 'captain' ? 'active' : ''}`} onClick={() => onNavigate('captain')}>
          Captain
        </button>
      </nav>

      <div className="topbar-right">
        {health && (
          <div className="pills">
            <Pill ok={health.provider_reachable} label={health.data_provider.toUpperCase()} />
            <Pill ok={health.ollama_running} label="Ollama" />
          </div>
        )}
        <button className="btn btn-ghost" onClick={onOpenSettings}>
          ⚙ {myTeam ? `Team ${myTeam}` : 'Settings'}
        </button>
      </div>
    </header>
  )
}
