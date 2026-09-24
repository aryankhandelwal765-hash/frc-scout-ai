import type { Health } from '../types'

export type Page = 'match' | 'captain'

interface Props {
  health: Health | null
  myTeam: number | null
  page: Page
  onNavigate: (p: Page) => void
  onOpenSettings: () => void
}

const NAV: { id: Page; label: string; icon: string }[] = [
  { id: 'match', label: 'Match Analysis', icon: 'analytics' },
  { id: 'captain', label: "Captain's Briefing", icon: 'military_tech' },
]

export function Sidebar({ health, myTeam, page, onNavigate, onOpenSettings }: Props) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <span className="mark material-symbols-outlined">robot_2</span>
        <div className="title">FRC Scout AI</div>
        <div className="sub">Tactical Intel · 2026</div>
      </div>

      <nav className="sidebar-nav">
        {NAV.map((n) => (
          <button
            key={n.id}
            className={`nav-item ${page === n.id ? 'active' : ''}`}
            onClick={() => onNavigate(n.id)}
          >
            <span className="material-symbols-outlined">{n.icon}</span>
            {n.label}
          </button>
        ))}
        <button className="nav-item" onClick={onOpenSettings}>
          <span className="material-symbols-outlined">settings</span>
          {myTeam ? `Settings · Team ${myTeam}` : 'AI Settings'}
        </button>
      </nav>

      <div className="sidebar-status">
        <div className={`side-stat ${health?.provider_reachable ? 'ok' : 'bad'}`}>
          <span className="dot" />
          <span className="material-symbols-outlined">cloud_done</span>
          {health ? `${health.data_provider.toUpperCase()} ${health.provider_reachable ? 'online' : 'offline'}` : 'TBA —'}
        </div>
        <div className={`side-stat ${health?.ollama_running ? 'ok' : 'bad'}`}>
          <span className="dot" />
          <span className="material-symbols-outlined">memory</span>
          {health?.ollama_running ? 'Ollama node' : 'Ollama offline'}
        </div>
      </div>
    </aside>
  )
}
