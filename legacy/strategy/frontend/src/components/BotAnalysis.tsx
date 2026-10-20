import ReactMarkdown from 'react-markdown'
import type { BotsState } from '../useBots'

export function BotAnalysis({ state }: { state: BotsState }) {
  if (!state.bots.length && !state.running) return null
  return (
    <section className="card">
      <h2 className="card-title">Bot-by-Bot Analysis</h2>
      {state.error && <div className="error-box">Ollama error: {state.error}</div>}
      <div className="bots-grid">
        {state.bots.map((b) => (
          <div key={b.team} className={`bot-card ${b.side}`}>
            <div className="bot-head">
              {b.side === 'red' ? '🔴' : '🔵'} {b.team} <span className="stat-name">{b.name}</span>
              {!b.done && <span className="cursor">▌</span>}
            </div>
            <div className="markdown small">
              <ReactMarkdown>{b.text || '_scouting…_'}</ReactMarkdown>
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}
