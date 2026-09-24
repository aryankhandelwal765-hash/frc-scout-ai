import { useEffect, useState } from 'react'
import { getEventMatches, getVisionNotes } from '../api'
import type { TbaMatch } from '../types'

export interface AnalyzeParams {
  red: number[]
  blue: number[]
  notes: string
  includeBots: boolean
  ourSide: 'red' | 'blue'
}

interface Props {
  onAnalyze: (p: AnalyzeParams) => void
  busy: boolean
  myTeam: number | null
  variant?: 'analysis' | 'captain'
  analyzeLabel?: string
}

export function MatchSetup({ onAnalyze, busy, myTeam, variant = 'analysis', analyzeLabel }: Props) {
  const [red, setRed] = useState<number[]>([254, 1678, 2056])
  const [blue, setBlue] = useState<number[]>([1114, 148, 971])
  const [notes, setNotes] = useState('')
  const [includeBots, setIncludeBots] = useState(false)
  const [ourSide, setOurSide] = useState<'red' | 'blue'>('red')

  const [eventKey, setEventKey] = useState('')
  const [matches, setMatches] = useState<TbaMatch[]>([])
  const [loadingMatches, setLoadingMatches] = useState(false)
  const [matchErr, setMatchErr] = useState('')
  const [mineOnly, setMineOnly] = useState(false)

  // Captain mode: default "our alliance" to whichever side has my team.
  useEffect(() => {
    if (variant !== 'captain' || !myTeam) return
    if (red.includes(myTeam)) setOurSide('red')
    else if (blue.includes(myTeam)) setOurSide('blue')
  }, [variant, myTeam, red, blue])

  const loadMatches = async () => {
    if (!eventKey.trim()) return
    setLoadingMatches(true)
    setMatchErr('')
    setMatches([])
    try {
      const res = await getEventMatches(eventKey.trim())
      if (!res.matches.length) setMatchErr('No matches found (check the key or schedule).')
      setMatches(res.matches)
    } catch (e) {
      setMatchErr(String(e))
    } finally {
      setLoadingMatches(false)
    }
  }

  const applyMatch = (m: TbaMatch) => {
    setRed(m.red)
    setBlue(m.blue)
  }

  const shown = mineOnly && myTeam
    ? matches.filter((m) => m.red.includes(myTeam) || m.blue.includes(myTeam))
    : matches

  const setTeam = (side: 'red' | 'blue', i: number, val: number) => {
    const arr = side === 'red' ? [...red] : [...blue]
    arr[i] = val
    side === 'red' ? setRed(arr) : setBlue(arr)
  }

  const ours = (side: 'red' | 'blue') =>
    variant === 'captain' && ourSide === side ? <span className="ours-tag">OURS</span> : null

  return (
    <section className="card">
      <h2 className="card-title">Match Setup</h2>

      {/* TBA import */}
      <div className="setup-import">
        <input
          className="input"
          placeholder="Event key — e.g. 2026caclv"
          value={eventKey}
          onChange={(e) => setEventKey(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && loadMatches()}
        />
        <button className="btn btn-ghost" onClick={loadMatches} disabled={loadingMatches}>
          {loadingMatches ? 'Loading…' : 'Import from TBA'}
        </button>
        {myTeam && (
          <label className="check">
            <input type="checkbox" checked={mineOnly} onChange={(e) => setMineOnly(e.target.checked)} />
            Team {myTeam} only
          </label>
        )}
      </div>
      {matchErr && <div className="warn-box">{matchErr}</div>}
      {shown.length > 0 && (
        <select
          className="input select match-select"
          defaultValue=""
          onChange={(e) => {
            const m = shown[Number(e.target.value)]
            if (m) applyMatch(m)
          }}
        >
          <option value="" disabled>Select a match ({shown.length})…</option>
          {shown.map((m, i) => (
            <option key={m.key} value={i}>
              {m.label}: 🔴 {m.red.join(', ')}  vs  🔵 {m.blue.join(', ')}
            </option>
          ))}
        </select>
      )}

      {/* Alliance input cards */}
      <div className="alliance-cards">
        <div className="alliance-card red">
          <div className="alliance-card-head">
            <div className="name">Red Alliance{ours('red')}</div>
            <span className="material-symbols-outlined">group</span>
          </div>
          <div className="team-inputs">
            {red.map((t, i) => (
              <input key={i} className="team-input" type="number" value={t}
                onChange={(e) => setTeam('red', i, Number(e.target.value))} />
            ))}
          </div>
        </div>
        <div className="alliance-card blue">
          <div className="alliance-card-head">
            <div className="name">Blue Alliance{ours('blue')}</div>
            <span className="material-symbols-outlined">group</span>
          </div>
          <div className="team-inputs">
            {blue.map((t, i) => (
              <input key={i} className="team-input" type="number" value={t}
                onChange={(e) => setTeam('blue', i, Number(e.target.value))} />
            ))}
          </div>
        </div>
      </div>

      <div className="setup-controls">
        {variant === 'captain' && (
          <div className="side-picker">
            <span>Our alliance:</span>
            <label className="check">
              <input type="radio" name="ourSide" checked={ourSide === 'red'} onChange={() => setOurSide('red')} /> Red
            </label>
            <label className="check">
              <input type="radio" name="ourSide" checked={ourSide === 'blue'} onChange={() => setOurSide('blue')} /> Blue
            </label>
          </div>
        )}

        <div className="notes-row">
          <textarea
            className="input textarea"
            placeholder="Scouting notes (optional) — e.g. '971 had a dead battery in quals, drivetrain fixed since.'"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
          />
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            title="Fill notes from CV vision output for these teams"
            onClick={async () => {
              try {
                const res = await getVisionNotes({ red_teams: red, blue_teams: blue })
                if (res.notes) setNotes(res.notes)
              } catch {
                // silent — user still has an empty field to type in
              }
            }}
          >
            Auto-fill from Vision
          </button>
        </div>

        {variant === 'analysis' && (
          <label className="check">
            <input type="checkbox" checked={includeBots} onChange={(e) => setIncludeBots(e.target.checked)} />
            Include bot-by-bot analysis (6 quick per-robot reads)
          </label>
        )}

        <button
          className="btn btn-primary btn-lg"
          onClick={() => onAnalyze({ red, blue, notes, includeBots, ourSide })}
          disabled={busy}
        >
          {busy ? 'Working…' : analyzeLabel ?? 'Analyze Match'}
        </button>
      </div>
    </section>
  )
}
