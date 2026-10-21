import { useEffect, useMemo, useState } from 'react'
import { events, scouting, seasons } from '../api'
import { SchemaForm } from '../components/SchemaForm'
import { enqueue, pending, sync } from '../offlineQueue'
import type { EventSummary, Match, ScoutingEntry, Season } from '../types'
import styles from './Pages.module.css'

const matchLabel = (m: Match) =>
  `${(m.comp_level ?? '').toUpperCase()} ${m.set_number && m.comp_level !== 'qm' ? `${m.set_number}-` : ''}${m.match_number ?? ''}`

export function ScoutingPage() {
  const [evs, setEvs] = useState<EventSummary[]>([])
  const [eventKey, setEventKey] = useState('')
  const [matches, setMatches] = useState<Match[]>([])
  const [season, setSeason] = useState<Season | null>(null)
  const [matchKey, setMatchKey] = useState('')
  const [team, setTeam] = useState<number | null>(null)
  const [values, setValues] = useState<Record<string, unknown>>({})
  const [notes, setNotes] = useState('')
  const [queued, setQueued] = useState(pending().length)
  const [mine, setMine] = useState<ScoutingEntry[]>([])
  const [msg, setMsg] = useState<{ kind: 'ok' | 'error' | 'warn'; text: string } | null>(null)

  const refreshMine = () => scouting.mine().then(setMine).catch(() => undefined)

  useEffect(() => {
    events.list().then((l) => {
      setEvs(l)
      if (l[0]) setEventKey(l[0].key)
    }).catch((e) => setMsg({ kind: 'error', text: String(e.message) }))
    refreshMine()
  }, [])

  useEffect(() => {
    if (!eventKey) return
    setMatchKey('')
    setTeam(null)
    events.matches(eventKey).then(setMatches).catch(() => setMatches([]))
    const year = evs.find((e) => e.key === eventKey)?.season
    if (year) seasons.get(year).then(setSeason).catch(() => setSeason(null))
  }, [eventKey, evs])

  const flush = async () => {
    try {
      const r = await sync()
      setQueued(pending().length)
      if (r.rejected.length) {
        setMsg({ kind: 'warn', text: `${r.rejected.length} queued entries were rejected: ${r.rejected[0].errors.join('; ')}` })
      } else if (r.sent) {
        setMsg({ kind: 'ok', text: `Synced ${r.sent} queued entries.` })
      }
      refreshMine()
    } catch {
      // still offline; the queue stays
    }
  }

  useEffect(() => {
    flush()
    window.addEventListener('online', flush)
    return () => window.removeEventListener('online', flush)
  }, [])

  const match = useMemo(() => matches.find((m) => m.key === matchKey), [matches, matchKey])

  const submit = async () => {
    if (!matchKey || team === null) return
    const entry = { match_key: matchKey, team_number: team, data: values, notes: notes || null }
    setMsg(null)
    try {
      await scouting.submit(entry)
      setMsg({ kind: 'ok', text: `Saved team ${team}.` })
    } catch (e) {
      const status = (e as { status?: number }).status
      if (status === undefined || status >= 500) {
        enqueue(entry)
        setQueued(pending().length)
        setMsg({ kind: 'warn', text: 'No connection. Saved on this device and will sync later.' })
      } else {
        setMsg({ kind: 'error', text: (e as Error).message })
        return
      }
    }
    setValues({})
    setNotes('')
    setTeam(null)
    refreshMine()
  }

  return (
    <div className={styles.page}>
      <h1>Scouting</h1>
      {queued > 0 && (
        <div className={styles.warn}>
          {queued} entries waiting to sync. <button className={styles.ghost} onClick={flush}>Sync now</button>
        </div>
      )}
      {msg && <div className={styles[msg.kind]}>{msg.text}</div>}

      <div className={styles.card}>
        <label className={styles.field}>
          <span>Event</span>
          <select value={eventKey} onChange={(e) => setEventKey(e.target.value)}>
            {evs.map((e) => (
              <option key={e.key} value={e.key}>{e.name ?? e.key}</option>
            ))}
          </select>
        </label>
        <div className={styles.matchGrid}>
          {matches.map((m) => (
            <button
              key={m.key}
              className={`${styles.chip} ${m.key === matchKey ? styles.chipOn : ''}`}
              onClick={() => { setMatchKey(m.key); setTeam(null) }}
            >
              {matchLabel(m)}
            </button>
          ))}
        </div>
        {matches.length === 0 && <p className={styles.muted}>No matches yet. An admin imports the event from the Admin page.</p>}
      </div>

      {match && (
        <div className={styles.card}>
          <div className={styles.row}>
            {match.teams.map((t) => (
              <button
                key={t.team_number}
                className={`${styles.chip} ${styles[t.alliance]} ${team === t.team_number ? styles.chipOn : ''}`}
                style={{ flex: '1 1 96px' }}
                onClick={() => setTeam(t.team_number)}
              >
                {t.team_number}
              </button>
            ))}
          </div>
          {team !== null && season && (
            <>
              <SchemaForm schema={season.scouting_schema} values={values} onChange={setValues} />
              <label className={styles.field}>
                <span>Notes</span>
                <input className={styles.input} value={notes} maxLength={2000} onChange={(e) => setNotes(e.target.value)} />
              </label>
              <button className={styles.button} onClick={submit}>Submit for team {team}</button>
            </>
          )}
        </div>
      )}

      <div className={styles.card}>
        <h2>My recent entries</h2>
        {mine.length === 0 ? (
          <p className={styles.muted}>Nothing submitted yet.</p>
        ) : (
          <div className={styles.scroll}>
            <table className={styles.table}>
              <thead><tr><th>Match</th><th>Team</th><th>Status</th><th>Time</th></tr></thead>
              <tbody>
                {mine.slice(0, 20).map((e) => (
                  <tr key={e.id}><td>{e.match_key.split('_').pop()}</td><td>{e.team_number}</td><td>{e.status}</td><td>{e.submitted_at}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
