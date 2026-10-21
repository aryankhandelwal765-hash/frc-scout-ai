import { useEffect, useState } from 'react'
import { events, picklists, predict, scouting } from '../api'
import type { EventSummary, Picklist, PicklistItem, Prediction, ScoutingEntry, TeamSummary } from '../types'
import styles from './Pages.module.css'

function Means({ title, data }: { title: string; data: Record<string, number> }) {
  const rows = Object.entries(data)
  if (rows.length === 0) return <p className={styles.muted}>{title}: no data</p>
  return (
    <div className={styles.scroll}>
      <table className={styles.table}>
        <thead><tr><th colSpan={2}>{title}</th></tr></thead>
        <tbody>
          {rows.map(([k, v]) => <tr key={k}><td>{k.replace(/_/g, ' ')}</td><td>{v}</td></tr>)}
        </tbody>
      </table>
    </div>
  )
}

export function StrategyPage() {
  const [evs, setEvs] = useState<EventSummary[]>([])
  const [eventKey, setEventKey] = useState('')
  const [teamInput, setTeamInput] = useState('')
  const [summary, setSummary] = useState<TeamSummary | null>(null)
  const [flagged, setFlagged] = useState<ScoutingEntry[]>([])
  const [lists, setLists] = useState<Picklist[]>([])
  const [active, setActive] = useState<Picklist | null>(null)
  const [newName, setNewName] = useState('')
  const [addTeam, setAddTeam] = useState('')
  const [err, setErr] = useState<string | null>(null)
  const [redIn, setRedIn] = useState('')
  const [blueIn, setBlueIn] = useState('')
  const [pred, setPred] = useState<Prediction | null>(null)

  const fail = (e: unknown) => setErr((e as Error).message)

  useEffect(() => {
    events.list().then((l) => { setEvs(l); if (l[0]) setEventKey(l[0].key) }).catch(fail)
    scouting.list({ status: 'flagged' }).then(setFlagged).catch(fail)
  }, [])

  useEffect(() => {
    if (!eventKey) return
    setActive(null)
    picklists.list(eventKey).then((l) => { setLists(l); setActive(l[0] ?? null) }).catch(fail)
  }, [eventKey])

  const lookup = async () => {
    setErr(null)
    setSummary(null)
    if (!eventKey || !teamInput) return
    try {
      setSummary(await events.teamSummary(eventKey, parseInt(teamInput, 10)))
    } catch (e) {
      fail(e)
    }
  }

  const runPredict = async () => {
    setErr(null)
    setPred(null)
    const nums = (t: string) => t.split(/[ ,]+/).filter(Boolean).map(Number)
    try {
      setPred(await predict(nums(redIn), nums(blueIn)))
    } catch (e) {
      fail(e)
    }
  }

  const create = async () => {
    if (!newName.trim() || !eventKey) return
    try {
      const p = await picklists.create(eventKey, newName.trim())
      setLists((l) => [p, ...l])
      setActive(p)
      setNewName('')
    } catch (e) { fail(e) }
  }

  const save = async (items: PicklistItem[]) => {
    if (!active) return
    const ranked = items.map((it, i) => ({ ...it, rank: i + 1 }))
    try {
      const p = await picklists.setItems(active.id, ranked)
      setActive(p)
      setLists((l) => l.map((x) => (x.id === p.id ? p : x)))
    } catch (e) { fail(e) }
  }

  const move = (i: number, d: number) => {
    if (!active) return
    const items = [...active.items]
    const j = i + d
    if (j < 0 || j >= items.length) return
    ;[items[i], items[j]] = [items[j], items[i]]
    save(items)
  }

  const review = async (id: number, status: 'ok' | 'rejected') => {
    try {
      await scouting.setStatus(id, status)
      setFlagged((l) => l.filter((e) => e.id !== id))
    } catch (e) { fail(e) }
  }

  return (
    <div className={styles.page}>
      <h1>Strategy</h1>
      {err && <div className={styles.error}>{err}</div>}
      <label className={styles.field}>
        <span>Event</span>
        <select value={eventKey} onChange={(e) => setEventKey(e.target.value)}>
          {evs.map((e) => <option key={e.key} value={e.key}>{e.name ?? e.key}</option>)}
        </select>
      </label>

      <div className={styles.card}>
        <h2>Team summary</h2>
        <div className={styles.row}>
          <input className={styles.input} type="number" inputMode="numeric" placeholder="Team number" value={teamInput} onChange={(e) => setTeamInput(e.target.value)} />
          <button className={styles.button} onClick={lookup}>Look up</button>
        </div>
        {summary && (
          <>
            <p><strong>{summary.team_number}</strong> {summary.name ?? ''}</p>
            {summary.history_rating !== null && (
              <p className={styles.muted}>
                Ten-season rating {Math.round(summary.history_rating)} (rank {summary.history_rank})
              </p>
            )}
            <p className={styles.muted}>
              {summary.matches_scouted} scouted matches, {summary.vision_matches} video matches,{' '}
              {summary.pit_entries} pit, {summary.prescout_entries} pre-scout
            </p>
            <Means title="Scouted averages" data={summary.scouting_means} />
            <Means title="Video averages" data={summary.vision_means} />
          </>
        )}
      </div>

      <div className={styles.card}>
        <h2>Match prediction</h2>
        <div className={styles.row}>
          <input className={styles.input} placeholder="Red teams, e.g. 254 1678 971" value={redIn} onChange={(e) => setRedIn(e.target.value)} />
          <input className={styles.input} placeholder="Blue teams" value={blueIn} onChange={(e) => setBlueIn(e.target.value)} />
          <button className={styles.button} onClick={runPredict}>Predict</button>
        </div>
        {pred && (
          <p>
            Red {Math.round(pred.red_win_prob * 100)}% · Blue {Math.round(pred.blue_win_prob * 100)}%
            {pred.unknown_teams.length > 0 && (
              <span className={styles.muted}> (no history for {pred.unknown_teams.join(', ')}; counted as average)</span>
            )}
          </p>
        )}
      </div>

      <div className={styles.card}>
        <h2>Pick lists</h2>
        <div className={styles.row}>
          <input className={styles.input} placeholder="New list name" value={newName} onChange={(e) => setNewName(e.target.value)} />
          <button className={styles.button} onClick={create}>Create</button>
        </div>
        {lists.length > 0 && (
          <select className={styles.input} value={active?.id ?? ''} onChange={(e) => setActive(lists.find((l) => l.id === Number(e.target.value)) ?? null)}>
            {lists.map((l) => <option key={l.id} value={l.id}>{l.name}</option>)}
          </select>
        )}
        {active && (
          <>
            <div className={styles.scroll}>
              <table className={styles.table}>
                <thead><tr><th>#</th><th>Team</th><th>Tier</th><th /></tr></thead>
                <tbody>
                  {active.items.map((it, i) => (
                    <tr key={it.team_number}>
                      <td>{it.rank}</td>
                      <td>{it.team_number}</td>
                      <td>{it.tier ?? ''}</td>
                      <td>
                        <button className={styles.ghost} onClick={() => move(i, -1)} aria-label="Move up">↑</button>{' '}
                        <button className={styles.ghost} onClick={() => move(i, 1)} aria-label="Move down">↓</button>{' '}
                        <button className={styles.ghost} onClick={() => save(active.items.filter((x) => x.team_number !== it.team_number))}>Remove</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className={styles.row}>
              <input className={styles.input} type="number" inputMode="numeric" placeholder="Add team" value={addTeam} onChange={(e) => setAddTeam(e.target.value)} />
              <button className={styles.button} onClick={() => {
                const n = parseInt(addTeam, 10)
                if (n && !active.items.some((x) => x.team_number === n)) save([...active.items, { team_number: n, rank: 0 }])
                setAddTeam('')
              }}>Add</button>
              <button className={styles.ghost} onClick={async () => {
                await picklists.remove(active.id)
                const rest = lists.filter((l) => l.id !== active.id)
                setLists(rest)
                setActive(rest[0] ?? null)
              }}>Delete list</button>
            </div>
          </>
        )}
      </div>

      <div className={styles.card}>
        <h2>Flagged entries</h2>
        {flagged.length === 0 ? <p className={styles.muted}>Nothing flagged.</p> : flagged.map((e) => (
          <div key={e.id} className={styles.row}>
            <span>{e.match_key.split('_').pop()} · team {e.team_number}</span>
            <button className={styles.ghost} onClick={() => review(e.id, 'ok')}>Keep</button>
            <button className={styles.ghost} onClick={() => review(e.id, 'rejected')}>Reject</button>
          </div>
        ))}
      </div>
    </div>
  )
}
