import { useEffect, useState } from 'react'
import { events, prescout, seasons } from '../api'
import { SchemaForm } from '../components/SchemaForm'
import type { EventSummary, PrescoutBatch, Season } from '../types'
import styles from './Pages.module.css'

export function PrescoutPage() {
  const [evs, setEvs] = useState<EventSummary[]>([])
  const [batches, setBatches] = useState<PrescoutBatch[]>([])
  const [batchId, setBatchId] = useState<number | null>(null)
  const [name, setName] = useState('')
  const [target, setTarget] = useState('')
  const [season, setSeason] = useState<Season | null>(null)
  const [team, setTeam] = useState('')
  const [video, setVideo] = useState('')
  const [notes, setNotes] = useState('')
  const [confidence, setConfidence] = useState('3')
  const [values, setValues] = useState<Record<string, unknown>>({})
  const [msg, setMsg] = useState<{ kind: 'ok' | 'error'; text: string } | null>(null)

  useEffect(() => {
    events.list().then((l) => {
      setEvs(l)
      if (l[0]) seasons.get(l[0].season).then(setSeason).catch(() => undefined)
    }).catch(() => undefined)
    // Batch listing is strategist-only; a scout starts from a batch they create here.
    prescout.batches().then(setBatches).catch(() => undefined)
  }, [])

  const createBatch = async () => {
    if (!name.trim()) return
    try {
      const b = await prescout.createBatch(name.trim(), target || null)
      setBatches((l) => [b, ...l])
      setBatchId(b.id)
      setName('')
    } catch (e) {
      setMsg({ kind: 'error', text: (e as Error).message })
    }
  }

  const add = async () => {
    if (batchId === null || !team || !season) return
    try {
      await prescout.addEntry(batchId, {
        team_number: parseInt(team, 10),
        season_year: season.year,
        data: values,
        video_url: video || null,
        notes: notes || null,
        confidence: parseInt(confidence, 10)
      })
      setMsg({ kind: 'ok', text: `Saved pre-scout for team ${team}.` })
      setTeam('')
      setValues({})
      setNotes('')
      setVideo('')
    } catch (e) {
      setMsg({ kind: 'error', text: (e as Error).message })
    }
  }

  return (
    <div className={styles.page}>
      <h1>Pre-scout</h1>
      <p className={styles.muted}>Kept apart from live match data. Nothing here is mixed into match entries.</p>
      {msg && <div className={styles[msg.kind]}>{msg.text}</div>}

      <div className={styles.card}>
        <h2>Batch</h2>
        <div className={styles.row}>
          <input className={styles.input} placeholder="New batch name" value={name} onChange={(e) => setName(e.target.value)} />
          <select className={styles.input} value={target} onChange={(e) => setTarget(e.target.value)}>
            <option value="">No target event</option>
            {evs.map((e) => <option key={e.key} value={e.key}>{e.name ?? e.key}</option>)}
          </select>
          <button className={styles.button} onClick={createBatch}>Create</button>
        </div>
        {batches.length > 0 && (
          <select className={styles.input} value={batchId ?? ''} onChange={(e) => setBatchId(e.target.value ? Number(e.target.value) : null)}>
            <option value="">Choose a batch…</option>
            {batches.map((b) => <option key={b.id} value={b.id}>{b.name}</option>)}
          </select>
        )}
      </div>

      {batchId !== null && season?.prescout_schema && (
        <div className={styles.card}>
          <h2>Add a team</h2>
          <div className={styles.fields}>
            <label className={styles.field}>
              <span>Team number *</span>
              <input type="number" inputMode="numeric" value={team} onChange={(e) => setTeam(e.target.value)} />
            </label>
            <label className={styles.field}>
              <span>Confidence (1-5)</span>
              <input type="number" min={1} max={5} value={confidence} onChange={(e) => setConfidence(e.target.value)} />
            </label>
          </div>
          <SchemaForm schema={season.prescout_schema} values={values} onChange={setValues} />
          <label className={styles.field}>
            <span>Video link</span>
            <input className={styles.input} value={video} maxLength={500} onChange={(e) => setVideo(e.target.value)} />
          </label>
          <label className={styles.field}>
            <span>Notes</span>
            <input className={styles.input} value={notes} maxLength={2000} onChange={(e) => setNotes(e.target.value)} />
          </label>
          <button className={styles.button} disabled={!team} onClick={add}>Save</button>
        </div>
      )}
    </div>
  )
}
