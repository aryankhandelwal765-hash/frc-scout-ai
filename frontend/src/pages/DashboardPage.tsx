import { useEffect, useState } from 'react'
import { events } from '../api'
import { pending } from '../offlineQueue'
import { useAuth } from '../useAuth'
import type { EventSummary, Match } from '../types'
import styles from './Pages.module.css'

export function DashboardPage() {
  const { user } = useAuth()
  const [evs, setEvs] = useState<EventSummary[]>([])
  const [matches, setMatches] = useState<Match[]>([])
  const queued = pending().length

  useEffect(() => {
    events.list().then((l) => {
      setEvs(l)
      if (l[0]) events.matches(l[0].key).then(setMatches).catch(() => undefined)
    }).catch(() => undefined)
  }, [])

  const played = matches.filter((m) => m.played).length
  return (
    <div className={styles.page}>
      <h1>Hello, {user?.display_name || user?.username}</h1>
      {queued > 0 && <div className={styles.warn}>{queued} scouting entries are waiting to sync from this device.</div>}
      {evs.length === 0 ? (
        <div className={styles.card}><p className={styles.muted}>No event imported yet. An admin can import one from the Admin page.</p></div>
      ) : (
        <div className={styles.card}>
          <h2>{evs[0].name ?? evs[0].key}</h2>
          <p className={styles.muted}>{matches.length} matches, {played} played</p>
        </div>
      )}
    </div>
  )
}
