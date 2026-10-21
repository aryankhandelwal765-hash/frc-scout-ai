import { scouting } from './api'
import type { ScoutingEntryIn } from './types'

const KEY = 'scout_pending_entries'

export function pending(): ScoutingEntryIn[] {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? '[]')
  } catch {
    return []
  }
}

function save(items: ScoutingEntryIn[]) {
  try {
    localStorage.setItem(KEY, JSON.stringify(items))
  } catch {
    // storage full or blocked: the entry stays only in memory for this session
  }
}

export function enqueue(entry: ScoutingEntryIn): ScoutingEntryIn {
  const withId = { ...entry, client_uuid: entry.client_uuid ?? crypto.randomUUID() }
  save([...pending(), withId])
  return withId
}

export interface SyncResult {
  sent: number
  rejected: { entry: ScoutingEntryIn; errors: string[] }[]
}

/** Sends every queued entry in one batch. Created and duplicate items leave the queue; invalid ones are returned. */
export async function sync(): Promise<SyncResult> {
  const queued = pending()
  if (queued.length === 0) return { sent: 0, rejected: [] }
  const { items } = await scouting.batch(queued)
  const keep: ScoutingEntryIn[] = []
  const rejected: SyncResult['rejected'] = []
  let sent = 0
  for (const item of items) {
    if (item.result === 'invalid') rejected.push({ entry: queued[item.index], errors: item.errors })
    else sent++
  }
  save(keep)
  return { sent, rejected }
}
