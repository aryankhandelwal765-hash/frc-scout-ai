// Typed client for the FRC Scout AI FastAPI backend (proxied at /api).

import type {
  Health,
  PredictResponse,
  Roles,
  StreamEvent,
  TbaMatch,
} from './types'

const BASE = '/api'

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`)
  if (!r.ok) throw new Error((await r.text()) || `${r.status} ${r.statusText}`)
  return r.json()
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error((await r.text()) || `${r.status} ${r.statusText}`)
  return r.json()
}

export const getHealth = () => getJSON<Health>('/health')

export const getSettings = () => getJSON<{ my_team: number | null }>('/settings')

export const saveSettings = (my_team: number | null) =>
  postJSON<{ my_team: number | null }>('/settings', { my_team })

export const getEventMatches = (eventKey: string) =>
  getJSON<{ event_key: string; matches: TbaMatch[]; count: number }>(
    `/events/${encodeURIComponent(eventKey)}/matches`,
  )

export interface MatchupBody {
  red_teams: number[]
  blue_teams: number[]
  notes?: string | null
  model?: string | null
  my_team?: number | null
  our_side?: 'red' | 'blue' | null
}

export const predict = (body: MatchupBody) =>
  postJSON<PredictResponse>('/predict', body)

export interface VisionNotesBody {
  red_teams: number[]
  blue_teams: number[]
  event_prefix?: string
}

export const getVisionNotes = (body: VisionNotesBody) =>
  postJSON<{ notes: string }>('/vision/notes', body)

export const getTeamVision = (team: number, event_prefix = '2026hop') =>
  getJSON<{
    team: number
    matches_count: number
    avg_balls_scored?: number
    max_balls_scored?: number
    id_confidence?: number
  }>(`/vision/team/${team}?event_prefix=${encodeURIComponent(event_prefix)}`)

export const getRoles = (body: MatchupBody) =>
  postJSON<{ roles: Roles }>('/roles', body)

/**
 * POST an NDJSON stream and invoke onEvent for each parsed line.
 * Used for /analyze/stream, /coach/stream, /bots/stream.
 */
export async function streamNdjson(
  path: string,
  body: MatchupBody,
  onEvent: (ev: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const r = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!r.ok || !r.body) {
    throw new Error((await r.text().catch(() => '')) || `${r.status} ${r.statusText}`)
  }
  const reader = r.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? '' // keep the incomplete trailing line
    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed) continue
      try {
        onEvent(JSON.parse(trimmed) as StreamEvent)
      } catch {
        // ignore malformed partial lines
      }
    }
  }
  if (buffer.trim()) {
    try {
      onEvent(JSON.parse(buffer.trim()) as StreamEvent)
    } catch {
      /* ignore */
    }
  }
}
