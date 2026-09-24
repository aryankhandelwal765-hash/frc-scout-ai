import { useCallback, useRef, useState } from 'react'
import { streamNdjson, type MatchupBody } from './api'
import type { StreamEvent } from './types'

export interface BotEntry {
  team: number
  name: string
  side: 'red' | 'blue'
  text: string
  done: boolean
}

export interface BotsState {
  running: boolean
  bots: BotEntry[]
  error?: string
}

/** Drives the /bots/stream NDJSON stream (per-robot reads tagged by team). */
export function useBots() {
  const [state, setState] = useState<BotsState>({ running: false, bots: [] })
  const abortRef = useRef<AbortController | null>(null)

  const reset = useCallback(() => {
    abortRef.current?.abort()
    setState({ running: false, bots: [] })
  }, [])

  const start = useCallback(async (body: MatchupBody) => {
    abortRef.current?.abort()
    const ac = new AbortController()
    abortRef.current = ac
    setState({ running: true, bots: [] })

    // Immutable updater — must stay pure: React 18 StrictMode invokes state
    // updaters twice, so any in-place mutation (e.g. b.text += ...) would append
    // each streamed token twice ("TeleTele...").
    const onEvent = (ev: StreamEvent) => {
      setState((s) => {
        if (ev.type === 'bot_start' && ev.team != null) {
          return {
            ...s,
            bots: [...s.bots, { team: ev.team, name: ev.name ?? '', side: ev.side ?? 'red', text: '', done: false }],
          }
        }
        if (ev.type === 'content' && ev.team != null) {
          return {
            ...s,
            bots: s.bots.map((x) =>
              x.team === ev.team && !x.done ? { ...x, text: x.text + (ev.text ?? '') } : x,
            ),
          }
        }
        if (ev.type === 'bot_done' && ev.team != null) {
          return { ...s, bots: s.bots.map((x) => (x.team === ev.team ? { ...x, done: true } : x)) }
        }
        if (ev.type === 'error') return { ...s, running: false, error: ev.text }
        if (ev.type === 'done') return { ...s, running: false }
        return s
      })
    }

    try {
      await streamNdjson('/bots/stream', body, onEvent, ac.signal)
      setState((s) => ({ ...s, running: false }))
    } catch (e) {
      if (!ac.signal.aborted) setState((s) => ({ ...s, running: false, error: String(e) }))
    }
  }, [])

  return { state, start, reset }
}
