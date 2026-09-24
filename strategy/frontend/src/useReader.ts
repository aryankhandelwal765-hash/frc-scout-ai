import { useCallback, useRef, useState } from 'react'
import { streamNdjson, type MatchupBody } from './api'
import type { StreamEvent } from './types'

export type ReadPhase = 'idle' | 'reasoning' | 'writing' | 'done' | 'error'

export interface ReaderState {
  phase: ReadPhase
  thinkTokens: number
  contentTokens: number
  elapsed: number // seconds
  text: string
  error?: string
}

const INITIAL: ReaderState = {
  phase: 'idle',
  thinkTokens: 0,
  contentTokens: 0,
  elapsed: 0,
  text: '',
}

/**
 * Drives one NDJSON streamed read and exposes live progress
 * (reasoning vs writing phase, token counts, elapsed clock, text).
 */
export function useReader() {
  const [state, setState] = useState<ReaderState>(INITIAL)
  const startRef = useRef(0)
  const timerRef = useRef<number | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const stopTimer = () => {
    if (timerRef.current !== null) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
  }

  const reset = useCallback(() => {
    stopTimer()
    abortRef.current?.abort()
    abortRef.current = null
    setState(INITIAL)
  }, [])

  const start = useCallback(async (path: string, body: MatchupBody) => {
    stopTimer()
    abortRef.current?.abort()
    const ac = new AbortController()
    abortRef.current = ac
    startRef.current = Date.now()
    setState({ ...INITIAL, phase: 'reasoning' })

    // Tick the elapsed clock smoothly even between tokens.
    timerRef.current = window.setInterval(() => {
      setState((s) =>
        s.phase === 'reasoning' || s.phase === 'writing'
          ? { ...s, elapsed: (Date.now() - startRef.current) / 1000 }
          : s,
      )
    }, 200)

    const onEvent = (ev: StreamEvent) => {
      setState((s) => {
        const elapsed = (Date.now() - startRef.current) / 1000
        switch (ev.type) {
          case 'thinking':
            return { ...s, phase: 'reasoning', thinkTokens: s.thinkTokens + 1, elapsed }
          case 'content':
            return {
              ...s,
              phase: 'writing',
              contentTokens: s.contentTokens + 1,
              text: s.text + (ev.text ?? ''),
              elapsed,
            }
          case 'done':
            return { ...s, phase: 'done', elapsed }
          case 'error':
            return { ...s, phase: 'error', error: ev.text, elapsed }
          default:
            return s
        }
      })
    }

    try {
      await streamNdjson(path, body, onEvent, ac.signal)
      setState((s) => (s.phase === 'error' ? s : { ...s, phase: 'done' }))
    } catch (e) {
      if (!ac.signal.aborted) {
        setState((s) => ({ ...s, phase: 'error', error: String(e) }))
      }
    } finally {
      stopTimer()
    }
  }, [])

  return { state, start, reset }
}
