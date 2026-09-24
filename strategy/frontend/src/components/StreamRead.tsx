import ReactMarkdown from 'react-markdown'
import type { ReaderState } from '../useReader'

interface Props {
  state: ReaderState
  targetTokens?: number
  thinkingLabel?: string
}

export function StreamRead({ state, targetTokens = 480, thinkingLabel = 'Reasoning' }: Props) {
  const { phase, thinkTokens, contentTokens, elapsed, text, error } = state

  let frac = 0
  if (phase === 'reasoning') frac = Math.min(0.15, (thinkTokens / 1200) * 0.15)
  else if (phase === 'writing')
    frac = 0.15 + Math.min(contentTokens / targetTokens, 1) * 0.84
  else if (phase === 'done') frac = 1

  const tps = elapsed > 0 ? contentTokens / elapsed : 0
  const eta = tps > 0 ? Math.max(targetTokens - contentTokens, 0) / tps : 0

  let status = ''
  if (phase === 'reasoning')
    status = `🧠 ${thinkingLabel}…  ·  ⏱ ${elapsed.toFixed(0)}s  ·  ${thinkTokens} reasoning tokens`
  else if (phase === 'writing')
    status = `✍️ Writing  ·  ⏱ ${elapsed.toFixed(0)}s  ·  ${contentTokens} tokens  ·  ${tps.toFixed(1)} tok/s  ·  ~${eta.toFixed(0)}s left (est.)`
  else if (phase === 'done')
    status = `✅ Done in ${elapsed.toFixed(0)}s  ·  ${contentTokens} tokens${thinkTokens ? `  ·  ${thinkTokens} reasoning tokens` : ''}`

  return (
    <div className="stream-read">
      {phase !== 'idle' && phase !== 'error' && (
        <>
          <div className="progress-track">
            <div
              className={`progress-fill ${phase === 'reasoning' || phase === 'writing' ? 'pulse' : ''}`}
              style={{ width: `${frac * 100}%` }}
            />
          </div>
          <div className="progress-status">{status}</div>
        </>
      )}
      {error && <div className="error-box">Ollama error: {error}</div>}
      {text && (
        <div className="markdown">
          <ReactMarkdown>{text}</ReactMarkdown>
          {(phase === 'reasoning' || phase === 'writing') && <span className="cursor">▌</span>}
        </div>
      )}
    </div>
  )
}
