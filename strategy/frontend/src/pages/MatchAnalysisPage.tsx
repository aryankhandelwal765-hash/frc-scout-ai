import { useState } from 'react'
import { predict } from '../api'
import type { PredictResponse } from '../types'
import { useReader } from '../useReader'
import { useBots } from '../useBots'
import { MatchSetup, type AnalyzeParams } from '../components/MatchSetup'
import { Dashboard } from '../components/Dashboard'
import { StreamRead } from '../components/StreamRead'
import { SynthCard } from '../components/SynthCard'
import { BotAnalysis } from '../components/BotAnalysis'
import { ScoutingVideo } from '../components/ScoutingVideo'
import { BotSnapshot } from '../components/BotSnapshot'

export function MatchAnalysisPage({ myTeam }: { myTeam: number | null }) {
  const [data, setData] = useState<PredictResponse | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')
  const [setupVisible, setSetupVisible] = useState(true)

  const strategy = useReader()
  const captain = useReader()
  const bots = useBots()

  const analyze = async ({ red, blue, notes, includeBots, ourSide }: AnalyzeParams) => {
    setRunning(true)
    setError('')
    setData(null)
    strategy.reset()
    captain.reset()
    bots.reset()
    setSetupVisible(false)  // hide match setup after clicked

    const body = { red_teams: red, blue_teams: blue, notes }
    try {
      const res = await predict(body)
      setData(res)

      // Determine our side — myTeam membership wins over the setup default.
      // Analysis page doesn't show a side picker; ourSide is always 'red' from
      // MatchSetup's default state, so we must not trust it. Use myTeam.
      let side: 'red' | 'blue' = 'red'
      if (myTeam) {
        if (blue.includes(myTeam)) side = 'blue'
        else if (red.includes(myTeam)) side = 'red'
      } else if (ourSide) {
        side = ourSide  // fallback: no saved team, respect any explicit picker
      }

      // Run strategy + captain in parallel (fire both, don't await sequentially)
      const strategyPromise = strategy.start('/analyze/stream', body)
      const captainPromise = captain.start('/captain/stream', { ...body, our_side: side })
      const botsPromise = includeBots ? bots.start(body) : Promise.resolve()

      await Promise.all([strategyPromise, captainPromise, botsPromise])
    } catch (e) {
      setError(String(e))
    } finally {
      setRunning(false)
    }
  }

  return (
    <>
      {setupVisible ? (
        <MatchSetup onAnalyze={analyze} busy={running} myTeam={myTeam} />
      ) : (
        <div className="setup-collapsed">
          <button
            className="edit-match-btn"
            onClick={() => setSetupVisible(true)}
          >
            <span>Edit Match</span>
          </button>
        </div>
      )}

      {error && <div className="error-box">{error}</div>}
      {data && <Dashboard prediction={data} />}
      {data && <BotSnapshot prediction={data} />}

      {data && (
        <div className="synth-parallel-grid">
          {strategy.state.phase !== 'idle' && (
            <SynthCard title="Strategy Read" kicker="Neutral Analysis">
              <StreamRead state={strategy.state} targetTokens={480} />
            </SynthCard>
          )}

          {captain.state.phase !== 'idle' && (
            <SynthCard title="Captain Briefing" kicker="Your Alliance">
              <StreamRead state={captain.state} targetTokens={480} />
            </SynthCard>
          )}
        </div>
      )}

      <BotAnalysis state={bots.state} />

      {data && <ScoutingVideo />}
    </>
  )
}
