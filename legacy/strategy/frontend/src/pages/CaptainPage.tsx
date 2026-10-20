import { useState } from 'react'
import { predict } from '../api'
import type { PredictResponse } from '../types'
import { useReader } from '../useReader'
import { MatchSetup, type AnalyzeParams } from '../components/MatchSetup'
import { Results } from '../components/Results'
import { StreamRead } from '../components/StreamRead'
import { SynthCard } from '../components/SynthCard'

export function CaptainPage({ myTeam }: { myTeam: number | null }) {
  const [data, setData] = useState<PredictResponse | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')
  const [side, setSide] = useState<'red' | 'blue'>('red')

  const captain = useReader()

  const analyze = async ({ red, blue, notes, ourSide }: AnalyzeParams) => {
    setRunning(true)
    setError('')
    setData(null)
    setSide(ourSide)
    captain.reset()
    const body = { red_teams: red, blue_teams: blue, notes }
    try {
      const res = await predict(body)
      setData(res)
      await captain.start('/captain/stream', { ...body, our_side: ourSide })
    } catch (e) {
      setError(String(e))
    } finally {
      setRunning(false)
    }
  }

  return (
    <>
      <div className="page-intro">
        <h1>Captain's Briefing</h1>
        <p>
          A pre-match plan for your alliance: what we do as a team, each robot's job, and the
          opponent weakness to exploit. Pick which alliance is yours below.
        </p>
      </div>

      <MatchSetup
        onAnalyze={analyze}
        busy={running}
        myTeam={myTeam}
        variant="captain"
        analyzeLabel="Generate Captain's Briefing"
      />

      {error && <div className="error-box">{error}</div>}
      {data && <Results data={data} />}

      {captain.state.phase !== 'idle' && (
        <SynthCard
          title="Captain's Briefing"
          kicker={`${side.toUpperCase()} alliance · game plan`}
          icon="military_tech"
        >
          <StreamRead state={captain.state} targetTokens={520} thinkingLabel="Planning" />
        </SynthCard>
      )}
    </>
  )
}
