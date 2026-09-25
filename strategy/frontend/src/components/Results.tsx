import { useEffect, useMemo, useState } from 'react'
import { getRoles } from '../api'
import type {
  PredictResponse,
  AllianceLayers,
  TeamStats,
  Volatility,
  Archetype,
  Chemistry,
  Roles,
} from '../types'

// Robot with every quant layer joined by team number.
interface Robot {
  team: number
  name: string
  rank: number | undefined
  stats: TeamStats
  arch: Archetype
  vol: Volatility | undefined
}

function mergeAlliance(stats: TeamStats[], layers: AllianceLayers): Robot[] {
  const archByTeam = new Map(layers.archetypes.map((a) => [a.team, a]))
  const volByTeam = new Map(layers.volatility.teams.map((v) => [v.team, v]))
  return stats
    .map((s) => {
      const arch = archByTeam.get(s.team)!
      return { team: s.team, name: s.name, rank: arch?.match_rank, stats: s, arch, vol: volByTeam.get(s.team) }
    })
    .sort((a, b) => (a.rank ?? 99) - (b.rank ?? 99))
}

function RobotCard({
  r,
  side,
  job,
  rolesLoading,
}: {
  r: Robot
  side: 'red' | 'blue'
  job: string | undefined
  rolesLoading: boolean
}) {
  const [showWarn, setShowWarn] = useState(false)
  const boomBust = r.vol?.volatility_rating === 'Boom-or-Bust'
  const s = r.stats

  return (
    <div className={`unit-card ${side}`}>
      <div className="unit-top">
        <div>
          <div className="unit-rank">RANK #{r.rank ?? '?'}</div>
          <div className="unit-team">
            {r.team}
            {boomBust && (
              <button
                className="warn-flag"
                onClick={() => setShowWarn((v) => !v)}
                title="Volatility warning — click for detail"
                aria-label="Volatility warning"
              >
                !
              </button>
            )}
          </div>
          <div className="unit-name">{r.name}</div>
        </div>
      </div>

      <div className="unit-metrics">
        <div><b>{s.epa.toFixed(0)}</b><span>OPR</span></div>
        <div><b>{s.auto.toFixed(0)}</b><span>Auto</span></div>
        <div><b>{s.teleop.toFixed(0)}</b><span>Teleop</span></div>
        <div><b>{s.endgame.toFixed(0)}</b><span>End</span></div>
      </div>

      <div className="unit-record">
        {(s.winrate * 100).toFixed(0)}% WR · {s.wins}-{s.losses}-{s.ties}
      </div>

      <div className={`unit-job ${job ? '' : 'loading'}`}>
        {job ? <span className="job-quote">“{job}”</span> : rolesLoading ? 'assigning role…' : '—'}
      </div>

      {boomBust && showWarn && r.vol && (
        <div className="unit-warn">
          <span className="material-symbols-outlined">warning</span>
          <span>
            Boom-or-bust — output swings match-to-match (floor ~{r.vol.floor.toFixed(0)} /
            ceiling ~{r.vol.ceiling.toFixed(0)} pts). Plan around the floor.
          </span>
        </div>
      )}
    </div>
  )
}

function ChemStrip({ layers, chem }: { layers: AllianceLayers; chem: Chemistry }) {
  const v = layers.volatility
  return (
    <div className="chem">
      <div>
        <div className="chem-score">{chem.overall_chemistry.toFixed(1)}<span>/10 chemistry</span></div>
        <div className="chem-sub">floor ~{v.alliance_floor.toFixed(0)} / ceiling ~{v.alliance_ceiling.toFixed(0)} pts</div>
      </div>
    </div>
  )
}

function AllianceBlock({
  side, robots, layers, winPct, score, roles, rolesLoading,
}: {
  side: 'red' | 'blue'
  robots: Robot[]
  layers: AllianceLayers
  winPct: number
  score: number
  roles: Roles
  rolesLoading: boolean
}) {
  return (
    <div className={`alliance-block ${side}`}>
      <div className="alliance-block-head">
        <span className="name">{side} alliance</span>
        <span className="meta">{winPct.toFixed(1)}% win · {score.toFixed(0)} pts</span>
        <span className="rule" />
      </div>
      <div className="unit-grid">
        {robots.map((r) => (
          <RobotCard key={r.team} r={r} side={side} job={roles[String(r.team)]} rolesLoading={rolesLoading} />
        ))}
      </div>
      <ChemStrip layers={layers} chem={layers.chemistry} />
    </div>
  )
}

export function Results({ data }: { data: PredictResponse }) {
  const p = data.prediction
  const redPct = p.red_win_prob * 100
  const bluePct = p.blue_win_prob * 100
  const redFav = p.epa_diff > 0

  const redRobots = useMemo(() => mergeAlliance(p.red_stats, data.red), [p.red_stats, data.red])
  const blueRobots = useMemo(() => mergeAlliance(p.blue_stats, data.blue), [p.blue_stats, data.blue])

  // AI main-job per robot — fetched once the match is known.
  const [roles, setRoles] = useState<Roles>({})
  const [rolesLoading, setRolesLoading] = useState(false)
  const redKey = p.red_teams.join(',')
  const blueKey = p.blue_teams.join(',')

  useEffect(() => {
    let cancelled = false
    setRoles({})
    setRolesLoading(true)
    getRoles({ red_teams: p.red_teams, blue_teams: p.blue_teams })
      .then((r) => { if (!cancelled) setRoles(r.roles ?? {}) })
      .catch(() => { if (!cancelled) setRoles({}) })
      .finally(() => { if (!cancelled) setRolesLoading(false) })
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [redKey, blueKey])

  return (
    <>
      {p.missing_teams.length > 0 && (
        <div className="warn-box">
          No data for teams: {p.missing_teams.join(', ')} — using zero placeholders.
        </div>
      )}

      {/* Win probability banner — split by margin of victory */}
      <div className="winbanner">
        <div className={`winbanner-fav ${redFav ? 'red' : 'blue'}`}>
          {redFav ? 'RED' : 'BLUE'} FAVORED · {Math.abs(p.epa_diff).toFixed(0)} pts
        </div>
        <div className="winbanner-side red" style={{ flexGrow: Math.max(redPct, 12) }}>
          <span className="lbl">RED PROBABILITY</span>
          <div className="pct">{redPct.toFixed(1)}%</div>
          <div className="pts">{p.red_predicted_score.toFixed(0)} pts projected</div>
        </div>
        <div className="winbanner-side blue" style={{ flexGrow: Math.max(bluePct, 12) }}>
          <span className="lbl">BLUE PROBABILITY</span>
          <div className="pct">{bluePct.toFixed(1)}%</div>
          <div className="pts">{p.blue_predicted_score.toFixed(0)} pts projected</div>
        </div>
      </div>

      {/* Unit performance — one bento card per robot, ranked #1–6 by OPR */}
      <div>
        <AllianceBlock
          side="red" robots={redRobots} layers={data.red}
          winPct={redPct} score={p.red_predicted_score} roles={roles} rolesLoading={rolesLoading}
        />
        <div style={{ height: 20 }} />
        <AllianceBlock
          side="blue" robots={blueRobots} layers={data.blue}
          winPct={bluePct} score={p.blue_predicted_score} roles={roles} rolesLoading={rolesLoading}
        />
      </div>
    </>
  )
}
