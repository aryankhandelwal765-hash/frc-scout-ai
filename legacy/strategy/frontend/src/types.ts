// Types mirroring the FastAPI backend responses.

export interface TeamStats {
  team: number
  name: string
  epa: number
  epa_sd: number
  epa_max: number
  auto: number
  teleop: number
  endgame: number
  winrate: number
  wins: number
  losses: number
  ties: number
  count: number
  rank: number | null
  percentile: number | null
  team_count: number | null
}

export interface Archetype {
  team: number
  name: string
  archetype: string
  tier: string
  match_rank?: number
  dominant_phase: string
  phase_shares: { auto: number; teleop: number; endgame: number }
  epa: number
  summary: string
}

// team number (as string) -> short main-job phrase
export type Roles = Record<string, string>

export interface Volatility {
  team: number
  name: string
  epa: number
  cv: number
  floor: number
  ceiling: number
  consistency_score: number
  volatility_rating: string
  summary: string
}

export interface AllianceVolatility {
  teams: Volatility[]
  alliance_floor: number
  alliance_ceiling: number
  alliance_sd: number
  alliance_consistency: number
}

export interface Chemistry {
  overall_chemistry: number
  phase_coverage: number
  top_heaviness: number
  endgame_strength: number
  distinct_phases: string[]
  notes: string[]
  summary: string
}

export interface AllianceLayers {
  archetypes: Archetype[]
  volatility: AllianceVolatility
  chemistry: Chemistry
}

export interface Prediction {
  red_teams: number[]
  blue_teams: number[]
  red_stats: TeamStats[]
  blue_stats: TeamStats[]
  epa_diff: number
  red_predicted_score: number
  blue_predicted_score: number
  red_win_prob: number
  blue_win_prob: number
  winner: string
  missing_teams: number[]
}

export interface PredictResponse {
  prediction: Prediction
  red: AllianceLayers
  blue: AllianceLayers
  my_team: number | null
}

export interface Health {
  status: string
  data_provider: string
  provider_configured: boolean
  provider_reachable: boolean
  ollama_running: boolean
  synthesis_model: string
  bot_model: string
  available_models: string[]
}

export interface TbaMatch {
  key: string
  label: string
  comp_level: string
  match_number: number
  set_number: number
  red: number[]
  blue: number[]
}

// NDJSON stream event.
export interface StreamEvent {
  type: 'thinking' | 'content' | 'bot_start' | 'bot_done' | 'done' | 'error'
  text?: string
  team?: number
  name?: string
  side?: 'red' | 'blue'
}
