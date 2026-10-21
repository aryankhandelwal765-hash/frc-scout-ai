export type Role = 'scout' | 'strategist' | 'admin'

export interface User {
  id: number
  username: string
  display_name: string | null
  role: Role
  active: boolean
  created_at: string
}

export interface LoginRequest {
  username: string
  password: string
}

export interface SignupRequest {
  invite_code: string
  username: string
  display_name?: string
  password: string
}

export interface InviteCreateRequest {
  role?: Role
  expires_in_hours?: number | null
}

export interface InviteCreatedResponse {
  id: number
  code: string
  role: Role
  expires_at: string | null
}

export interface UserListResponse extends User {}

export interface DeactivateUserResponse extends User {}

export interface ImportSummaryResponse {
  event_key: string
  season: number
  teams: number
  matches: number
  match_teams: number
}

export interface OkResponse {
  ok: boolean
}

export interface ValidationError {
  loc: (string | number)[]
  msg: string
  type: string
}

export interface HTTPValidationError {
  detail: ValidationError[]
}

export type JsonSchema = {
  type?: string
  properties?: Record<string, JsonSchema>
  required?: string[]
  enum?: (string | number)[]
  minimum?: number
  maximum?: number
  maxLength?: number
  description?: string
}

export interface EventSummary {
  key: string
  season: number
  name: string | null
  start_date: string | null
  end_date: string | null
}

export interface MatchTeam {
  team_number: number
  alliance: 'red' | 'blue'
  slot: number | null
}

export interface Match {
  key: string
  comp_level: string | null
  set_number: number | null
  match_number: number | null
  red_score: number | null
  blue_score: number | null
  played: boolean
  teams: MatchTeam[]
}

export interface Season {
  year: number
  game_name: string | null
  scouting_schema: JsonSchema
  prescout_schema: JsonSchema | null
}

export type EntryStatus = 'ok' | 'flagged' | 'rejected'

export interface ScoutingEntryIn {
  match_key: string
  team_number: number
  data: Record<string, unknown>
  notes?: string | null
  client_uuid?: string
}

export interface ScoutingEntry extends ScoutingEntryIn {
  id: number
  scouter_id: number
  status: EntryStatus
  submitted_at: string
}

export interface BatchItem {
  index: number
  client_uuid: string | null
  result: 'created' | 'duplicate' | 'invalid'
  id: number | null
  errors: string[]
}

export interface PrescoutBatch {
  id: number
  name: string
  source_type: string
  target_event_key: string | null
  created_by: number | null
  created_at: string
}

export interface PicklistItem {
  team_number: number
  rank: number
  tier?: string | null
  note?: string | null
}

export interface Picklist {
  id: number
  event_key: string
  name: string
  created_by: number | null
  created_at: string
  items: PicklistItem[]
}

export interface TeamSummary {
  team_number: number
  name: string | null
  event_key: string
  matches_scouted: number
  scouting_means: Record<string, number>
  vision_matches: number
  vision_means: Record<string, number>
  prescout_entries: number
  pit_entries: number
  history_rating: number | null
  history_rank: number | null
}

export interface Prediction {
  red_win_prob: number
  blue_win_prob: number
  rating_diff: number
  unknown_teams: number[]
}

export interface VisionImportResponse {
  event_key: string
  matches: number
  rows: number
}
