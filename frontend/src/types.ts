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
