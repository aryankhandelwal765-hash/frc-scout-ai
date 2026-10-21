import type {
  User,
  LoginRequest,
  SignupRequest,
  InviteCreateRequest,
  InviteCreatedResponse,
  UserListResponse,
  DeactivateUserResponse,
  ImportSummaryResponse,
  OkResponse,
  EventSummary,
  Match,
  Season,
  ScoutingEntryIn,
  ScoutingEntry,
  BatchItem,
  PrescoutBatch,
  Picklist,
  PicklistItem,
  TeamSummary,
  VisionImportResponse,
  EntryStatus,
  Prediction
} from './types'

const BASE = '/api'

export class ApiError extends Error {
  constructor(
    public status: number,
    public statusText: string,
    message?: string
  ) {
    super(message || `${status} ${statusText}`)
    this.name = 'ApiError'
  }
}

async function fetchApi<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const url = `${BASE}${path}`
  const response = await fetch(url, {
    ...options,
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers
    }
  })

  if (!response.ok) {
    let message: string | undefined
    try {
      const body = await response.json()
      const d = body?.detail
      message = Array.isArray(d) ? d.map((x) => (typeof x === 'string' ? x : x.msg)).join('; ') : d
    } catch {
      // no JSON body
    }
    const error = new ApiError(response.status, response.statusText, message)
    if (response.status === 401) {
      sessionStorage.removeItem('auth_state')
      window.location.href = '/login'
    }
    throw error
  }

  if (response.status === 204) {
    return undefined as T
  }

  return response.json() as Promise<T>
}

// Auth endpoints
export const auth = {
  login: (data: LoginRequest) =>
    fetchApi<User>('/auth/login', {
      method: 'POST',
      body: JSON.stringify(data)
    }),

  signup: (data: SignupRequest) =>
    fetchApi<User>('/auth/signup', {
      method: 'POST',
      body: JSON.stringify(data)
    }),

  logout: () =>
    fetchApi<OkResponse>('/auth/logout', {
      method: 'POST'
    }),

  me: () =>
    fetchApi<User>('/auth/me', {
      method: 'GET'
    })
}

// Admin endpoints
export const admin = {
  createInvite: (data: InviteCreateRequest) =>
    fetchApi<InviteCreatedResponse>('/admin/invites', {
      method: 'POST',
      body: JSON.stringify(data)
    }),

  listUsers: () =>
    fetchApi<UserListResponse[]>('/admin/users', {
      method: 'GET'
    }),

  deactivateUser: (userId: number) =>
    fetchApi<DeactivateUserResponse>(`/admin/users/${userId}/deactivate`, {
      method: 'POST'
    }),

  importEvent: (eventKey: string) =>
    fetchApi<ImportSummaryResponse>(`/admin/import/event/${eventKey}`, {
      method: 'POST'
    }),

  importVision: (eventKey: string) =>
    fetchApi<VisionImportResponse>(`/admin/import/vision/${eventKey}`, {
      method: 'POST'
    })
}

const json = (body: unknown): RequestInit => ({ method: 'POST', body: JSON.stringify(body) })

export const events = {
  list: () => fetchApi<EventSummary[]>('/events'),
  matches: (eventKey: string) => fetchApi<Match[]>(`/events/${eventKey}/matches`),
  teamSummary: (eventKey: string, team: number) =>
    fetchApi<TeamSummary>(`/events/${eventKey}/teams/${team}/summary`)
}

export const seasons = {
  get: (year: number) => fetchApi<Season>(`/seasons/${year}`)
}

export const scouting = {
  submit: (entry: ScoutingEntryIn) => fetchApi<ScoutingEntry>('/scouting/entries', json(entry)),
  batch: (entries: ScoutingEntryIn[]) =>
    fetchApi<{ items: BatchItem[] }>('/scouting/entries/batch', json({ entries })),
  mine: () => fetchApi<ScoutingEntry[]>('/scouting/entries/mine'),
  list: (params: { team?: number; match_key?: string; status?: EntryStatus } = {}) => {
    const q = new URLSearchParams(
      Object.entries(params).map(([k, v]) => [k, String(v)] as [string, string])
    )
    return fetchApi<ScoutingEntry[]>(`/scouting/entries?${q}`)
  },
  setStatus: (id: number, status: EntryStatus) =>
    fetchApi<ScoutingEntry>(`/scouting/entries/${id}/status`, {
      method: 'PATCH',
      body: JSON.stringify({ status })
    })
}

export const prescout = {
  batches: () => fetchApi<PrescoutBatch[]>('/prescout/batches'),
  createBatch: (name: string, target_event_key: string | null) =>
    fetchApi<PrescoutBatch>('/prescout/batches', json({ name, source_type: 'manual', target_event_key })),
  addEntry: (batchId: number, body: Record<string, unknown>) =>
    fetchApi<unknown>(`/prescout/batches/${batchId}/entries`, json(body))
}

export const picklists = {
  list: (eventKey: string) =>
    fetchApi<Picklist[]>(`/picklists?event_key=${encodeURIComponent(eventKey)}`),
  create: (event_key: string, name: string) =>
    fetchApi<Picklist>('/picklists', json({ event_key, name })),
  setItems: (id: number, items: PicklistItem[]) =>
    fetchApi<Picklist>(`/picklists/${id}/items`, { method: 'PUT', body: JSON.stringify({ items }) }),
  remove: (id: number) => fetchApi<OkResponse>(`/picklists/${id}`, { method: 'DELETE' })
}

export const predict = (red: number[], blue: number[]) =>
  fetchApi<Prediction>(`/predict?red=${red.join(',')}&blue=${blue.join(',')}`)
