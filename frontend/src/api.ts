import type {
  User,
  LoginRequest,
  SignupRequest,
  InviteCreateRequest,
  InviteCreatedResponse,
  UserListResponse,
  DeactivateUserResponse,
  ImportSummaryResponse,
  OkResponse
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
    const error = new ApiError(response.status, response.statusText)
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
    })
}
