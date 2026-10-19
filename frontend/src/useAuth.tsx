import { useState, useCallback, useEffect, useContext, createContext } from 'react'
import type { User } from './types'
import { auth } from './api'

interface AuthContextType {
  user: User | null
  loading: boolean
  error: string | null
  isAuthenticated: boolean
  login: (username: string, password: string) => Promise<User>
  signup: (inviteCode: string, username: string, password: string, displayName?: string) => Promise<User>
  logout: () => Promise<void>
  restoreSession: () => Promise<void>
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const restoreSession = useCallback(async () => {
    try {
      setLoading(true)
      const currentUser = await auth.me()
      setUser(currentUser)
      setError(null)
    } catch {
      setUser(null)
      setError(null) // Session expired is normal, not an error
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    restoreSession()
  }, [restoreSession])

  const login = useCallback(
    async (username: string, password: string): Promise<User> => {
      try {
        setError(null)
        const user = await auth.login({ username, password })
        setUser(user)
        return user
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : 'Login failed'
        setError(errorMsg)
        throw err
      }
    },
    []
  )

  const signup = useCallback(
    async (
      inviteCode: string,
      username: string,
      password: string,
      displayName?: string
    ): Promise<User> => {
      try {
        setError(null)
        const user = await auth.signup({
          invite_code: inviteCode,
          username,
          password,
          display_name: displayName
        })
        setUser(user)
        return user
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : 'Signup failed'
        setError(errorMsg)
        throw err
      }
    },
    []
  )

  const logout = useCallback(async () => {
    try {
      await auth.logout()
    } finally {
      setUser(null)
      setError(null)
    }
  }, [])

  const value: AuthContextType = {
    user,
    loading,
    error,
    isAuthenticated: user !== null,
    login,
    signup,
    logout,
    restoreSession
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider')
  }
  return context
}
