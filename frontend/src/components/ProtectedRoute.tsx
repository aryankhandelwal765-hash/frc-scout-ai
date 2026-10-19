import { Navigate } from 'react-router'
import { useAuth } from '../useAuth'
import type { Role } from '../types'

interface ProtectedRouteProps {
  children: React.ReactNode
  requiredRoles?: Role[]
}

export function ProtectedRoute({ children, requiredRoles = [] }: ProtectedRouteProps) {
  const { user, loading } = useAuth()

  if (loading) {
    return <div style={{ padding: '20px', textAlign: 'center' }}>Loading...</div>
  }

  if (!user) {
    return <Navigate to="/login" replace />
  }

  if (requiredRoles.length > 0 && !requiredRoles.includes(user.role)) {
    return (
      <div style={{ padding: '20px', textAlign: 'center' }}>
        <h2>Access Denied</h2>
        <p>You don't have permission to access this page.</p>
        <p>Required role: {requiredRoles.join(' or ')}</p>
      </div>
    )
  }

  return <>{children}</>
}
