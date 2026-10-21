import { BrowserRouter, Routes, Route, Navigate } from 'react-router'
import { AuthProvider } from './useAuth'
import { ProtectedRoute } from './components/ProtectedRoute'
import { Shell } from './components/Shell'
import { LoginPage } from './pages/LoginPage'
import { SignupPage } from './pages/SignupPage'
import { DashboardPage } from './pages/DashboardPage'
import { ScoutingPage } from './pages/ScoutingPage'
import { PrescoutPage } from './pages/PrescoutPage'
import { StrategyPage } from './pages/StrategyPage'
import { AdminPage } from './pages/AdminPage'
import './global.css'

function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/signup" element={<SignupPage />} />

      <Route
        element={
          <ProtectedRoute>
            <Shell />
          </ProtectedRoute>
        }
      >
        <Route path="/" element={<DashboardPage />} />
        <Route path="/scouting" element={<ScoutingPage />} />
        <Route path="/prescout" element={<PrescoutPage />} />
        <Route path="/strategy" element={<StrategyPage />} />
        <Route path="/admin" element={<AdminPage />} />
      </Route>

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </AuthProvider>
  )
}

export default App
