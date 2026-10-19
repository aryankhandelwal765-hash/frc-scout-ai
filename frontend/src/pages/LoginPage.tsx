import { useState, FormEvent } from 'react'
import { useNavigate, Link } from 'react-router'
import { useAuth } from '../useAuth'
import styles from './AuthPages.module.css'

export function LoginPage() {
  const navigate = useNavigate()
  const { login, error: authError } = useAuth()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    setError(null)

    if (!username || !password) {
      setError('Username and password are required')
      return
    }

    try {
      setLoading(true)
      await login(username, password)
      navigate('/')
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Login failed'
      if (msg.includes('429')) {
        setError('Too many login attempts. Please try again later.')
      } else if (msg.includes('401')) {
        setError('Invalid username or password')
      } else {
        setError(msg)
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.container}>
      <div className={styles.card}>
        <h1>Scout AI</h1>
        <p className={styles.subtitle}>FRC Match Analysis</p>

        <form onSubmit={handleSubmit} className={styles.form}>
          {(error || authError) && <div className={styles.error}>{error || authError}</div>}

          <label>
            <span>Username</span>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              disabled={loading}
              autoComplete="username"
              autoFocus
            />
          </label>

          <label>
            <span>Password</span>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={loading}
              autoComplete="current-password"
            />
          </label>

          <button type="submit" disabled={loading} className={styles.primaryButton}>
            {loading ? 'Logging in...' : 'Login'}
          </button>
        </form>

        <p className={styles.footer}>
          Don't have an account? <Link to="/signup">Sign up</Link>
        </p>
      </div>
    </div>
  )
}
