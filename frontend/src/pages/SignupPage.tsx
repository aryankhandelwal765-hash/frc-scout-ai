import { useState, FormEvent } from 'react'
import { useNavigate, Link } from 'react-router'
import { useAuth } from '../useAuth'
import styles from './AuthPages.module.css'

const USERNAME_PATTERN = /^[A-Za-z0-9_.-]{3,32}$/
const PASSWORD_MIN = 10
const PASSWORD_MAX = 128
const INVITE_CODE_MIN = 8
const INVITE_CODE_MAX = 128

export function SignupPage() {
  const navigate = useNavigate()
  const { signup, error: authError } = useAuth()
  const [formData, setFormData] = useState({
    inviteCode: '',
    username: '',
    displayName: '',
    password: '',
    confirmPassword: ''
  })
  const [loading, setLoading] = useState(false)
  const [errors, setErrors] = useState<Record<string, string>>({})

  const validateForm = () => {
    const newErrors: Record<string, string> = {}

    if (!formData.inviteCode.trim()) {
      newErrors.inviteCode = 'Invite code is required'
    } else if (
      formData.inviteCode.length < INVITE_CODE_MIN ||
      formData.inviteCode.length > INVITE_CODE_MAX
    ) {
      newErrors.inviteCode = `Invite code must be ${INVITE_CODE_MIN}-${INVITE_CODE_MAX} characters`
    }

    if (!formData.username.trim()) {
      newErrors.username = 'Username is required'
    } else if (!USERNAME_PATTERN.test(formData.username)) {
      newErrors.username = 'Username must be 3-32 characters: letters, numbers, dots, hyphens, underscores'
    }

    if (formData.password.length < PASSWORD_MIN) {
      newErrors.password = `Password must be at least ${PASSWORD_MIN} characters`
    } else if (formData.password.length > PASSWORD_MAX) {
      newErrors.password = `Password must be at most ${PASSWORD_MAX} characters`
    }

    if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match'
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()

    if (!validateForm()) {
      return
    }

    try {
      setLoading(true)
      await signup(formData.inviteCode, formData.username, formData.password, formData.displayName)
      navigate('/')
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Signup failed'
      if (msg.includes('400') || msg.includes('invite')) {
        setErrors({ form: 'Invalid or expired invite code' })
      } else if (msg.includes('409')) {
        setErrors({ form: 'Username already exists' })
      } else if (msg.includes('429')) {
        setErrors({ form: 'Too many signup attempts. Please try again later.' })
      } else {
        setErrors({ form: msg })
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.container}>
      <div className={styles.card}>
        <h1>Sign up</h1>
        <p className={styles.subtitle}>Create your Scout AI account</p>

        <form onSubmit={handleSubmit} className={styles.form}>
          {errors.form && <div className={styles.error}>{errors.form}</div>}
          {authError && <div className={styles.error}>{authError}</div>}

          <label>
            <span>Invite Code</span>
            <input
              type="text"
              value={formData.inviteCode}
              onChange={(e) => setFormData({ ...formData, inviteCode: e.target.value })}
              disabled={loading}
              autoComplete="off"
              autoFocus
              aria-invalid={!!errors.inviteCode}
            />
            {errors.inviteCode && <p style={{ color: 'var(--color-error)', fontSize: 'var(--font-size-xs)' }}>{errors.inviteCode}</p>}
          </label>

          <label>
            <span>Username</span>
            <input
              type="text"
              value={formData.username}
              onChange={(e) => setFormData({ ...formData, username: e.target.value })}
              disabled={loading}
              autoComplete="username"
              aria-invalid={!!errors.username}
            />
            {errors.username && <p style={{ color: 'var(--color-error)', fontSize: 'var(--font-size-xs)' }}>{errors.username}</p>}
          </label>

          <label>
            <span>Display Name (optional)</span>
            <input
              type="text"
              value={formData.displayName}
              onChange={(e) => setFormData({ ...formData, displayName: e.target.value })}
              disabled={loading}
              autoComplete="name"
            />
          </label>

          <label>
            <span>Password</span>
            <input
              type="password"
              value={formData.password}
              onChange={(e) => setFormData({ ...formData, password: e.target.value })}
              disabled={loading}
              autoComplete="new-password"
              aria-invalid={!!errors.password}
            />
            {errors.password && <p style={{ color: 'var(--color-error)', fontSize: 'var(--font-size-xs)' }}>{errors.password}</p>}
          </label>

          <label>
            <span>Confirm Password</span>
            <input
              type="password"
              value={formData.confirmPassword}
              onChange={(e) => setFormData({ ...formData, confirmPassword: e.target.value })}
              disabled={loading}
              autoComplete="new-password"
              aria-invalid={!!errors.confirmPassword}
            />
            {errors.confirmPassword && <p style={{ color: 'var(--color-error)', fontSize: 'var(--font-size-xs)' }}>{errors.confirmPassword}</p>}
          </label>

          <button type="submit" disabled={loading} className={styles.primaryButton}>
            {loading ? 'Creating account...' : 'Sign up'}
          </button>
        </form>

        <p className={styles.footer}>
          Already have an account? <Link to="/login">Login</Link>
        </p>
      </div>
    </div>
  )
}
