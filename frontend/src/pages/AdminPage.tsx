import { useState, useEffect, FormEvent } from 'react'
import { admin, ApiError } from '../api'
import type { User, Role } from '../types'
import styles from './AdminPage.module.css'

export function AdminPage() {
  const [activeTab, setActiveTab] = useState<'invites' | 'users' | 'import'>('invites')
  const [users, setUsers] = useState<User[]>([])
  const [usersLoading, setUsersLoading] = useState(false)
  const [inviteRole, setInviteRole] = useState<Role>('scout')
  const [inviteExpiry, setInviteExpiry] = useState('')
  const [inviteLoading, setInviteLoading] = useState(false)
  const [inviteCode, setInviteCode] = useState<string | null>(null)
  const [inviteError, setInviteError] = useState<string | null>(null)
  const [importEventKey, setImportEventKey] = useState('')
  const [importLoading, setImportLoading] = useState(false)
  const [importError, setImportError] = useState<string | null>(null)
  const [importResult, setImportResult] = useState<any>(null)
  const [visionResult, setVisionResult] = useState<string | null>(null)
  const [deactivateLoading, setDeactivateLoading] = useState<number | null>(null)
  const [deactivateError, setDeactivateError] = useState<string | null>(null)
  const [userToConfirmDeactivate, setUserToConfirmDeactivate] = useState<User | null>(null)

  // Load users on mount and when tab changes
  useEffect(() => {
    if (activeTab === 'users') {
      loadUsers()
    }
  }, [activeTab])

  const loadUsers = async () => {
    try {
      setUsersLoading(true)
      const data = await admin.listUsers()
      setUsers(data)
    } catch (err) {
      console.error('Failed to load users:', err)
    } finally {
      setUsersLoading(false)
    }
  }

  const handleCreateInvite = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    setInviteError(null)
    setInviteCode(null)

    try {
      setInviteLoading(true)
      const expiryHours = inviteExpiry ? parseInt(inviteExpiry) : null
      const result = await admin.createInvite({
        role: inviteRole,
        expires_in_hours: expiryHours || null
      })
      setInviteCode(result.code)
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to create invite'
      setInviteError(msg)
    } finally {
      setInviteLoading(false)
    }
  }

  const handleImportVision = async () => {
    setImportError(null)
    setImportResult(null)
    if (!importEventKey.trim()) {
      setImportError('Event key is required')
      return
    }
    try {
      setImportLoading(true)
      const r = await admin.importVision(importEventKey)
      setVisionResult(`Loaded ${r.rows} video rows across ${r.matches} matches.`)
    } catch (err) {
      setImportError(err instanceof Error ? err.message : 'Failed to import video data')
    } finally {
      setImportLoading(false)
    }
  }

  const handleImportEvent = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    setImportError(null)
    setImportResult(null)

    if (!importEventKey.trim()) {
      setImportError('Event key is required')
      return
    }

    try {
      setImportLoading(true)
      const result = await admin.importEvent(importEventKey)
      setImportResult(result)
      setImportEventKey('')
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to import event'
      const status = err instanceof ApiError ? err.status : 0
      if (status === 409) {
        setImportError('Season not seeded. Please seed the season first.')
      } else if (status === 503) {
        setImportError('TBA API key not configured.')
      } else if (status === 502) {
        setImportError('Failed to reach TBA API.')
      } else {
        setImportError(msg)
      }
    } finally {
      setImportLoading(false)
    }
  }

  const handleDeactivateUser = async (user: User) => {
    try {
      setDeactivateLoading(user.id)
      setDeactivateError(null)
      await admin.deactivateUser(user.id)
      setUsers((prev) => prev.map((u) => (u.id === user.id ? { ...u, active: false } : u)))
      setUserToConfirmDeactivate(null)
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to deactivate user'
      if (msg.includes('409')) {
        setDeactivateError('Cannot deactivate yourself or the last admin.')
      } else {
        setDeactivateError(msg)
      }
    } finally {
      setDeactivateLoading(null)
    }
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  return (
    <div className={styles.container}>
      <h1>Admin Panel</h1>

      <div className={styles.tabs}>
        <button
          className={`${styles.tab} ${activeTab === 'invites' ? styles.active : ''}`}
          onClick={() => setActiveTab('invites')}
        >
          Invites
        </button>
        <button
          className={`${styles.tab} ${activeTab === 'users' ? styles.active : ''}`}
          onClick={() => setActiveTab('users')}
        >
          Users
        </button>
        <button
          className={`${styles.tab} ${activeTab === 'import' ? styles.active : ''}`}
          onClick={() => setActiveTab('import')}
        >
          Import
        </button>
      </div>

      {activeTab === 'invites' && (
        <div className={styles.tabContent}>
          <h2>Create Invite Code</h2>
          {inviteError && <div className={styles.error}>{inviteError}</div>}
          {inviteCode && (
            <div className={styles.success}>
              <p>Invite code created successfully. Show this code only once:</p>
              <div className={styles.codeBox}>
                <code>{inviteCode}</code>
                <button
                  type="button"
                  className={styles.copyButton}
                  onClick={() => copyToClipboard(inviteCode)}
                >
                  Copy
                </button>
              </div>
              <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)' }}>
                Once you leave this page, the code cannot be recovered.
              </p>
            </div>
          )}
          <form onSubmit={handleCreateInvite} className={styles.form}>
            <label>
              <span>Role</span>
              <select value={inviteRole} onChange={(e) => setInviteRole(e.target.value as Role)}>
                <option value="scout">Scout</option>
                <option value="strategist">Strategist</option>
                <option value="admin">Admin</option>
              </select>
            </label>

            <label>
              <span>Expiry (hours, leave empty for no expiry)</span>
              <input
                type="number"
                min="1"
                max="8760"
                value={inviteExpiry}
                onChange={(e) => setInviteExpiry(e.target.value)}
                placeholder="e.g., 24"
              />
            </label>

            <button type="submit" disabled={inviteLoading} className={styles.primaryButton}>
              {inviteLoading ? 'Creating...' : 'Create Invite'}
            </button>
          </form>
        </div>
      )}

      {activeTab === 'users' && (
        <div className={styles.tabContent}>
          <h2>Users</h2>
          {deactivateError && <div className={styles.error}>{deactivateError}</div>}
          {usersLoading ? (
            <p>Loading users...</p>
          ) : users.length === 0 ? (
            <p>No users yet.</p>
          ) : (
            <div className={styles.usersList}>
              {users.map((user) => (
                <div key={user.id} className={styles.userCard}>
                  <div className={styles.userInfo}>
                    <h3>{user.display_name || user.username}</h3>
                    <p className={styles.username}>@{user.username}</p>
                    <p className={styles.role}>Role: {user.role}</p>
                    <p className={styles.created}>
                      Created: {new Date(user.created_at).toLocaleDateString()}
                    </p>
                    <p className={user.active ? styles.active : styles.inactive}>
                      {user.active ? 'Active' : 'Inactive'}
                    </p>
                  </div>
                  {user.active && (
                    <button
                      type="button"
                      className={styles.deactivateButton}
                      onClick={() => setUserToConfirmDeactivate(user)}
                      disabled={deactivateLoading === user.id}
                    >
                      {deactivateLoading === user.id ? 'Deactivating...' : 'Deactivate'}
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}

          {userToConfirmDeactivate && (
            <div className={styles.modal}>
              <div className={styles.modalContent}>
                <h3>Deactivate User?</h3>
                <p>
                  Are you sure you want to deactivate <strong>{userToConfirmDeactivate.username}</strong>?
                </p>
                <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)' }}>
                  They will not be able to log in after this.
                </p>
                <div className={styles.modalButtons}>
                  <button
                    type="button"
                    className={styles.cancelButton}
                    onClick={() => setUserToConfirmDeactivate(null)}
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    className={styles.deactivateButton}
                    onClick={() => handleDeactivateUser(userToConfirmDeactivate)}
                    disabled={deactivateLoading !== null}
                  >
                    Deactivate
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'import' && (
        <div className={styles.tabContent}>
          <h2>Import Event</h2>
          {importError && <div className={styles.error}>{importError}</div>}
          {importResult && (
            <div className={styles.success}>
              <h3>Import Complete</h3>
              <ul>
                <li>Event: {importResult.event_key}</li>
                <li>Season: {importResult.season}</li>
                <li>Teams: {importResult.teams}</li>
                <li>Matches: {importResult.matches}</li>
                <li>Match Teams: {importResult.match_teams}</li>
              </ul>
            </div>
          )}
          <form onSubmit={handleImportEvent} className={styles.form}>
            <label>
              <span>Event Key</span>
              <input
                type="text"
                value={importEventKey}
                onChange={(e) => setImportEventKey(e.target.value)}
                placeholder="e.g., 2026hop"
                pattern="^[0-9]{4}[a-z0-9]{2,30}$"
                disabled={importLoading}
              />
              <p style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)' }}>
                Format: 4 digits + 2-30 lowercase alphanumerics (e.g., 2026hop)
              </p>
            </label>

            <button type="submit" disabled={importLoading} className={styles.primaryButton}>
              {importLoading ? 'Importing...' : 'Import Event'}
            </button>
            <button type="button" onClick={handleImportVision} disabled={importLoading} className={styles.primaryButton}>
              Import video data
            </button>
          </form>
          {visionResult && <div className={styles.success}>{visionResult}</div>}
        </div>
      )}
    </div>
  )
}
