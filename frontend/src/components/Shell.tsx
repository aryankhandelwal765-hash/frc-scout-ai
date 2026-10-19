import { useState } from 'react'
import { Outlet, useNavigate, useLocation } from 'react-router'
import { useAuth } from '../useAuth'
import styles from './Shell.module.css'

interface NavItem {
  path: string
  label: string
  roles: string[]
}

const NAV_ITEMS: NavItem[] = [
  { path: '/', label: 'Dashboard', roles: ['scout', 'strategist', 'admin'] },
  { path: '/scouting', label: 'Scouting', roles: ['scout', 'strategist', 'admin'] },
  { path: '/prescout', label: 'Pre-scout', roles: ['scout', 'strategist', 'admin'] },
  { path: '/strategy', label: 'Strategy', roles: ['strategist', 'admin'] },
  { path: '/admin', label: 'Admin', roles: ['admin'] }
]

export function Shell() {
  const navigate = useNavigate()
  const location = useLocation()
  const { user, logout } = useAuth()
  const [sidebarOpen, setSidebarOpen] = useState(false)

  if (!user) {
    return null
  }

  const filteredNavItems = NAV_ITEMS.filter((item) => item.roles.includes(user.role))
  const isCurrentPath = (path: string) => location.pathname === path
  const isPrescoutSection = location.pathname.startsWith('/prescout')

  const handleLogout = async () => {
    await logout()
    navigate('/login')
  }

  return (
    <div className={styles.container}>
      {/* Mobile header */}
      <header className={styles.mobileHeader}>
        <button className={styles.menuButton} onClick={() => setSidebarOpen(!sidebarOpen)}>
          <span></span>
          <span></span>
          <span></span>
        </button>
        <h1 className={styles.appTitle}>Scout AI</h1>
        <div className={styles.userMenu}>
          <span className={styles.username}>{user.display_name || user.username}</span>
        </div>
      </header>

      {/* Sidebar */}
      <nav className={`${styles.sidebar} ${sidebarOpen ? styles.sidebarOpen : ''}`}>
        <div className={styles.sidebarHeader}>
          <h1>Scout AI</h1>
        </div>

        <div className={styles.navSections}>
          <div className={styles.navSection}>
            <div className={styles.sectionLabel}>Main</div>
            <ul className={styles.navList}>
              {filteredNavItems.filter((item) => !item.path.startsWith('/prescout')).map((item) => (
                <li key={item.path}>
                  <button
                    className={`${styles.navItem} ${isCurrentPath(item.path) ? styles.active : ''}`}
                    onClick={() => {
                      navigate(item.path)
                      setSidebarOpen(false)
                    }}
                  >
                    {item.label}
                  </button>
                </li>
              ))}
            </ul>
          </div>

          {filteredNavItems.some((item) => item.path.startsWith('/prescout')) && (
            <div className={styles.navSection}>
              <div className={`${styles.sectionLabel} ${isPrescoutSection ? styles.sectionLabelActive : ''}`}>
                Pre-scout
              </div>
              <ul className={styles.navList}>
                {filteredNavItems.filter((item) => item.path.startsWith('/prescout')).map((item) => (
                  <li key={item.path}>
                    <button
                      className={`${styles.navItem} ${isCurrentPath(item.path) ? styles.active : ''}`}
                      onClick={() => {
                        navigate(item.path)
                        setSidebarOpen(false)
                      }}
                    >
                      {item.label}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <div className={styles.sidebarFooter}>
          <div className={styles.userInfo}>
            <p className={styles.userRole}>{user.role.toUpperCase()}</p>
            <p className={styles.userDisplay}>{user.display_name || user.username}</p>
          </div>
          <button className={styles.logoutButton} onClick={handleLogout}>
            Logout
          </button>
        </div>
      </nav>

      {/* Main content */}
      <main className={styles.main}>
        <Outlet />
      </main>

      {/* Mobile bottom tab bar */}
      <nav className={styles.mobileTabBar}>
        <ul>
          {filteredNavItems.filter((item) => !item.path.startsWith('/prescout')).map((item) => (
            <li key={item.path}>
              <button
                className={`${styles.tabBarItem} ${isCurrentPath(item.path) ? styles.active : ''}`}
                onClick={() => navigate(item.path)}
                title={item.label}
              >
                {item.label}
              </button>
            </li>
          ))}
        </ul>
      </nav>

      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div className={styles.sidebarOverlay} onClick={() => setSidebarOpen(false)} />
      )}
    </div>
  )
}
