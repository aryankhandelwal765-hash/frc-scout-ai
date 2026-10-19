import styles from './EmptyStatePage.module.css'

interface EmptyStatePageProps {
  title: string
  description: string
}

export function EmptyStatePage({ title, description }: EmptyStatePageProps) {
  return (
    <div className={styles.container}>
      <div className={styles.content}>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
    </div>
  )
}

// Placeholder pages
export function DashboardPage() {
  return (
    <EmptyStatePage
      title="Dashboard"
      description="Team performance analytics and match insights coming in Phase 1."
    />
  )
}

export function ScoutingPage() {
  return (
    <EmptyStatePage
      title="Scouting"
      description="Field observations and robot analysis tools coming in Phase 1."
    />
  )
}

export function PrescoutPage() {
  return (
    <EmptyStatePage
      title="Pre-scout"
      description="Pre-match preparation and team research coming in Phase 1."
    />
  )
}

export function StrategyPage() {
  return (
    <EmptyStatePage
      title="Strategy"
      description="Alliance strategy and match planning coming in Phase 1."
    />
  )
}
