import type { ReactNode } from 'react'

interface Props {
  title: string
  kicker: string
  icon?: string
  children: ReactNode
}

/** The "Captain Synthesis" glass box: bolt icon header + primary-tinted panel. */
export function SynthCard({ title, kicker, icon = 'bolt', children }: Props) {
  return (
    <section className="synth-card">
      <div className="synth-head">
        <div className="synth-icon">
          <span className="material-symbols-outlined">{icon}</span>
        </div>
        <div>
          <div className="synth-title">{title}</div>
          <div className="synth-kicker">{kicker}</div>
        </div>
      </div>
      {children}
    </section>
  )
}
