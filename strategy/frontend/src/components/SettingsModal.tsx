import { useState } from 'react'
import type { Health } from '../types'
import { saveSettings } from '../api'

interface Props {
  health: Health | null
  myTeam: number | null
  onClose: () => void
  onSaved: (team: number | null) => void
}

export function SettingsModal({ health, myTeam, onClose, onSaved }: Props) {
  const [value, setValue] = useState<string>(myTeam ? String(myTeam) : '')
  const [saving, setSaving] = useState(false)

  const save = async () => {
    setSaving(true)
    try {
      const team = value.trim() ? parseInt(value, 10) : null
      const res = await saveSettings(team)
      onSaved(res.my_team)
      onClose()
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>Settings</h2>

        <label className="field-label">Your team number</label>
        <p className="hint">
          Any match with your team gets a dedicated coaching read — the best moves your
          team should make to win.
        </p>
        <div className="row">
          <input
            className="input"
            type="number"
            placeholder="e.g. 9692"
            value={value}
            onChange={(e) => setValue(e.target.value)}
          />
          <button className="btn btn-primary" onClick={save} disabled={saving}>
            {saving ? 'Saving…' : 'Save'}
          </button>
        </div>

        {health && (
          <div className="status-grid">
            <div>
              <div className="status-head">Data provider</div>
              <div>Provider: <code>{health.data_provider}</code></div>
              <div>API key: {health.provider_configured ? '✅' : '❌'}</div>
              <div>Reachable: {health.provider_reachable ? '✅' : '❌'}</div>
            </div>
            <div>
              <div className="status-head">Ollama</div>
              <div>Running: {health.ollama_running ? '✅' : '❌'}</div>
              <div>Strategy: <code>{health.synthesis_model}</code></div>
              <div>Bots: <code>{health.bot_model}</code></div>
            </div>
          </div>
        )}

        <div className="modal-actions">
          <button className="btn btn-ghost" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  )
}
