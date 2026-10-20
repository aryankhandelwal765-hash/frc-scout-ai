import type { AllianceLayers } from '../types';

interface AllianceSectionProps {
  side: 'red' | 'blue';
  data: AllianceLayers;
}

export function AllianceSection({ side, data }: AllianceSectionProps) {
  const archetype = data.archetypes[0]; // top archetype for the alliance
  const chem = data.chemistry;
  const vol = data.volatility;

  const sideLabel = side === 'red' ? 'RED' : 'BLUE';
  const sideClass = `alliance-section ${side}`;

  return (
    <div className={sideClass}>
      <div className="alliance-header">
        <h2 className="alliance-title">{sideLabel}</h2>
        <div className="alliance-badge">{archetype.tier}</div>
      </div>

      <div className="alliance-summary">
        <div className="summary-row">
          <span className="label">OPR</span>
          <span className="value mono">{archetype.epa.toFixed(0)}</span>
        </div>
      </div>

      <div className="phase-breakdown">
        <div className="phase-label">Auto vs Teleop</div>
        <div className="phase-bars">
          <div
            className="phase-bar auto"
            style={{ width: `${archetype.phase_shares.auto * 100}%` }}
            title={`Auto: ${(archetype.phase_shares.auto * 100).toFixed(0)}%`}
          />
          <div
            className="phase-bar teleop"
            style={{ width: `${archetype.phase_shares.teleop * 100}%` }}
            title={`Teleop: ${(archetype.phase_shares.teleop * 100).toFixed(0)}%`}
          />
        </div>
      </div>

      <div className="alliance-metrics">
        <div className="metric">
          <span className="metric-label">Chemistry</span>
          <span className="metric-value">{chem.overall_chemistry.toFixed(1)}/10</span>
        </div>
        <div className="metric">
          <span className="metric-label">Consistency</span>
          <span className="metric-value">{vol.alliance_consistency.toFixed(1)}/10</span>
        </div>
      </div>

      <div className="alliance-notes">
        <div className="note-label">Strategy</div>
        <p>{archetype.summary}</p>
      </div>
    </div>
  );
}
