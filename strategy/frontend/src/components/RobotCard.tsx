import { useState, useRef } from 'react';
import { animate } from 'motion';
import type { Archetype, TeamStats, Volatility } from '../types';

interface RobotCardProps {
  stats: TeamStats;
  archetype: Archetype;
  volatility: Volatility;
  role?: string;
  side: 'red' | 'blue';
}

export function RobotCard({ stats, archetype, volatility, role, side }: RobotCardProps) {
  const [expanded, setExpanded] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);

  const handleTap = async () => {
    if (!cardRef.current) return;

    if (!expanded) {
      // Expand
      await animate(
        cardRef.current,
        { height: 'auto' },
        { type: 'spring', bounce: 0, duration: 0.4 }
      );
      setExpanded(true);
    } else {
      // Collapse
      await animate(
        cardRef.current,
        { height: 'auto' },
        { type: 'spring', bounce: 0, duration: 0.3 }
      );
      setExpanded(false);
    }
  };

  const handlePointerDown = () => {
    if (cardRef.current) {
      animate(cardRef.current, { scale: 0.97 }, { duration: 0.1 });
    }
  };

  const handlePointerUp = () => {
    if (cardRef.current) {
      animate(cardRef.current, { scale: 1 }, { type: 'spring', bounce: 0, duration: 0.2 });
    }
  };

  const boomOrBust = volatility.volatility_rating === 'HIGH-VARIANCE';

  return (
    <div
      ref={cardRef}
      className={`robot-card ${side} ${expanded ? 'expanded' : 'collapsed'}`}
      onPointerDown={handlePointerDown}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      onClick={handleTap}
    >
      <div className="robot-header">
        <div className="robot-rank">
          <span className="rank-number">{archetype.match_rank || '—'}</span>
          <span className="rank-label">/6</span>
        </div>
        <div className="robot-info">
          <h3 className="robot-name">#{stats.team}</h3>
          <p className="robot-subname">{stats.name || 'Unknown'}</p>
        </div>
        <div className="robot-badges">
          {boomOrBust && (
            <div className="badge boom" title="HIGH-VARIANCE">
              !
            </div>
          )}
        </div>
      </div>

      <div className="robot-stats">
        <div className="stat mono">
          <span className="stat-label">OPR</span>
          <span className="stat-value">{stats.epa.toFixed(0)}</span>
        </div>
        <div className="stat-split mono">
          <span title="Auto">A {stats.auto.toFixed(0)}</span>
          <span title="Teleop">T {stats.teleop.toFixed(0)}</span>
        </div>
      </div>

      <div className="robot-record mono">
        <span>{stats.wins}</span>
        <span>-</span>
        <span>{stats.losses}</span>
        <span>-</span>
        <span>{stats.ties}</span>
      </div>

      {role && (
        <div className="robot-job">
          <p className="job-text">{role}</p>
        </div>
      )}

      {expanded && (
        <div className="robot-details">
          <div className="detail-row">
            <span className="detail-label">Floor</span>
            <span className="detail-value mono">{volatility.floor.toFixed(0)}</span>
          </div>
          <div className="detail-row">
            <span className="detail-label">Ceiling</span>
            <span className="detail-value mono">{volatility.ceiling.toFixed(0)}</span>
          </div>
          <div className="detail-row">
            <span className="detail-label">Consistency</span>
            <span className="detail-value">{volatility.consistency_score.toFixed(1)}/10</span>
          </div>
          <div className="detail-row">
            <span className="detail-label">Volatility</span>
            <span className="detail-value">{volatility.volatility_rating}</span>
          </div>
          <p className="detail-summary">{volatility.summary}</p>
        </div>
      )}
    </div>
  );
}
