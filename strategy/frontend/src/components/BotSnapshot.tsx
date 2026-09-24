import { useEffect, useState } from 'react';
import type { PredictResponse } from '../types';

interface Snapshot {
  team: number;
  matches_count: number;
  avg_balls_scored: number;
  max_balls_scored: number;
  id_confidence: number;
  recent_match: string | null;
  clip_url: string | null;
}

interface BotSnapshotProps {
  prediction: PredictResponse;
}

export function BotSnapshot({ prediction }: BotSnapshotProps) {
  const [snaps, setSnaps] = useState<Record<number, Snapshot | { error: string }>>({});
  const [loading, setLoading] = useState(true);

  const allTeams = [
    ...prediction.prediction.red_teams.map((t) => ({ team: t, side: 'red' as const })),
    ...prediction.prediction.blue_teams.map((t) => ({ team: t, side: 'blue' as const })),
  ];

  useEffect(() => {
    let cancelled = false;
    Promise.all(
      allTeams.map(async ({ team }) => {
        try {
          const r = await fetch(`/api/vision/snapshot/${team}`);
          if (!r.ok) return { team, error: `HTTP ${r.status}` };
          return { team, data: (await r.json()) as Snapshot };
        } catch (e) {
          return { team, error: String(e) };
        }
      }),
    ).then((results) => {
      if (cancelled) return;
      const next: Record<number, Snapshot | { error: string }> = {};
      for (const r of results) {
        if ('data' in r && r.data) next[r.team] = r.data;
        else if ('error' in r && r.error) next[r.team] = { error: r.error };
      }
      setSnaps(next);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prediction.prediction.red_teams.join(','), prediction.prediction.blue_teams.join(',')]);

  return (
    <div className="bot-snapshot-panel">
      <div className="panel-header">
        <span className="panel-title">Pre-Match Snapshot · 30s per bot</span>
        {loading && <span className="snapshot-loading">Loading…</span>}
      </div>

      <div className="snapshot-grid">
        {allTeams.map(({ team, side }) => {
          const snap = snaps[team];
          return (
            <div key={team} className={`snapshot-card ${side}`}>
              <div className="snap-header">
                <span className={`snap-team ${side}`}>#{team}</span>
                {snap && 'matches_count' in snap && (
                  <span className="snap-meta mono">
                    {snap.avg_balls_scored} avg · {snap.matches_count} matches
                  </span>
                )}
              </div>

              {snap && 'clip_url' in snap && snap.clip_url ? (
                <video
                  className="snap-video"
                  controls
                  muted
                  preload="metadata"
                  src={`/api${snap.clip_url}`}
                />
              ) : snap && 'error' in snap ? (
                <div className="snap-empty">No vision data</div>
              ) : (
                <div className="snap-empty">Loading clip…</div>
              )}

              {snap && 'recent_match' in snap && snap.recent_match && (
                <div className="snap-footer mono">from {snap.recent_match}</div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
