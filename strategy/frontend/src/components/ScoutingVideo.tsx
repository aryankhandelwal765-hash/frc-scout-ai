import { useEffect, useState } from 'react';

interface MatchInfo {
  match_key: string;
  ready_to_render: boolean;
  render_status: 'fresh' | 'stale' | 'missing';
  size_mb: number;
}

export function ScoutingVideo() {
  const [matches, setMatches] = useState<MatchInfo[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [rendering, setRendering] = useState(false);
  const [renderErr, setRenderErr] = useState<string | null>(null);

  const loadMatches = () => {
    fetch('/api/scouting/matches')
      .then((r) => r.json())
      .then((data) => {
        setMatches(data.matches || []);
        // Auto-select first fresh render if any, else first with any render
        const fresh = data.matches?.find((m: MatchInfo) => m.render_status === 'fresh');
        const anyRender = data.matches?.find((m: MatchInfo) => m.render_status !== 'missing');
        setSelected(fresh?.match_key || anyRender?.match_key || null);
        setLoading(false);
      })
      .catch((e) => {
        setError(String(e));
        setLoading(false);
      });
  };

  useEffect(() => loadMatches(), []);

  const triggerRender = async () => {
    if (!selected) return;
    setRendering(true);
    setRenderErr(null);
    try {
      const r = await fetch(`/api/scouting/render/${selected}`, { method: 'POST' });
      if (!r.ok) {
        const txt = await r.text();
        setRenderErr(txt || `${r.status} ${r.statusText}`);
      } else {
        loadMatches(); // refresh list
      }
    } catch (e) {
      setRenderErr(String(e));
    } finally {
      setRendering(false);
    }
  };

  if (loading) {
    return (
      <div className="scouting-video-panel">
        <div className="panel-header">
          <span className="panel-title">Scouting Video</span>
        </div>
        <div className="video-empty">Loading…</div>
      </div>
    );
  }

  if (error || matches.length === 0) {
    return (
      <div className="scouting-video-panel">
        <div className="panel-header">
          <span className="panel-title">Scouting Video</span>
        </div>
        <div className="video-empty">
          {error ? `Error: ${error}` : 'No preprocessed matches available.'}
        </div>
      </div>
    );
  }

  const currentMatch = matches.find((m) => m.match_key === selected);
  const canRender = currentMatch?.ready_to_render && !rendering;
  const canPlay = currentMatch && currentMatch.render_status !== 'missing';

  return (
    <div className="scouting-video-panel">
      <div className="panel-header">
        <span className="panel-title">Scouting Video · Robot Boxes</span>
        <div className="video-controls">
          <select
            className="video-picker"
            value={selected || ''}
            onChange={(e) => {
              setSelected(e.target.value);
              setRenderErr(null);
            }}
          >
            {matches.map((m) => {
              const badge =
                m.render_status === 'fresh'
                  ? '✓'
                  : m.render_status === 'stale'
                  ? '~'
                  : m.ready_to_render
                  ? '○'
                  : '—';
              return (
                <option key={m.match_key} value={m.match_key}>
                  {badge} {m.match_key} {m.size_mb ? `(${m.size_mb} MB)` : ''}
                </option>
              );
            })}
          </select>

          {currentMatch && (
            <button
              className={`render-btn ${
                currentMatch.render_status === 'stale' ? 'stale' : ''
              }`}
              onClick={triggerRender}
              disabled={!canRender}
              title={
                !currentMatch.ready_to_render
                  ? 'Needs preprocessing (run full pipeline)'
                  : currentMatch.render_status === 'fresh'
                  ? 'Re-render this match'
                  : currentMatch.render_status === 'stale'
                  ? 'Re-render (current is stale)'
                  : 'Render this match'
              }
            >
              {rendering
                ? 'Rendering…'
                : currentMatch.render_status === 'missing'
                ? 'Render'
                : 'Re-render'}
            </button>
          )}
        </div>
      </div>

      <div className="video-legend">
        <span className="legend-chip fresh">✓ Fresh</span>
        <span className="legend-chip stale">~ Stale</span>
        <span className="legend-chip ready">○ Ready to render</span>
        <span className="legend-chip none">— No preproc</span>
      </div>

      {renderErr && <div className="video-empty error">Render error: {renderErr}</div>}

      {canPlay ? (
        <video
          key={selected}
          className="scouting-video"
          controls
          preload="metadata"
          src={`/api/scouting/videos/${selected}`}
        />
      ) : (
        <div className="video-empty">
          {rendering
            ? 'Rendering match — this can take a few minutes for a full-length match…'
            : currentMatch?.ready_to_render
            ? 'No render yet. Click Render to build one.'
            : 'This match has no preprocessing. Run the full pipeline first.'}
        </div>
      )}
    </div>
  );
}
