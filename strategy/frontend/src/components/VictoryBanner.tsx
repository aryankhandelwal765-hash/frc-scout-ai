import type { Prediction } from '../types';

interface VictoryBannerProps {
  prediction: Prediction;
}

export function VictoryBanner({ prediction }: VictoryBannerProps) {
  const redWinPct = (prediction.red_win_prob * 100).toFixed(1);
  const blueWinPct = (prediction.blue_win_prob * 100).toFixed(1);
  const leader = prediction.red_win_prob > 0.5 ? 'RED' : 'BLUE';
  const advantage = Math.abs(prediction.red_win_prob - prediction.blue_win_prob) * 100;

  const redWidth = prediction.red_win_prob * 100;

  return (
    <div className="victory-banner">
      <div className="victory-bar">
        <div
          className="victory-red"
          style={{ width: `${redWidth}%` }}
        />
        <div className="victory-blue" style={{ width: `${100 - redWidth}%` }} />
      </div>

      <div className="victory-text">
        <div className="victory-side red">
          <div className="victory-pct">{redWinPct}%</div>
          <div className="victory-opr">{prediction.red_stats.reduce((sum, t) => sum + t.epa, 0).toFixed(0)} OPR</div>
        </div>

        <div className="victory-center">
          <div className="victory-leader">{leader} FAVORED</div>
          <div className="victory-margin">+{advantage.toFixed(0)}%</div>
        </div>

        <div className="victory-side blue">
          <div className="victory-pct">{blueWinPct}%</div>
          <div className="victory-opr">{prediction.blue_stats.reduce((sum, t) => sum + t.epa, 0).toFixed(0)} OPR</div>
        </div>
      </div>
    </div>
  );
}
