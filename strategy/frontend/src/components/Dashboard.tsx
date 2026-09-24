import type { PredictResponse, Roles } from '../types';
import { VictoryBanner } from './VictoryBanner';
import { AllianceSection } from './AllianceSection';
import { RobotCard } from './RobotCard';

interface DashboardProps {
  prediction: PredictResponse;
  roles?: Roles;
}

export function Dashboard({ prediction, roles = {} }: DashboardProps) {
  const { red, blue, prediction: pred } = prediction;

  return (
    <div className="dashboard">
      <VictoryBanner prediction={pred} />

      <div className="dashboard-grid">
        {/* Alliance summaries */}
        <div className="alliance-summaries">
          <AllianceSection side="red" data={red} />
          <AllianceSection side="blue" data={blue} />
        </div>

        {/* Robot cards in two columns */}
        <div className="robots-section">
          <div className="robots-column red">
            <div className="robots-header">ROBOTS</div>
            {pred.red_stats.map((stats, i) => (
              <RobotCard
                key={`${stats.team}-red`}
                stats={stats}
                archetype={red.archetypes[i] || red.archetypes[0]}
                volatility={red.volatility.teams[i] || red.volatility.teams[0]}
                role={roles[stats.team.toString()]}
                side="red"
              />
            ))}
          </div>

          <div className="robots-column blue">
            <div className="robots-header">ROBOTS</div>
            {pred.blue_stats.map((stats, i) => (
              <RobotCard
                key={`${stats.team}-blue`}
                stats={stats}
                archetype={blue.archetypes[i] || blue.archetypes[0]}
                volatility={blue.volatility.teams[i] || blue.volatility.teams[0]}
                role={roles[stats.team.toString()]}
                side="blue"
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
