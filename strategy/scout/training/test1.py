import json

import statbotics

sb = statbotics.Statbotics()

team_year = sb.get_team_year(
    2056,
    2026
)

print(
    json.dumps(
        team_year,
        indent=2
    )
)