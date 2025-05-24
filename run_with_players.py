import os
from pathlib import Path
from time import sleep

from rlbot import flat
from rlbot.config import load_player_config
from rlbot.managers import MatchManager

if __name__ == "__main__":
    root_dir = Path(__file__).parent

    match_manager = MatchManager(root_dir)
    AUTO_MATCH = True
    WITH_HUMAN = False
    OPPONENT = True
    N_PLAYERS_PER_TEAM = 11
    player_config_path = os.path.join(root_dir, "src/bot.toml")
    player_configurations = []
    if OPPONENT:
        team_range = range(2)
    else:
        team_range = range(1)
    if AUTO_MATCH:
        auto_start_agents = True
        for team in team_range:
            for _ in range(N_PLAYERS_PER_TEAM):
                player_configurations.append(
                    load_player_config(
                        team=team,
                        path=player_config_path,
                        type=flat.CustomBot(),
                    )
                )
    else:
        auto_start_agents = False
        spawn_id = 0
        for team in team_range:
            for _ in range(N_PLAYERS_PER_TEAM):
                spawn_id += 1
                player_configuration = load_player_config(
                    team=team,
                    path=player_config_path,
                    type=flat.CustomBot(),
                )
                player_configuration.spawn_id = spawn_id
                player_configurations.append(player_configuration)
    if WITH_HUMAN:
        player_configurations.append(
            load_player_config(
                team=1,
                path=player_config_path,
                type=flat.Human(),
            )
        )
    match_manager.start_match(
        config=flat.MatchConfiguration(
            launcher=flat.Launcher.Steam,
            auto_start_agents=auto_start_agents,
            game_mode=flat.GameMode.Soccer,
            enable_state_setting=True,
            enable_rendering=True,
            existing_match_behavior=flat.ExistingMatchBehavior.Restart,
            player_configurations=player_configurations,
            game_map_upk="Labs_Utopia_P",
            # mutators=flat.MutatorSettings(gravity=flat.GravityMutator.Low),
        )
    )

    sleep(5)

    while (
        match_manager.packet is None
        or match_manager.packet.match_info.match_phase != flat.MatchPhase.Ended
    ):
        sleep(0.1)

    match_manager.shut_down()
