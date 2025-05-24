import numpy as np
import torch
from rlbot.flat import (
    Color,
    ControllerState,
    DesiredCarState,
    DesiredPhysics,
    GamePacket,
    MatchPhase,
    Rotator,
    RotatorPartial,
    Vector3,
    Vector3Partial,
)
from rlbot.managers import Bot
from rlgym_compat import V1GameState as GameState
from rlgym_compat.sim_extra_info import SimExtraInfo

from agent import Agent
from nexto_obs import BOOST_LOCATIONS, NextoObsBuilder

KICKOFF_CONTROLS = (
    11 * 4 * [ControllerState(throttle=1, boost=True)]
    + 4 * 4 * [ControllerState(throttle=1, boost=True, steer=-1)]
    + 2 * 4 * [ControllerState(throttle=1, jump=True, boost=True)]
    + 1 * 4 * [ControllerState(throttle=1, boost=True)]
    + 1 * 4 * [ControllerState(throttle=1, yaw=0.8, pitch=-0.7, jump=True, boost=True)]
    + 13 * 4 * [ControllerState(throttle=1, pitch=1, boost=True)]
    + 10 * 4 * [ControllerState(throttle=1, roll=1, pitch=0.5)]
)

KICKOFF_NUMPY = np.array(
    [
        [
            scs.throttle,
            scs.steer,
            scs.pitch,
            scs.yaw,
            scs.roll,
            scs.jump,
            scs.boost,
            scs.handbrake,
        ]
        for scs in KICKOFF_CONTROLS
    ]
)


class Nexto(Bot):
    def initialize(self):
        # name,
        # team,
        # index,
        beta = 1
        render = False
        hardcoded_kickoffs = (True,)
        stochastic_kickoffs = True

        self.obs_builder = None
        self.agent = Agent()
        self.tick_skip = 8

        # Beta controls randomness:
        # 1=best action, 0.5=sampling from probability, 0=random, -1=worst action, or anywhere inbetween
        self.beta = beta
        self.render = render
        self.hardcoded_kickoffs = hardcoded_kickoffs
        self.stochastic_kickoffs = stochastic_kickoffs

        self.game_state: GameState = None
        self.controls = None
        self.action = None
        self.update_action = True
        self.ticks = 0
        self.prev_time = 0
        self.kickoff_index = -1
        self.gamemode = None

        self.orangeGoals = 0
        self.blueGoals = 0
        self.demoedCount = 0
        self.lastFrameBall = None
        self.lastFrameDemod = False
        self.lastPacket = None

        # self.extra_info = SimExtraInfo(
        #     self.field_info, self.match_config, tick_skip=self.tick_skip
        # )
        self.game_state = GameState(
            self.field_info,
            self.match_config,
            tick_skip=self.tick_skip,
            standard_map=False,
        )

        # Initialize the rlgym GameState object now that the game is active and the info is available
        self.obs_builder = NextoObsBuilder(field_info=self.field_info)
        self.ticks = self.tick_skip  # So we take an action the first tick
        self.prev_time = 0
        self.controls = ControllerState()
        self.action = np.zeros(8)
        self.update_action = True
        self.kickoff_index = -1

        self.logger.info(f"Nexto Ready - Index: {self.index}")
        self.logger.info(
            "Remember to run Nexto at 120fps with vsync off! "
            + "Stable 240/360 is second best if that's better for your eyes"
        )
        self.logger.info(
            "Also check out the RLGym Twitch stream to watch live bot training and occasional showmatches!"
        )

    def render_attention_weights(self, weights, positions, n=3):
        if weights is None:
            return
        mean_weights = torch.mean(torch.stack(weights), dim=0).numpy()[0][0]

        top = sorted(
            range(len(mean_weights)), key=lambda i: mean_weights[i], reverse=True
        )
        top.remove(0)  # Self

        self.renderer.begin_rendering("attention_weights")

        invert = np.array([-1, -1, 1]) if self.team == 1 else np.ones(3)
        loc = positions[0] * invert
        mx = mean_weights[~(np.arange(len(mean_weights)) == 1)].max()
        c = 1
        for i in top[:n]:
            weight = mean_weights[i] / mx
            dest = positions[i] * invert
            color = self.renderer.create_color(
                255, round(255 * (1 - weight)), round(255), round(255 * (1 - weight))
            )
            self.renderer.draw_string_3d(dest, 2, 2, str(c), color)
            c += 1
            self.renderer.draw_line_3d(loc, dest, color)
        self.renderer.end_rendering()
        # self.logger.info("Completed initialization")

    def get_output(self, packet: GamePacket) -> ControllerState:
        # self.logger.info("test")
        # raise Exception("blah")
        # extra_info = self.extra_info.get_extra_info(packet)
        # self.game_state.update(packet, extra_info)
        self.game_state.update(packet)
        cur_time = packet.match_info.frame_num
        delta = cur_time - self.prev_time
        self.prev_time = cur_time

        self.ticks += delta

        if self.update_action and len(self.game_state.players) > self.index:
            self.update_action = False

            player = self.game_state.players[self.index]
            teammates = [
                p
                for p in self.game_state.players
                if p.team_num == self.team and p != player
            ]
            opponents = [p for p in self.game_state.players if p.team_num != self.team]

            self.game_state.players = [player] + teammates + opponents

            obs = self.obs_builder.build_obs(player, self.game_state, self.action)

            beta = self.beta
            if packet.match_info.match_phase == MatchPhase.Ended:
                # or not (packet.game_info.is_kickoff_pause or packet.game_info.is_round_active): Removed due to kickoff
                beta = 0  # Celebrate with random actions
            if (
                self.stochastic_kickoffs
                and packet.match_info.match_phase == MatchPhase.Kickoff
            ):
                beta = 0.5
            self.action, weights = self.agent.act(obs, beta)

            if self.render:
                positions = np.asarray(
                    [p.car_data.position for p in self.game_state.players]
                    + [self.game_state.ball.position]
                    + list(BOOST_LOCATIONS)
                )
                self.render_attention_weights(weights, positions)

        if self.ticks >= self.tick_skip - 1:
            self.update_controls(self.action)

        if self.ticks >= self.tick_skip:
            self.ticks = 0
            self.update_action = True

        if self.hardcoded_kickoffs:
            self.maybe_do_kickoff(packet, delta)

        return self.controls

    def maybe_do_kickoff(self, packet: GamePacket, ticks_elapsed):
        if packet.match_info.match_phase == MatchPhase.Countdown:
            ball = packet.balls[0]
            if self.kickoff_index >= 0:
                self.kickoff_index += round(ticks_elapsed)
            elif self.kickoff_index == -1:
                is_kickoff_taker = False
                ball_pos = np.array(
                    [
                        ball.physics.location.x,
                        ball.physics.location.y,
                    ]
                )
                positions = np.array(
                    [
                        [car.physics.location.x, car.physics.location.y]
                        for car in packet.players
                    ]
                )
                distances = np.linalg.norm(positions - ball_pos, axis=1)
                if abs(distances.min() - distances[self.index]) <= 10:
                    is_kickoff_taker = True
                    indices = np.argsort(distances)
                    for index in indices:
                        if (
                            abs(distances[index] - distances[self.index]) <= 10
                            and packet.players[index].team == self.team
                            and index != self.index
                        ):
                            if self.team == 0:
                                is_left = positions[index, 0] < positions[self.index, 0]
                            else:
                                is_left = positions[index, 0] > positions[self.index, 0]
                            if not is_left:
                                is_kickoff_taker = False  # Left goes

                self.kickoff_index = 0 if is_kickoff_taker else -2

            if (
                0 <= self.kickoff_index < len(KICKOFF_NUMPY)
                and ball.physics.location.y == 0
            ):
                action = KICKOFF_NUMPY[self.kickoff_index]
                self.action = action
                self.update_controls(self.action)
        else:
            self.kickoff_index = -1

    def update_controls(self, action):
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0
        self.controls.boost = action[6] > 0
        self.controls.handbrake = action[7] > 0


if __name__ == "__main__":
    # Connect to RLBot and run
    # Having the agent id here allows for easier development,
    # as otherwise the RLBOT_AGENT_ID environment variable must be set.
    Nexto("doublesize-nexto-v3").run()
