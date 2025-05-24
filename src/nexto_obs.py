from collections import Counter
from typing import Any

import numpy as np
from rlgym_compat import V1GameState as GameState
from rlgym_compat import V1PlayerData as PlayerData
from rlgym_compat.common_values import BLUE_TEAM, ORANGE_TEAM

MAX_REL_POS_DIFF_LEN = 5.5
BOOST_LOCATIONS = (
    (-7168.0, -9216.0, 73.0),
    (-4284.0, -9216.0, 70.0),
    (-2898.0, -9216.0, 70.0),
    (0.0, -9216.0, 70.0),
    (2898.0, -9216.0, 70.0),
    (4284.0, -9216.0, 70.0),
    (7168.0, -9216.0, 73.0),
    (-1792.0, -8836.0, 70.0),
    (1792.0, -8836.0, 70.0),
    (-5888.0, -7768.0, 70.0),
    (-3675.0, -7768.0, 70.0),
    (-1792.0, -7768.0, 70.0),
    (1792.0, -7768.0, 70.0),
    (3675.0, -7768.0, 70.0),
    (5888.0, -7768.0, 70.0),
    (0.0, -7186.0, 70.0),
    (-7680.0, -6144.0, 70.0),
    (7680.0, -6144.0, 70.0),
    (0.0, -5776.0, 70.0),
    (-4398.0, -5720.0, 70.0),
    (-1792.0, -5720.0, 70.0),
    (1792.0, -5720.0, 70.0),
    (4398.0, -5720.0, 70.0),
    (-5888.0, -4696.0, 70.0),
    (5888.0, -4696.0, 70.0),
    (-2818.0, -4096.0, 73.0),
    (-1792.0, -4096.0, 70.0),
    (0.0, -4096.0, 70.0),
    (1792.0, -4096.0, 70.0),
    (2818.0, -4096.0, 73.0),
    (-940.0, -3308.0, 70.0),
    (940.0, -3308.0, 70.0),
    (-7680.0, -3072.0, 70.0),
    (7680.0, -3072.0, 70.0),
    (-3318.0, -2484.0, 70.0),
    (3318.0, -2484.0, 70.0),
    (-1792.0, -2288.0, 70.0),
    (0.0, -2288.0, 70.0),
    (1792.0, -2288.0, 70.0),
    (-5888.0, -1624.0, 70.0),
    (5888.0, -1624.0, 70.0),
    (-2048.0, -1036.0, 70.0),
    (2048.0, -1036.0, 70.0),
    (0.0, -1024.0, 70.0),
    (-7680.0, 0.0, 73.0),
    (-5120.0, 0.0, 70.0),
    (-3584.0, 0.0, 73.0),
    (-1024.0, 0.0, 70.0),
    (1024.0, 0.0, 70.0),
    (3584.0, 0.0, 73.0),
    (5120.0, 0.0, 70.0),
    (7680.0, 0.0, 73.0),
    (0.0, 1024.0, 70.0),
    (-2048.0, 1036.0, 70.0),
    (2048.0, 1036.0, 70.0),
    (-5888.0, 1624.0, 70.0),
    (5888.0, 1624.0, 70.0),
    (-1792.0, 2288.0, 70.0),
    (0.0, 2288.0, 70.0),
    (1792.0, 2288.0, 70.0),
    (-3318.0, 2484.0, 70.0),
    (3318.0, 2484.0, 70.0),
    (-7680.0, 3072.0, 70.0),
    (7680.0, 3072.0, 70.0),
    (-940.0, 3308.0, 70.0),
    (940.0, 3308.0, 70.0),
    (-2818.0, 4096.0, 73.0),
    (-1792.0, 4096.0, 70.0),
    (0.0, 4096.0, 70.0),
    (1792.0, 4096.0, 70.0),
    (2818.0, 4096.0, 73.0),
    (-5888.0, 4696.0, 70.0),
    (5888.0, 4696.0, 70.0),
    (-4398.0, 5720.0, 70.0),
    (-1792.0, 5720.0, 70.0),
    (1792.0, 5720.0, 70.0),
    (4398.0, 5720.0, 70.0),
    (0.0, 5776.0, 70.0),
    (-7680.0, 6144.0, 70.0),
    (7680.0, 6144.0, 70.0),
    (0.0, 7186.0, 70.0),
    (-5888.0, 7768.0, 70.0),
    (-3675.0, 7768.0, 70.0),
    (-1792.0, 7768.0, 70.0),
    (1792.0, 7768.0, 70.0),
    (3675.0, 7768.0, 70.0),
    (5888.0, 7768.0, 70.0),
    (-1792.0, 8836.0, 70.0),
    (1792.0, 8836.0, 70.0),
    (-7168.0, 9216.0, 73.0),
    (-4284.0, 9216.0, 70.0),
    (-2898.0, 9216.0, 70.0),
    (0.0, 9216.0, 70.0),
    (2898.0, 9216.0, 70.0),
    (4284.0, 9216.0, 70.0),
    (7168.0, 9216.0, 73.0),
)


def rotation_to_quaternion(m: np.ndarray) -> np.ndarray:
    trace = np.trace(m)
    q = np.zeros(4)

    if trace > 0:
        s = (trace + 1) ** 0.5
        q[0] = s * 0.5
        s = 0.5 / s
        q[1] = (m[2, 1] - m[1, 2]) * s
        q[2] = (m[0, 2] - m[2, 0]) * s
        q[3] = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
            s = (1 + m[0, 0] - m[1, 1] - m[2, 2]) ** 0.5
            inv_s = 0.5 / s
            q[1] = 0.5 * s
            q[2] = (m[1, 0] + m[0, 1]) * inv_s
            q[3] = (m[2, 0] + m[0, 2]) * inv_s
            q[0] = (m[2, 1] - m[1, 2]) * inv_s
        elif m[1, 1] > m[2, 2]:
            s = (1 + m[1, 1] - m[0, 0] - m[2, 2]) ** 0.5
            inv_s = 0.5 / s
            q[1] = (m[0, 1] + m[1, 0]) * inv_s
            q[2] = 0.5 * s
            q[3] = (m[1, 2] + m[2, 1]) * inv_s
            q[0] = (m[0, 2] - m[2, 0]) * inv_s
        else:
            s = (1 + m[2, 2] - m[0, 0] - m[1, 1]) ** 0.5
            inv_s = 0.5 / s
            q[1] = (m[0, 2] + m[2, 0]) * inv_s
            q[2] = (m[1, 2] + m[2, 1]) * inv_s
            q[3] = 0.5 * s
            q[0] = (m[1, 0] - m[0, 1]) * inv_s

    # q[[0, 1, 2, 3]] = q[[3, 0, 1, 2]]

    return -q


def encode_gamestate(state: GameState, should_print=False):
    state_vals = [0, state.blue_score, state.orange_score]
    state_vals += state.boost_pads.tolist()

    for bd in (state.ball, state.inverted_ball):
        state_vals += bd.position.tolist()
        state_vals += bd.linear_velocity.tolist()
        state_vals += bd.angular_velocity.tolist()

    for p in state.players:
        state_vals += [p.car_id, p.team_num]
        for cd in (p.car_data, p.inverted_car_data):
            state_vals += cd.position.tolist()
            state_vals += rotation_to_quaternion(cd.rotation_mtx()).tolist()
            state_vals += cd.linear_velocity.tolist()
            state_vals += cd.angular_velocity.tolist()
        state_vals += [
            0,
            0,
            0,
            0,
            0,
            p.is_demoed,
            p.on_ground,
            p.ball_touched,
            p.has_flip,
            p.boost_amount,
        ]
    return state_vals


class BatchedObsBuilder:
    def __init__(self):
        super().__init__()
        self.current_state = None
        self.current_obs = None

    def batched_build_obs(self, encoded_states: np.ndarray) -> Any:
        raise NotImplementedError

    def add_actions(self, obs: Any, previous_actions: np.ndarray, player_index=None):
        # Modify current obs to include action
        # player_index=None means actions for all players should be provided
        raise NotImplementedError

    def _reset(self, initial_state: GameState):
        raise NotImplementedError

    def reset(self, initial_state: GameState):
        self.current_state = False
        self.current_obs = None
        self._reset(initial_state)

    def build_obs(
        self,
        player: PlayerData,
        state: GameState,
        previous_action: np.ndarray,
        should_print=False,
    ) -> Any:
        # if state != self.current_state:
        self.current_obs = self.batched_build_obs(
            np.expand_dims(encode_gamestate(state, should_print=should_print), axis=0),
            should_print=should_print,
        )
        self.current_state = state

        for i, p in enumerate(state.players):
            if p == player:
                self.add_actions(self.current_obs, previous_action, i)
                return self.current_obs[i]


IS_SELF, IS_MATE, IS_OPP, IS_BALL, IS_BOOST = range(5)
POS = slice(5, 8)
LIN_VEL = slice(8, 11)
FW = slice(11, 14)
UP = slice(14, 17)
ANG_VEL = slice(17, 20)
BOOST, DEMO, ON_GROUND, HAS_FLIP = range(20, 24)
ACTIONS = range(24, 32)

BALL_STATE_LENGTH = 18
PLAYER_CAR_STATE_LENGTH = 13
PLAYER_TERTIARY_INFO_LENGTH = 10
PLAYER_INFO_LENGTH = 2 + 2 * PLAYER_CAR_STATE_LENGTH + PLAYER_TERTIARY_INFO_LENGTH


class NextoObsBuilder(BatchedObsBuilder):
    _invert = np.array([1] * 5 + [-1, -1, 1] * 5 + [1] * 4)
    _norm = np.array([1.0] * 5 + [2300] * 6 + [1] * 6 + [5.5] * 3 + [1] * 4)

    def __init__(self, field_info=None, n_players=None, tick_skip=8):
        super().__init__()
        self.n_players = n_players
        self.demo_timers = None
        self.boost_timers = None
        self.tick_skip = tick_skip
        if field_info is None:
            self._boost_locations = np.array(BOOST_LOCATIONS)
            self._boost_types = self._boost_locations[:, 2] > 72
        else:
            self._boost_locations = np.array(
                [
                    [bp.location.x, bp.location.y, bp.location.z]
                    for bp in field_info.boost_pads
                ]
            )
            self._boost_types = np.array(
                [bp.is_full_boost for bp in field_info.boost_pads]
            )

    def _reset(self, initial_state: GameState):
        self.demo_timers = np.zeros(len(initial_state.players))
        self.boost_timers = np.zeros(len(initial_state.boost_pads))

    @staticmethod
    def _quats_to_rot_mtx(quats: np.ndarray) -> np.ndarray:
        # From rlgym.utils.math.quat_to_rot_mtx
        w = -quats[:, 0]
        x = -quats[:, 1]
        y = -quats[:, 2]
        z = -quats[:, 3]

        theta = np.zeros((quats.shape[0], 3, 3))

        norm = np.einsum("fq,fq->f", quats, quats)

        sel = norm != 0

        w = w[sel]
        x = x[sel]
        y = y[sel]
        z = z[sel]

        s = 1.0 / norm[sel]

        # front direction
        theta[sel, 0, 0] = 1.0 - 2.0 * s * (y * y + z * z)
        theta[sel, 1, 0] = 2.0 * s * (x * y + z * w)
        theta[sel, 2, 0] = 2.0 * s * (x * z - y * w)

        # left direction
        theta[sel, 0, 1] = 2.0 * s * (x * y - z * w)
        theta[sel, 1, 1] = 1.0 - 2.0 * s * (x * x + z * z)
        theta[sel, 2, 1] = 2.0 * s * (y * z + x * w)

        # up direction
        theta[sel, 0, 2] = 2.0 * s * (x * z + y * w)
        theta[sel, 1, 2] = 2.0 * s * (y * z - x * w)
        theta[sel, 2, 2] = 1.0 - 2.0 * s * (x * x + y * y)

        return theta

    @staticmethod
    def convert_to_relative(q, kv, should_print=False):
        # kv[..., POS.start:LIN_VEL.stop] -= q[..., POS.start:LIN_VEL.stop]
        kv[..., POS] -= q[..., POS]
        forward = q[..., FW]
        theta = np.arctan2(forward[..., 0], forward[..., 1])
        theta = np.expand_dims(theta, axis=-1)
        ct = np.cos(theta)
        st = np.sin(theta)
        xs = kv[..., POS.start : ANG_VEL.stop : 3]
        ys = kv[..., POS.start + 1 : ANG_VEL.stop : 3]
        # Use temp variables to prevent modifying original array
        nx = ct * xs - st * ys
        ny = st * xs + ct * ys
        kv[..., POS.start : ANG_VEL.stop : 3] = nx  # x-components
        kv[..., POS.start + 1 : ANG_VEL.stop : 3] = ny  # y-components
        # modify relative components to be within distribution for what's possible on a standard map
        # This method uses a nonlinear scaling method to make sure everything stays within a certain distance away
        lens = np.linalg.norm(kv[..., POS], axis=-1)
        filtered_lens = lens[lens > 0.1]
        kv[lens > 0.1, POS] = (
            kv[lens > 0.1, POS]
            / np.stack([filtered_lens] * 3, axis=-1)
            * np.stack(
                [
                    (
                        MAX_REL_POS_DIFF_LEN
                        * filtered_lens
                        / (
                            MAX_REL_POS_DIFF_LEN * MAX_REL_POS_DIFF_LEN
                            + filtered_lens * filtered_lens
                        )
                    )
                ]
                * 3,
                axis=-1,
            )
        )

    def batched_build_obs(self, encoded_states: np.ndarray, should_print=False):
        ball_start_index = 3 + len(self._boost_locations)
        players_start_index = ball_start_index + BALL_STATE_LENGTH
        player_length = PLAYER_INFO_LENGTH

        n_players = (encoded_states.shape[1] - players_start_index) // player_length
        lim_players = n_players if self.n_players is None else self.n_players
        n_entities = lim_players + 1 + 96

        # SELECTORS
        sel_players = slice(0, lim_players)
        sel_ball = sel_players.stop
        sel_boosts = slice(sel_ball + 1, None)

        # MAIN ARRAYS
        q = np.zeros((n_players, encoded_states.shape[0], 1, 32))
        kv = np.zeros(
            (n_players, encoded_states.shape[0], n_entities, 24)
        )  # Keys and values are (mostly) shared
        m = np.zeros((n_players, encoded_states.shape[0], n_entities))  # Mask is shared

        # BALL
        kv[:, :, sel_ball, 3] = 1
        kv[:, :, sel_ball, np.r_[POS, LIN_VEL, ANG_VEL]] = encoded_states[
            :, ball_start_index : ball_start_index + 9
        ]

        # BOOSTS
        kv[:, :, sel_boosts, IS_BOOST] = 1
        kv[:, :, sel_boosts, POS] = self._boost_locations
        kv[:, :, sel_boosts, BOOST] = 0.12 + 0.88 * (self._boost_locations[:, 2] > 72)
        kv[:, :, sel_boosts, DEMO] = encoded_states[:, 3 : 3 + 96]  # FIXME boost timer

        # PLAYERS
        teams = encoded_states[0, players_start_index + 1 :: player_length]
        kv[:, :, :n_players, IS_MATE] = 1 - teams  # Default team is blue
        kv[:, :, :n_players, IS_OPP] = teams
        for i in range(n_players):
            encoded_player = encoded_states[
                :,
                players_start_index
                + i * player_length : players_start_index
                + (i + 1) * player_length,
            ]

            kv[i, :, i, IS_SELF] = 1
            kv[:, :, i, POS] = encoded_player[
                :, 2:5
            ]  # TODO constants for these indices
            kv[:, :, i, LIN_VEL] = encoded_player[:, 9:12]
            quats = encoded_player[:, 5:9]
            rot_mtx = self._quats_to_rot_mtx(quats)
            kv[:, :, i, FW] = rot_mtx[:, :, 0]
            kv[:, :, i, UP] = rot_mtx[:, :, 2]
            kv[:, :, i, ANG_VEL] = encoded_player[:, 12:15]
            kv[:, :, i, BOOST] = encoded_player[:, 37]
            kv[:, :, i, DEMO] = encoded_player[:, 33]  # FIXME demo timer
            kv[:, :, i, ON_GROUND] = encoded_player[:, 34]
            kv[:, :, i, HAS_FLIP] = encoded_player[:, 36]

        kv[teams == 1] *= self._invert
        kv[np.argwhere(teams == 1), ..., (IS_MATE, IS_OPP)] = kv[
            np.argwhere(teams == 1), ..., (IS_OPP, IS_MATE)
        ]  # Swap teams

        kv /= self._norm

        # kv[..., POS.start] = kv[..., POS.start] / 2
        # kv[..., POS.start + 1] = kv[..., POS.start + 1] / 2

        for i in range(n_players):
            q[i, :, 0, : kv.shape[-1]] = kv[i, :, i, :].copy()

        self.convert_to_relative(q, kv, should_print=should_print)
        # Convert absolute positions of query to be halved so that they are within distribution
        q[..., POS.start] = q[..., POS.start] / 2
        q[..., POS.start + 1] = q[..., POS.start + 1] / 2
        # kv[:, :, :, 5:11] -= q[:, :, :, 5:11]

        # MASK
        m[:, :, n_players:lim_players] = 1

        return [(q[i], kv[i], m[i]) for i in range(n_players)]

    def add_actions(self, obs: Any, previous_actions: np.ndarray, player_index=None):
        if player_index is None:
            for (q, kv, m), act in zip(obs, previous_actions):
                q[:, 0, ACTIONS] = act
        else:
            q, kv, m = obs[player_index]
            q[:, 0, ACTIONS] = previous_actions
