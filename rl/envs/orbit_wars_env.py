from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
from kaggle_environments.utils import structify
from typing import Any
import random
from ..action import ActionBuilder
from ..config import ActionSpaceConfig, DEFAULT_CONFIG, EnvConfig, RewardConfig
from ..opponents import NearestPlanetOpponent
from ..rewards import RewardCalculator


def _get_field(obs, name, default=None):
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


class OrbitWarsSelfPlayEnv:
    def __init__(
        self,
        opponent=None,
        *,
        env_config: EnvConfig = DEFAULT_CONFIG.env,
        reward_config: RewardConfig = DEFAULT_CONFIG.reward,
        action_config: ActionSpaceConfig = DEFAULT_CONFIG.action,
    ):
        self._episode_steps = env_config.episode_steps
        self._act_timeout = env_config.act_timeout
        self._debug = env_config.debug
        self._next_seed = env_config.seed if env_config.seed is not None else 42
        self.num_players = env_config.num_players
        self._env = None  # created on first reset()
        self.player_index = 0
        self._action_builder = ActionBuilder(action_config)
        self._last_actions = None
        self._last_source_ships = None
        self._reward_calc = RewardCalculator(reward_config, env_config)
        self.max_launches_per_source = action_config.max_launches_per_source

        opponent = opponent or NearestPlanetOpponent()
        if not isinstance(opponent, list):
            opponent = [opponent] * (self.num_players - 1)
        self._opponents = opponent

    @property
    def last_actions(self):
        return self._last_actions

    @property
    def last_source_ships(self):
        return self._last_source_ships

    def get_opponents_data(self):
        """Returns list of (opp_idx_in_list, opponent_object, opponent_obs).

        Used for batched opponent inference. Call before step() to collect
        observations for all opponents, batch their GPU inference, and pass
        the results back via step(opponent_actions=...).
        """
        result = []
        for opp_idx in range(self.num_players):
            if opp_idx == self.player_index:
                continue
            opp_idx_in_list = opp_idx - 1 if opp_idx > self.player_index else opp_idx
            if opp_idx_in_list < len(self._opponents):
                result.append(
                    (opp_idx_in_list, self._opponents[opp_idx_in_list], self._get_obs(opp_idx))
                )
        return result

    def set_opponent(self, opponent, index=None):
        if index is not None:
            while len(self._opponents) <= index:
                self._opponents.append(NearestPlanetOpponent())
            self._opponents[index] = opponent or NearestPlanetOpponent()
        else:
            opponent = opponent or NearestPlanetOpponent()
            self._opponents = [opponent] * (self.num_players - 1)

    def reset(self):
        # Advance seed so each episode gets a different planet layout.
        self.player_index = random.choice([0, 1])
        self._next_seed += 10
        self._env = make(
            "orbit_wars",
            configuration=structify({
                "episodeSteps": self._episode_steps,
                "actTimeout": self._act_timeout,
                "seed": self._next_seed,
                "numPlayers": self.num_players,
            }),
            debug=self._debug,
        )
        obs = self._get_obs(self.player_index)
        player_id = _get_field(obs, "player", 0)

        # Skip degenerate seeds
        planets = [Planet(*p) for p in obs.get("planets", [])]
        my_ships = sum(p.ships for p in planets if p.owner == player_id)
        n_owned = sum(1 for p in planets if p.owner == player_id)
        while (my_ships == 0 or n_owned == 0):
            self._next_seed += 1
            self._env = make(
                "orbit_wars",
                configuration=structify({
                    "episodeSteps": self._episode_steps,
                    "actTimeout": self._act_timeout,
                    "seed": self._next_seed,
                    "numPlayers": self.num_players,
                }),
                debug=self._debug,
            )
            obs = self._get_obs(self.player_index)
            player_id = _get_field(obs, "player", 0)
            planets = [Planet(*p) for p in obs.get("planets", [])]
            my_ships = sum(p.ships for p in planets if p.owner == player_id)
            n_owned = sum(1 for p in planets if p.owner == player_id)

        self._reward_calc.reset(obs, player_id)
        self._last_actions = None
        self._last_source_ships = None
        return obs

    def step(self, only_economy, opponent_actions=None, my_action_override=None, update_count=0):
        """Step the environment.

        Args:
            opponent_actions: Optional dict mapping opp_idx_in_list -> action_list.
            my_action_override: Pre-decoded moves list.
        """
        obs = self._get_obs(self.player_index)
        player_id = _get_field(obs, "player", 0)
        my_action = my_action_override if my_action_override is not None else []
        my_action = my_action or []
        my_action, invalid_count, total_ships, per_launch = self._sanitize_action(my_action, obs, player_id)

        actions: list[Any] = [None] * self.num_players
        actions[self.player_index] = my_action
        for opp_idx in range(self.num_players):
            if opp_idx == self.player_index:
                continue
            opp_idx_in_list = opp_idx - 1 if opp_idx > self.player_index else opp_idx
            if only_economy:
                opp_action = []
            elif opponent_actions is not None and opp_idx_in_list in opponent_actions:
                opp_action = opponent_actions[opp_idx_in_list]
            elif opp_idx_in_list < len(self._opponents):
                opp_obs = self._get_obs(opp_idx)
                opp_action = self._opponents[opp_idx_in_list].act(opp_obs)
            else:
                opp_action = []
            actions[opp_idx] = opp_action if opp_action else []
        self._env.step(actions)

        obs = self._get_obs(self.player_index)
        my_total, enemy_total, diff, reward = self._compute_reward(
            obs, player_id, update_count, total_ships, per_launch)

        info = {
            "my_total": my_total,
            "enemy_total": enemy_total,
            "diff": diff,
            "invalid_count": invalid_count,
        }
        return obs, reward, self._is_done(), info

    def _compute_reward(self, obs, player_id, update_count, total_ships=0, per_launch=None):
        """Delegate to RewardCalculator."""
        return self._reward_calc.compute(obs, player_id, self._is_done(),
                                         update_count, total_ships, per_launch)

    def _get_obs(self, index):
        return self._env.state[index]["observation"]

    def _is_done(self):
        if hasattr(self._env, "done"):
            return bool(self._env.done)
        return all(_get_field(state, "status") != "ACTIVE" for state in self._env.state)

    def _sanitize_action(self, action, obs, player_id):
        if not action:
            return [], 0, 0, []
        if not isinstance(action, list):
            return [], len(action) if hasattr(action, '__len__') else 1, 0, []

        raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        planets = [Planet(*p) for p in raw_planets]
        ships_by_id = {p.id: p.ships for p in planets if p.owner == player_id}
        used = {}
        invalid_count = 0

        clean = []
        for move in action:
            if not isinstance(move, (list, tuple)) or len(move) != 3:
                invalid_count += 1
                continue
            from_id, angle, ships = move
            try:
                from_id = int(from_id)
                ships = int(ships)
            except (TypeError, ValueError):
                invalid_count += 1
                continue
            if from_id not in ships_by_id:
                invalid_count += 1
                continue
            remaining = ships_by_id[from_id] - used.get(from_id, 0)
            if ships < 1 or ships > remaining:
                invalid_count += 1
                continue
            used[from_id] = used.get(from_id, 0) + ships
            clean.append([from_id, float(angle), ships])

        total_ships = sum(m[2] for m in clean)
        per_launch = [m[2] for m in clean]
        return clean, invalid_count, total_ships, per_launch
