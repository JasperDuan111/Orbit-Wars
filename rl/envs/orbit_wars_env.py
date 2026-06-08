from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
from kaggle_environments.utils import structify
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
        configuration = {
            "episodeSteps": env_config.episode_steps,
            "actTimeout": env_config.act_timeout,
        }
        if env_config.seed is not None:
            configuration["seed"] = env_config.seed
        configuration["numPlayers"] = env_config.num_players
        self._env = make("orbit_wars", configuration=structify(configuration), debug=env_config.debug)
        self.num_players = env_config.num_players
        self.player_index = 0
        self._action_builder = ActionBuilder(action_config)
        self._last_actions = None
        self._last_source_ships = None
        # ── Reward calculator (stateful, per-episode tracking) ──
        self._reward_calc = RewardCalculator(reward_config)
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
        try:
            self._env.reset(num_agents=self.num_players)
        except TypeError:
            self._env.reset()
        obs = self._get_obs(self.player_index)
        player_id = _get_field(obs, "player", 0)
        self._reward_calc.reset(obs, player_id)
        self._last_actions, self._last_source_ships = self._action_builder.build(obs)
        return obs

    def step(self, action_indices, opponent_actions=None,
             action_templates_override=None, source_ships_override=None):
        """Step the environment.

        Args:
            action_indices: Discrete action indices for the main agent.
            opponent_actions: Optional dict mapping opp_idx_in_list -> action_list.
                When provided, skips calling opponent.act() individually and
                uses the pre-computed actions (enables batched GPU inference).
            action_templates_override: Optional — use these templates instead of
                stored _last_actions (for learned source selection).
            source_ships_override: Optional — use these ships instead of
                stored _last_source_ships.
        """
        obs = self._get_obs(self.player_index)
        player_id = _get_field(obs, "player", 0)
        templates = action_templates_override if action_templates_override is not None else self._last_actions
        ships = source_ships_override if source_ships_override is not None else self._last_source_ships
        my_action = self._action_builder.decode(
            action_indices,
            templates,
            ships,
            self.max_launches_per_source,
        )
        my_action, invalid_count = self._sanitize_action(my_action, obs, player_id)

        actions = [None] * self.num_players
        actions[self.player_index] = my_action
        for opp_idx in range(self.num_players):
            if opp_idx == self.player_index:
                continue
            opp_idx_in_list = opp_idx - 1 if opp_idx > self.player_index else opp_idx
            if opponent_actions is not None and opp_idx_in_list in opponent_actions:
                opp_action = opponent_actions[opp_idx_in_list]
            elif opp_idx_in_list < len(self._opponents):
                opp_obs = self._get_obs(opp_idx)
                opp_action = self._opponents[opp_idx_in_list].act(opp_obs)
            else:
                opp_action = []
            actions[opp_idx] = opp_action if opp_action else []
        self._env.step(actions)

        obs = self._get_obs(self.player_index)
        my_total, enemy_total, diff, reward = self._compute_reward(obs, player_id)

        self._last_actions, self._last_source_ships = self._action_builder.build(obs)
        info = {
            "my_total": my_total,
            "enemy_total": enemy_total,
            "diff": diff,
            "invalid_count": invalid_count,
        }
        if invalid_count > 0:
            reward -= self._reward_calc.invalid_action_penalty * invalid_count
        return obs, reward, self._is_done(), info

    def _compute_reward(self, obs, player_id):
        """Delegate to RewardCalculator.  Returns (my_total, enemy_total, diff, reward)."""
        return self._reward_calc.compute(obs, player_id, self._is_done())

    def _get_obs(self, index):
        return self._env.state[index]["observation"]

    def _is_done(self):
        if hasattr(self._env, "done"):
            return bool(self._env.done)
        return all(state.status != "ACTIVE" for state in self._env.state)

    def _sanitize_action(self, action, obs, player_id):
        if not action:
            return [], 0
        if not isinstance(action, list):
            return [], len(action) if hasattr(action, '__len__') else 1

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

        return clean, invalid_count
