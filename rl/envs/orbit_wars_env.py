from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
from kaggle_environments.utils import structify
from ..action import ActionBuilder
from ..config import ActionSpaceConfig, DEFAULT_CONFIG, EnvConfig, RewardConfig
from ..obs import ship_totals
from ..opponents import NearestPlanetOpponent


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
        self._last_diff = None
        self._last_planet_diff = None
        self._last_production_diff = None
        self.reward_scale = reward_config.reward_scale
        self.invalid_action_penalty = reward_config.invalid_action_penalty
        self.terminal_reward_scale = reward_config.terminal_reward_scale
        self.win_reward = reward_config.win_reward
        self.lose_penalty = reward_config.lose_penalty
        self.planet_control_scale = reward_config.planet_control_scale
        self.production_scale = reward_config.production_scale
        self.survival_reward = reward_config.survival_reward
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
        my_total, enemy_total = ship_totals(obs, player_id)
        self._last_diff = my_total - enemy_total
        planet_diff, production_diff = self._planet_and_production_diff(obs, player_id)
        self._last_planet_diff = planet_diff
        self._last_production_diff = production_diff
        self._last_actions, self._last_source_ships = self._action_builder.build(obs)
        return obs

    def step(self, action_indices, opponent_actions=None):
        """Step the environment.

        Args:
            action_indices: Discrete action indices for the main agent.
            opponent_actions: Optional dict mapping opp_idx_in_list -> action_list.
                When provided, skips calling opponent.act() individually and
                uses the pre-computed actions (enables batched GPU inference).
        """
        obs = self._get_obs(self.player_index)
        player_id = _get_field(obs, "player", 0)
        my_action = self._action_builder.decode(
            action_indices,
            self._last_actions,
            self._last_source_ships,
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
        my_total, enemy_total, diff, done, reward = self._compute_reward(obs, player_id)

        self._last_actions, self._last_source_ships = self._action_builder.build(obs)
        info = {
            "my_total": my_total,
            "enemy_total": enemy_total,
            "diff": diff,
            "invalid_count": invalid_count,
        }
        if invalid_count > 0:
            reward -= self.invalid_action_penalty * invalid_count
        return obs, reward, done, info

    def _compute_reward(self, obs, player_id):
        my_total, enemy_total = ship_totals(obs, player_id)
        diff = my_total - enemy_total
        reward = (diff - self._last_diff) * self.reward_scale if self._last_diff is not None else 0.0
        planet_diff, production_diff = self._planet_and_production_diff(obs, player_id)
        if self._last_planet_diff is not None:
            reward += (planet_diff - self._last_planet_diff) * self.planet_control_scale
        if self._last_production_diff is not None:
            reward += (production_diff - self._last_production_diff) * self.production_scale
        done = self._is_done()
        if done:
            reward += diff * self.terminal_reward_scale
            if diff > 0:
                reward += self.win_reward
            elif diff < 0:
                reward += self.lose_penalty
        elif self.survival_reward:
            reward += self.survival_reward
        self._last_diff = diff
        self._last_planet_diff = planet_diff
        self._last_production_diff = production_diff
        return my_total, enemy_total, diff, done, reward

    def _planet_and_production_diff(self, obs, player_id):
        raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        planets = [Planet(*p) for p in raw_planets]
        my_planets = [p for p in planets if p.owner == player_id]
        enemy_planets = [p for p in planets if p.owner not in (-1, player_id)]
        planet_diff = len(my_planets) - len(enemy_planets)
        my_production = sum(p.production for p in my_planets)
        enemy_production = sum(p.production for p in enemy_planets)
        production_diff = my_production - enemy_production
        return planet_diff, production_diff

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
