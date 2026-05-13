from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
from ..action import ActionBuilder
from ..config import MAX_LAUNCHES_PER_SOURCE
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
        num_players=2,
        episode_steps=500,
        act_timeout=1,
        seed=None,
        reward_scale=0.01,
        invalid_action_penalty=0.05,
        terminal_reward_scale=0.01,
        max_launches_per_source=MAX_LAUNCHES_PER_SOURCE,
        debug=False,
    ):
        configuration = {"episodeSteps": episode_steps, "actTimeout": act_timeout}
        if seed is not None:
            configuration["seed"] = seed
        configuration["numPlayers"] = num_players
        self._env = make("orbit_wars", configuration=configuration, debug=debug)
        self.num_players = num_players
        self.player_index = 0
        self._action_builder = ActionBuilder()
        self._last_actions = None
        self._last_source_ships = None
        self._last_diff = None
        self.reward_scale = reward_scale
        self.invalid_action_penalty = invalid_action_penalty
        self.terminal_reward_scale = terminal_reward_scale
        self.max_launches_per_source = max_launches_per_source

        opponent = opponent or NearestPlanetOpponent()
        if not isinstance(opponent, list):
            opponent = [opponent] * (num_players - 1)
        self._opponents = opponent

    @property
    def last_actions(self):
        return self._last_actions

    @property
    def last_source_ships(self):
        return self._last_source_ships

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
        self._last_actions, self._last_source_ships = self._action_builder.build(obs)
        return obs

    def step(self, action_indices):
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
            opp_obs = self._get_obs(opp_idx)
            opp_idx_in_list = opp_idx - 1 if opp_idx > self.player_index else opp_idx
            if opp_idx_in_list < len(self._opponents):
                opp_action = self._opponents[opp_idx_in_list].act(opp_obs)
                actions[opp_idx] = opp_action if opp_action else []
            else:
                actions[opp_idx] = []
        self._env.step(actions)

        obs = self._get_obs(self.player_index)
        my_total, enemy_total = ship_totals(obs, player_id)
        diff = my_total - enemy_total
        reward = (diff - self._last_diff) * self.reward_scale if self._last_diff is not None else 0.0
        done = self._is_done()
        if done:
            reward += diff * self.terminal_reward_scale
        self._last_diff = diff

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

    def _get_obs(self, index):
        return self._env.state[index].observation

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
