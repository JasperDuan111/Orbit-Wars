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
        self.opponent_index = 1 if num_players > 1 else 0
        self._opponent = opponent or NearestPlanetOpponent()
        self._action_builder = ActionBuilder()
        self._last_actions = None
        self._last_source_ships = None
        self._last_diff = None
        self.reward_scale = reward_scale
        self.invalid_action_penalty = invalid_action_penalty
        self.terminal_reward_scale = terminal_reward_scale
        self.max_launches_per_source = max_launches_per_source

    def set_opponent(self, opponent):
        self._opponent = opponent or NearestPlanetOpponent()

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
        my_action, invalid = self._sanitize_action(my_action, obs, player_id)

        opp_obs = self._get_obs(self.opponent_index)
        opp_action = self._opponent.act(opp_obs) if self._opponent else []
        if opp_action is None:
            opp_action = []
        self._env.step([my_action, opp_action])

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
            "invalid_action": invalid,
        }
        if invalid:
            reward -= self.invalid_action_penalty
        return obs, reward, done, info

    def _get_obs(self, index):
        return self._env.state[index].observation

    def _is_done(self):
        if hasattr(self._env, "done"):
            return bool(self._env.done)
        return all(state.status != "ACTIVE" for state in self._env.state)

    def _sanitize_action(self, action, obs, player_id):
        if not action:
            return [], False
        if not isinstance(action, list):
            return [], True

        raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        planets = [Planet(*p) for p in raw_planets]
        ships_by_id = {p.id: p.ships for p in planets if p.owner == player_id}
        used = {}

        clean = []
        for move in action:
            if not isinstance(move, (list, tuple)) or len(move) != 3:
                return [], True
            from_id, angle, ships = move
            try:
                from_id = int(from_id)
                ships = int(ships)
            except (TypeError, ValueError):
                return [], True
            if from_id not in ships_by_id:
                return [], True
            remaining = ships_by_id[from_id] - used.get(from_id, 0)
            if ships < 1 or ships > remaining:
                return [], True
            used[from_id] = used.get(from_id, 0) + ships
            clean.append([from_id, float(angle), ships])

        return clean, False
