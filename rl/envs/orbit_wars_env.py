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
        # ── Reward config ──
        self.economic_scale = reward_config.economic_scale
        self.territory_scale = reward_config.territory_scale
        self.combat_efficiency_scale = reward_config.combat_efficiency_scale
        self.idle_penalty_scale = reward_config.idle_penalty_scale
        self.production_weight = reward_config.production_weight
        self.idle_threshold = reward_config.idle_threshold
        self.early_game_steps = reward_config.early_game_steps
        self.mid_game_steps = reward_config.mid_game_steps
        self.survival_reward_early = reward_config.survival_reward_early
        self.survival_reward_mid = reward_config.survival_reward_mid
        self.survival_reward_late = reward_config.survival_reward_late
        self.terminal_win_scale = reward_config.terminal_win_scale
        self.terminal_lose_scale = reward_config.terminal_lose_scale
        self.invalid_action_penalty = reward_config.invalid_action_penalty
        self.max_launches_per_source = action_config.max_launches_per_source
        # ── Tracking state across steps ──
        self._last_planet_owners = None   # dict[int,int]  planet_id → owner
        self._last_econ_ratio = None       # float  my_prod / (my_prod + eny_prod + 1)
        self._last_my_total = None         # int  my total ships (planets + fleets)
        self._last_enemy_total = None      # int  enemy total ships

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
        self._last_my_total = my_total
        self._last_enemy_total = enemy_total
        self._last_planet_owners = _planet_owner_map(obs)
        my_prod, eny_prod = _planet_production_totals(obs, player_id)
        self._last_econ_ratio = my_prod / (my_prod + eny_prod + 1.0)
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
            reward -= self.invalid_action_penalty * invalid_count
        return obs, reward, self._is_done(), info

    def _compute_reward(self, obs, player_id):
        """Six-dimension reward (all components ∈ [−1, +1] per step).
        Returns: (my_total, enemy_total, diff, reward)
        """
        step = _get_field(obs, "step", 0)
        my_total, enemy_total = ship_totals(obs, player_id)
        diff = my_total - enemy_total
        reward = 0.0

        my_prod, eny_prod = _planet_production_totals(obs, player_id)
        cur_econ_ratio = my_prod / (my_prod + eny_prod + 1.0)
        if self._last_econ_ratio is not None:
            reward += (cur_econ_ratio - self._last_econ_ratio) * self.economic_scale

        cur_owners = _planet_owner_map(obs)
        if self._last_planet_owners is not None:
            for pid, new_owner in cur_owners.items():
                old_owner = self._last_planet_owners.get(pid)
                if old_owner is not None and old_owner != new_owner:
                    production = _planet_production(obs, pid)
                    quality = 1.0 + production * self.production_weight
                    if new_owner == player_id:
                        reward += quality * self.territory_scale
                    elif old_owner == player_id:
                        reward -= quality * self.territory_scale

        if self._last_my_total is not None and self._last_enemy_total is not None:
            my_production_now = my_prod  # ships spawned this step
            my_combat_loss = max(0.0, self._last_my_total + my_production_now - my_total)
            eny_combat_loss = max(0.0, self._last_enemy_total + eny_prod - enemy_total)
            total_loss = my_combat_loss + eny_combat_loss + 1e-8
            efficiency = eny_combat_loss / total_loss - 0.5  # [-0.5, +0.5]
            if my_combat_loss + eny_combat_loss > 1:
                reward += efficiency * self.combat_efficiency_scale

        idle_ratio = _idle_ship_ratio(obs, player_id)
        if idle_ratio > self.idle_threshold:
            reward -= (idle_ratio - self.idle_threshold) * self.idle_penalty_scale

        if not self._is_done():
            if step < self.early_game_steps:
                reward += self.survival_reward_early
            elif step < self.mid_game_steps:
                reward += self.survival_reward_mid
            else:
                reward += self.survival_reward_late

        if self._is_done():
            if diff > 0:
                reward += self.terminal_win_scale
            else:
                reward += self.terminal_lose_scale

        # ── Update tracking state ──
        self._last_my_total = my_total
        self._last_enemy_total = enemy_total
        self._last_econ_ratio = cur_econ_ratio
        self._last_planet_owners = cur_owners

        return my_total, enemy_total, diff, reward

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


def _parse_planets_list(obs):
    raw = _get_field(obs, "planets", [])
    return [Planet(*p) for p in raw]


def _planet_owner_map(obs):
    """Return {planet_id: owner} for all planets in observation."""
    planets = _parse_planets_list(obs)
    return {p.id: p.owner for p in planets}


def _planet_production_totals(obs, player_id):
    """Return (my_total_production, enemy_total_production)."""
    planets = _parse_planets_list(obs)
    my_prod = sum(p.production for p in planets if p.owner == player_id)
    eny_prod = sum(p.production for p in planets if p.owner not in (-1, player_id))
    return my_prod, eny_prod


def _planet_production(obs, planet_id):
    """Return the production rate of a specific planet."""
    planets = _parse_planets_list(obs)
    for p in planets:
        if p.id == planet_id:
            return p.production
    return 0


def _idle_ship_ratio(obs, player_id):
    """Fraction of total ships that are sitting on owned planets (not in flight)."""
    planets = _parse_planets_list(obs)
    ships_on_planets = sum(p.ships for p in planets if p.owner == player_id)
    my_total, _ = ship_totals(obs, player_id)
    if my_total < 1:
        return 0.0
    return ships_on_planets / my_total
