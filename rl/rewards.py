"""Reward function for Orbit Wars PPO training."""

import math
from typing import Dict, Optional, Tuple

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from .config import EnvConfig, RewardConfig, DEFAULT_CONFIG
from .obs import ship_totals


# ======================================================================
# Shared observation helpers
# ======================================================================

def _get_field(obs, name, default=None):
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def _parse_planets_list(obs):
    raw = _get_field(obs, "planets", [])
    return [Planet(*p) for p in raw]


# ------------------------------------------------------------------
# Public helpers  (also used by the env for debugging)
# ------------------------------------------------------------------

def planet_owner_map(obs) -> Dict[int, int]:
    """Return {planet_id: owner} for all planets in the observation."""
    return {p.id: p.owner for p in _parse_planets_list(obs)}


def planet_production_totals(obs, player_id) -> Tuple[float, float]:
    """Return (my_total_production, enemy_total_production)."""
    planets = _parse_planets_list(obs)
    my_prod = sum(p.production for p in planets if p.owner == player_id)
    eny_prod = sum(p.production for p in planets if p.owner not in (-1, player_id))
    return my_prod, eny_prod


def total_planet_production(obs) -> float:
    """Return total production for all planets on the board."""
    return sum(p.production for p in _parse_planets_list(obs))


def _prod_log_value(production: float) -> float:
    if production <= 0:
        return 0.0
    return production * math.log(production)


def planet_production(obs, planet_id) -> float:
    """Return the production rate of a specific planet."""
    for p in _parse_planets_list(obs):
        if p.id == planet_id:
            return p.production
    return 0.0


def planet_counts(obs, player_id) -> Tuple[int, int]:
    """Return (my_planet_count, enemy_planet_count)."""
    planets = _parse_planets_list(obs)
    my = sum(1 for p in planets if p.owner == player_id)
    eny = sum(1 for p in planets if p.owner not in (-1, player_id))
    return my, eny


def idle_ship_ratio(obs, player_id) -> float:
    """Fraction of total ships sitting on owned planets (not in flight)."""
    planets = _parse_planets_list(obs)
    ships_on_planets = sum(p.ships for p in planets if p.owner == player_id)
    my_total, _ = ship_totals(obs, player_id)
    if my_total < 1:
        return 0.0
    return ships_on_planets / my_total


# ======================================================================
# RewardCalculator
# ======================================================================

class RewardCalculator:
    """Multi-dimension reward with per-episode tracking state.

    Usage::

        calc = RewardCalculator(reward_config)
        calc.reset(obs, player_id)
        my_total, enemy_total, diff, r = calc.compute(obs, player_id, done)
    """

    def __init__(self, reward_config: RewardConfig = DEFAULT_CONFIG.reward, env_config: EnvConfig = DEFAULT_CONFIG.env):
        # ── Scales ──
        self.fleet_advantage_scale: float = reward_config.fleet_advantage_scale
        self.production_advantage_scale: float = reward_config.production_advantage_scale
        self.terminal_economy_scale: float = reward_config.terminal_economy_scale
        self.planet_count_scale: float = reward_config.planet_count_scale
        self.territory_scale: float = reward_config.territory_scale
        self.production_weight: float = reward_config.production_weight
        self.enemy_capture_bonus: float = reward_config.enemy_capture_bonus
        self.territory_loss_penalty: float = reward_config.territory_loss_penalty
        self.no_action_penalty_scale: float = reward_config.no_action_penalty_scale
        self.launch_bonus_scale: float = reward_config.launch_bonus_scale
        self.defense_success_scale: float = getattr(reward_config, 'defense_success_scale', 0.01)
        self.fleet_arrival_scale: float = getattr(reward_config, 'fleet_arrival_scale', 0.005)
        self.terminal_win_scale: float = reward_config.terminal_win_scale
        self.terminal_lose_scale: float = reward_config.terminal_lose_scale
        self.invalid_action_penalty: float = reward_config.invalid_action_penalty
        self.out_of_boundary_penalty_scale: float = reward_config.out_of_boundary_penalty_scale
        self.suicide_penalty_scale: float = reward_config.suicide_penalty_scale
        self.only_economy = reward_config.only_economy
        self.no_action_grace_steps: int = reward_config.no_action_grace_steps

        # ── Per-episode tracking ──
        self._last_planet_owners: Optional[Dict[int, int]] = None
        self._idle_steps: int = 0

        self.episode_steps = env_config.episode_steps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, obs, player_id: int):
        """(Re-)initialise tracking state for a new episode."""
        self._last_planet_owners = planet_owner_map(obs)
        self._idle_steps = 0

    def compute(
        self, obs, player_id: int, is_done: bool, update_cont: int,
        total_ships: int = 0,
    ) -> Tuple[float, float, float, float]:
        """Compute the per-step reward.

        Returns
        -------
        my_total : float
        enemy_total : float
        diff : float      ``my_total - enemy_total``
        reward : float
        """
        my_total, enemy_total = ship_totals(obs, player_id)
        diff = my_total - enemy_total
        rewards_helper = obs["rewards_helper"]
        out_of_boundary = rewards_helper["out_of_boundary"]
        suicide = rewards_helper["suicide"]
        defense_ships = rewards_helper.get("defense_ships", {}).get(player_id, 0)
        arrival_ships = rewards_helper.get("arrival_ships", {}).get(player_id, 0)

        reward = 0.0
        if self.only_economy:
            # ── Economy-only: learn to grow fleet & production ──
            reward += self._launch_bonus(total_ships)
            reward += self._economy_fleet_advantage(my_total, enemy_total)
            reward += self._economy_production_advantage(obs, player_id)
            reward += self._wrong_action_penalty(out_of_boundary, suicide)
            reward += self._no_action_penalty(total_ships)
        else:
            # ── Combat: map control + territory events + terminal ──
            reward += self._launch_bonus(obs, total_ships)
            reward += self._economy_fleet_advantage(my_total, enemy_total)
            reward += self._planet_count_advantage(obs, player_id)
            reward += self._economy_production_advantage(obs, player_id)
            reward += self._territory_change(obs, player_id)
            reward += self._defense_success(defense_ships)
            reward += self._fleet_arrival(arrival_ships)
            reward += self._terminal_reward(diff, is_done)
            reward += self._wrong_action_penalty(out_of_boundary, suicide)
            reward += self._no_action_penalty(total_ships)
    

        # ── Persist ──
        self._last_planet_owners = planet_owner_map(obs)

        return my_total, enemy_total, diff, reward

    # ------------------------------------------------------------------
    #  Reward components
    # ------------------------------------------------------------------

    def _economy_fleet_advantage(self, my_total: float, enemy_total: float) -> float:
        return (my_total - enemy_total) * self.fleet_advantage_scale

    def _wrong_action_penalty(self, out_of_boundary: int, suicide: int):
        return - (
            out_of_boundary * self.out_of_boundary_penalty_scale
            + suicide * self.suicide_penalty_scale
        )

    def _economy_production_advantage(self, obs, player_id: int) -> float:
        my_prod, enemy_prod = planet_production_totals(obs, player_id)
        advantage = _prod_log_value(my_prod) - _prod_log_value(enemy_prod)
        return advantage * self.production_advantage_scale

    def _economy_terminal(self, obs, player_id: int, my_total: float, is_done: bool) -> float:
        if not is_done:
            return 0.0
        total_prod = total_planet_production(obs)
        if total_prod <= 0:
            return 0.0
        return ((my_total / total_prod) - 0.5) * self.terminal_economy_scale

    # 1 ── Planet-count advantage  (per step, dense) ─────────────────
    def _planet_count_advantage(self, obs, player_id: int) -> float:
        """+scale per extra planet owned over the enemy.

        If we own 5 planets and the enemy owns 2 → +3 × scale per step.
        This directly rewards map control, the true measure of strategic
        success.
        """
        my_planets, eny_planets = planet_counts(obs, player_id)
        advantage = my_planets - eny_planets
        return advantage * self.planet_count_scale

    # 2 ── Territory change  (event-driven, sparse) ──────────────────
    def _territory_change(self, obs, player_id: int) -> float:
        """Reward planet-ownership transitions.

        - Enemy  → me:  + quality × territory_scale × enemy_capture_bonus   (highest — requires mass)
        - Neutral → me: + quality × territory_scale                          (baseline — free planets)
        - Me → enemy:    − quality × territory_scale × territory_loss_penalty (larger loss penalty)
        - Me → neutral:  − quality × territory_scale (small loss — may be strategic)
        """
        prev_owners = self._last_planet_owners
        if prev_owners is None:
            return 0.0

        cur_owners = planet_owner_map(obs)
        reward = 0.0

        for pid, new_owner in cur_owners.items():
            old_owner = prev_owners.get(pid)
            if old_owner is None or old_owner == new_owner:
                continue

            prod = planet_production(obs, pid)
            quality = 1.0 + prod * self.production_weight
            base = quality * self.territory_scale

            if new_owner == player_id:
                if old_owner != -1:                      # enemy → me (hardest, biggest reward)
                    reward += base * self.enemy_capture_bonus
                else:                                     # neutral → me
                    reward += base
            elif old_owner == player_id:
                if new_owner != -1:                      # me → enemy (worst loss)
                    reward -= base * self.territory_loss_penalty
                else:                                     # me → neutral
                    reward -= base

        return reward

    def _no_action_penalty(self, total_ships: int) -> float:
        """Penalise prolonged inaction — escalating.

        After *no_action_grace_steps* consecutive idle steps the penalty
        grows with idle duration, so the model cannot afford to sit still
        indefinitely.
        """
        if total_ships > 0:
            self._idle_steps = 0
            return 0.0
        self._idle_steps += 1
        if self._idle_steps <= self.no_action_grace_steps:
            return 0.0
        excess = self._idle_steps - self.no_action_grace_steps
        return -self.no_action_penalty_scale * (1.0 + 0.1 * excess)

    def _launch_bonus(self, obs, total_ships: int) -> float:
        """Per-ship launch bonus with a minimum-ships gate.

        Fleets smaller than 10 ships earn nothing — this discourages the
        "many tiny launches" degenerate strategy where the model fires
        2–4 ship fleets that arrive but lose every combat.

        Not-launching (0 ships) is NOT penalised here — ``_no_action_penalty``
        handles prolonged inaction with its own grace + escalating mechanism.
        The penalty gradient for 1–9 ships is deliberately mild so the model
        can occasionally risk a small fleet without catastrophic punishment.
        """
        if total_ships == 0:
            return 0.0                      # idle penalty handles this
        if obs["step"] <= 50 and total_ships < 10:
            return -0.05                    # early game: gentle nudge
        if total_ships < 10:
            return -0.5 + 0.45 * (total_ships - 1) / 8.0   # 1船=-0.50, 5船=-0.28, 9船=-0.05
        return total_ships * self.launch_bonus_scale

    def _defense_success(self, defense_ships: int) -> float:
        """Reward for destroying enemy ships while defending own planets.

        Bridges the launch→capture credit-assignment gap: the model
        learns that fleets en route to enemy planets are valuable even
        before they arrive, and that intercepting enemy attacks pays off.
        """
        return defense_ships * self.defense_success_scale

    def _fleet_arrival(self, arrival_ships: int) -> float:
        """Reward for ships that successfully reached an enemy or neutral planet.

        This is the "your fleet navigated correctly" signal — it tells
        the model that the launch angle and target choice were good,
        closing the 4–8 step credit gap between launch and capture.
        """
        return arrival_ships * self.fleet_arrival_scale

    # 4 ── Terminal reward ─────────────────────────────────────────
    def _terminal_reward(self, diff: float, is_done: bool) -> float:
        """Asymmetric win/loss: losing hurts as much as winning helps.

        ±15 makes the terminal signal the largest single reward event,
        ensuring the agent ultimately optimises for victory.
        """
        if not is_done:
            return 0.0
        return self.terminal_win_scale if diff > 0 else self.terminal_lose_scale
