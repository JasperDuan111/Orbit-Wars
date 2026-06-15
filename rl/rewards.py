"""Reward function for Orbit Wars PPO training."""

import math
from typing import Dict, Optional, Tuple

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from .config import RewardConfig, DEFAULT_CONFIG
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

    def __init__(self, reward_config: Optional[RewardConfig] = None):
        config = reward_config or DEFAULT_CONFIG.reward

        # ── Scales ──
        self.fleet_advantage_scale: float = config.fleet_advantage_scale
        self.production_advantage_scale: float = config.production_advantage_scale
        self.terminal_economy_scale: float = config.terminal_economy_scale
        self.planet_count_scale: float = config.planet_count_scale
        self.territory_scale: float = config.territory_scale
        self.production_weight: float = config.production_weight
        self.neutral_capture_bonus: float = config.neutral_capture_bonus
        self.idle_penalty_scale: float = config.idle_penalty_scale
        self.terminal_win_scale: float = config.terminal_win_scale
        self.terminal_lose_scale: float = config.terminal_lose_scale
        self.launch_bonus_scale: float = config.launch_bonus_scale
        self.invalid_action_penalty: float = config.invalid_action_penalty

        # ── Per-episode tracking ──
        self._last_planet_owners: Optional[Dict[int, int]] = None

        self.only_economy = config.only_economy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, obs, player_id: int):
        """(Re-)initialise tracking state for a new episode."""
        self._last_planet_owners = planet_owner_map(obs)

    def compute(
        self, obs, player_id: int, is_done: bool, update_cont: int
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

        reward = 0.0
        if self.only_economy:
            reward += self._economy_fleet_advantage(my_total, enemy_total)
            reward += self._economy_production_advantage(obs, player_id)
            reward += self._economy_terminal(obs, player_id, my_total, is_done)

        # ── Persist ──
        self._last_planet_owners = planet_owner_map(obs)

        return my_total, enemy_total, diff, reward

    # ------------------------------------------------------------------
    #  Reward components
    # ------------------------------------------------------------------

    def _economy_fleet_advantage(self, my_total: float, enemy_total: float) -> float:
        return (my_total - enemy_total) * self.fleet_advantage_scale

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
        return (my_total / total_prod) * self.terminal_economy_scale

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

        - Capturing a planet:   + quality × territory_scale
        - Losing a planet:      − quality × territory_scale
        - Neutral → me:  extra ×(1 + neutral_capture_bonus)
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

            if new_owner == player_id:
                bonus = quality * self.territory_scale
                if old_owner == -1:                      # neutral → me
                    bonus *= (1.0 + self.neutral_capture_bonus)
                reward += bonus
            elif old_owner == player_id:                  # me → enemy / neutral
                reward -= quality * self.territory_scale

        return reward

    # 3 ── Idle-ship penalty  (per step, dense) ─────────────────────
    def _idle_penalty(self, obs, player_id: int) -> float:
        """Penalise every idle ship.  No threshold — every parked ship
        costs something, so the agent must always be launching.

        Full idle (1.0) → −idle_penalty_scale per step.
        """
        idle = idle_ship_ratio(obs, player_id)
        return -idle * self.idle_penalty_scale

    # 4 ── Terminal reward ─────────────────────────────────────────
    def _terminal_reward(self, diff: float, is_done: bool) -> float:
        """Asymmetric win/loss: losing hurts as much as winning helps.

        ±15 makes the terminal signal the largest single reward event,
        ensuring the agent ultimately optimises for victory.
        """
        if not is_done:
            return 0.0
        return self.terminal_win_scale if diff > 0 else self.terminal_lose_scale
