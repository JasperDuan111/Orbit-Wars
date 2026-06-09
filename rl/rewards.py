"""Reward function for Orbit Wars PPO training.

Design principles
-----------------
1. **Planet count, not ship count.**  Owning more planets is the real
   strategic objective; ship counts fluctuate with combat and production
   but are means, not ends.  Rewarding planet-count advantage directly
   incentivises expansion.

2. **Idle penalty with zero threshold.**  Every ship parked on a planet
   is a ship not doing anything useful.  A per-step penalty proportional
   to idle fraction forces the agent to *always* be doing something.

3. **Sparse, high-magnitude territory events.**  Capturing/losing a
   planet is the clearest signal in the game — it should produce a
   reward spike the agent cannot miss.  Neutral captures get a large
   bonus because zero-cost expansion is the most important early-game
   skill.

4. **Asymmetric terminal reward.**  Losing (−15) hurts more than winning
   (+15) rewards — this asymmetry makes "sit still and hope" a losing
   strategy in expectation, even against weak opponents.

5. **No ship-advantage shaping.**  Ship-count deltas punish launching
   fleets (ships in combat look like "losses") — a perverse incentive
   the previous reward suffered from.
"""

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, obs, player_id: int):
        """(Re-)initialise tracking state for a new episode."""
        self._last_planet_owners = planet_owner_map(obs)

    def compute(
        self, obs, player_id: int, is_done: bool,
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

        # ── Dense: planet-count advantage ──
        reward += self._planet_count_advantage(obs, player_id)

        # ── Sparse: territory changes ──
        reward += self._territory_change(obs, player_id)

        # ── Dense: idle-ship penalty (fires from 0 %, no threshold) ──
        reward += self._idle_penalty(obs, player_id)

        # ── Terminal ──
        reward += self._terminal_reward(diff, is_done)

        # ── Persist ──
        self._last_planet_owners = planet_owner_map(obs)

        return my_total, enemy_total, diff, reward

    # ------------------------------------------------------------------
    #  Reward components
    # ------------------------------------------------------------------

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
