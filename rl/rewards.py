"""Reward function for Orbit Wars PPO training.

Design principles
-----------------
1. **Dense signals use the *change* in ship advantage, not the absolute
   level** — this rewards actions that improve position rather than
   rewarding already-strong positions.

2. **Sparse event rewards for territory changes** — capturing a planet
   is a discrete, interpretable event that should produce a clear reward
   spike.  Neutral captures get an extra multiplier because they cost
   nothing to take and are the primary early-game expansion mechanism.

3. **Small combat-efficiency bonus** for favourable ship exchanges.

4. **Gentle idle penalty** to discourage hoarding ships on planets
   without launching attacks.

5. **Terminal reward is the dominant signal** — ±10 dwarfs any single
   dense step (~0.01), so the agent eventually optimises for winning,
   not for farming shaping rewards.

6. **No survival bonus.**  Per-step "stay alive" rewards teach the
   agent to stall rather than to win.
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
# Public helper functions (reusable for analysis / debugging)
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
        self.ship_delta_scale: float = config.ship_advantage_delta_scale
        self.territory_scale: float = config.territory_scale
        self.production_weight: float = config.production_weight
        self.neutral_capture_bonus: float = config.neutral_capture_bonus
        self.combat_efficiency_scale: float = config.combat_efficiency_scale
        self.idle_penalty_scale: float = config.idle_penalty_scale
        self.idle_threshold: float = config.idle_threshold
        self.terminal_win_scale: float = config.terminal_win_scale
        self.terminal_lose_scale: float = config.terminal_lose_scale
        self.invalid_action_penalty: float = config.invalid_action_penalty

        # ── Per-episode tracking ──
        self._last_planet_owners: Optional[Dict[int, int]] = None
        self._last_my_total: Optional[float] = None
        self._last_enemy_total: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, obs, player_id: int):
        """(Re-)initialise tracking state for a new episode."""
        my_total, enemy_total = ship_totals(obs, player_id)
        self._last_my_total = my_total
        self._last_enemy_total = enemy_total
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
        my_prod, eny_prod = planet_production_totals(obs, player_id)

        reward = 0.0

        # ── Dense: ship-advantage trajectory ──
        reward += self._ship_advantage_delta(my_total, enemy_total)

        # ── Sparse: territory changes ──
        reward += self._territory_change(obs, player_id)

        # ── Sparse: combat exchange efficiency ──
        reward += self._combat_efficiency(my_total, enemy_total, my_prod, eny_prod)

        # ── Dense: idle-ship penalty ──
        reward += self._idle_penalty(obs, player_id)

        # ── Terminal ──
        reward += self._terminal_reward(diff, is_done)

        # ── Persist tracking state for next step's delta computations ──
        self._last_my_total = my_total
        self._last_enemy_total = enemy_total
        self._last_planet_owners = planet_owner_map(obs)

        return my_total, enemy_total, diff, reward

    # ------------------------------------------------------------------
    #  Reward components
    # ------------------------------------------------------------------

    # 1 ── Dense: change in normalized ship advantage ────────────────
    def _ship_advantage_delta(self, my_total: float, enemy_total: float) -> float:
        """Reward the *change* in the normalized ship ratio since last step.

        ratio = my / (my + enemy + 1)

        Using the **change** rather than the **absolute** level avoids
        rewarding passive hoarding: only actions that *improve* the
        relative position produce positive reward here.

        Per-step magnitude: ~±0.01–0.05.  Total per episode: ±1 at most.
        """
        if self._last_my_total is None or self._last_enemy_total is None:
            return 0.0

        def _ratio(m, e):
            return m / (m + e + 1.0)

        prev = _ratio(self._last_my_total, self._last_enemy_total)
        cur = _ratio(my_total, enemy_total)
        return (cur - prev) * self.ship_delta_scale

    # 2 ── Territory change ──────────────────────────────────────────
    def _territory_change(self, obs, player_id: int) -> float:
        """Reward planet-ownership transitions.

        - Capturing a planet:   + quality × territory_scale
        - Losing a planet:      − quality × territory_scale
        - Capturing from -1 (neutral):  × (1 + neutral_capture_bonus)

        Neutral captures get a bonus because they cost no combat losses
        and are the fundamental early-game expansion mechanism.
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
            elif old_owner == player_id:                  # me → someone else
                reward -= quality * self.territory_scale

        return reward

    # 3 ── Combat efficiency ───────────────────────────────────────
    def _combat_efficiency(
        self,
        my_total: float, enemy_total: float,
        my_prod: float, eny_prod: float,
    ) -> float:
        """Estimate exchange ratio from ship-count deltas.

        Inferred combat losses::

            my_loss  ≈ last_my  + my_prod  − my_curr
            eny_loss ≈ last_enemy + eny_prod − enemy_curr

        Rewards trades where the enemy lost more ships than we did.
        Noisy (ships in flight are counted in total), so the scale is
        kept small.
        """
        if self._last_my_total is None or self._last_enemy_total is None:
            return 0.0

        my_loss = max(0.0, self._last_my_total + my_prod - my_total)
        eny_loss = max(0.0, self._last_enemy_total + eny_prod - enemy_total)

        if my_loss + eny_loss <= 1:
            return 0.0

        efficiency = eny_loss / (my_loss + eny_loss) - 0.5   # [−0.5, +0.5]
        return efficiency * self.combat_efficiency_scale

    # 4 ── Idle-ship penalty ───────────────────────────────────────
    def _idle_penalty(self, obs, player_id: int) -> float:
        """Penalise having too many ships parked on planets.

        Only fires when the fraction of idle ships exceeds
        ``idle_threshold`` (default 0.5).
        """
        idle = idle_ship_ratio(obs, player_id)
        if idle <= self.idle_threshold:
            return 0.0
        return -(idle - self.idle_threshold) * self.idle_penalty_scale

    # 5 ── Terminal reward ─────────────────────────────────────────
    def _terminal_reward(self, diff: float, is_done: bool) -> float:
        """Large win/loss bonus at episode end.

        ±10 is the dominant signal — roughly 10× any single dense step
        and comparable to the total shaping reward accumulated over the
        whole episode.
        """
        if not is_done:
            return 0.0
        return self.terminal_win_scale if diff > 0 else self.terminal_lose_scale
