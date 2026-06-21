"""Action space for Orbit Wars — per-slot source → target → fraction.

All hot-path functions are vectorised over batch and slot dimensions.
The only per-sample Python loops are the sequential ship-consumption
steps that cannot be parallelised due to the state dependency.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from .config import ActionSpaceConfig, DEFAULT_CONFIG

# ── Game constants for sun-collision detection ────────────────────
_BOARD_SIZE = 100.0
_SUN_CENTER = (_BOARD_SIZE / 2.0, _BOARD_SIZE / 2.0)
_SUN_RADIUS = 10.0
_DEFAULT_MAX_SPEED = 6.0
_ROTATION_RADIUS_LIMIT = 50.0


def _fleet_speed(ships: int, max_speed: float = _DEFAULT_MAX_SPEED) -> float:
    """Replicate the game engine fleet speed formula.

    ``orbit_wars.py:142-143``::

        speed = 1.0 + (max_speed - 1.0) * (log(ships) / log(1000)) ** 1.5
        speed = min(speed, max_speed)
    """
    if ships <= 0:
        return 0.0
    speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5
    return min(speed, max_speed)


def _point_to_segment_dist_sq(px: float, py: float,
                               ax: float, ay: float,
                               bx: float, by: float) -> float:
    """Minimum squared distance from point *p* to segment *a*→*b*."""
    l2 = (ax - bx) ** 2 + (ay - by) ** 2
    if l2 == 0.0:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = max(0.0, min(1.0, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / l2))
    proj_x = ax + t * (bx - ax)
    proj_y = ay + t * (by - ay)
    return (px - proj_x) ** 2 + (py - proj_y) ** 2


def trajectory_blocked(
    src_x: float, src_y: float,
    angle: float,
    tgt_x: float, tgt_y: float,
    center: Tuple[float, float] = _SUN_CENTER,
    sun_radius: float = _SUN_RADIUS,
    board_size: float = _BOARD_SIZE,
    src_radius: float = 0.0,
) -> bool:
    """Return True if the straight-line ray from src at *angle* hits the sun
    OR the board edge **before** reaching (tgt_x, tgt_y).

    ``src_radius`` shifts the ray start outward to match the engine's
    actual fleet spawn point (planet surface + 0.1).  Without this, edge
    distances are underestimated, letting fleets that would overshoot the
    board through the check.
    """
    # Engine spawn: planet_center + (radius + 0.1) * direction
    offset = src_radius + 0.1
    src_x = src_x + math.cos(angle) * offset
    src_y = src_y + math.sin(angle) * offset

    # If the spawn point is already outside the board, the fleet is
    # immediately destroyed by the engine — block it before launch.
    if not (0 <= src_x <= board_size and 0 <= src_y <= board_size):
        return True

    dx, dy = math.cos(angle), math.sin(angle)

    # Distance from source to target projected along the ray
    proj_tgt = (tgt_x - src_x) * dx + (tgt_y - src_y) * dy
    if proj_tgt <= 0:
        return True

    # ── Sun check ──
    to_cx, to_cy = center[0] - src_x, center[1] - src_y
    proj_sun = to_cx * dx + to_cy * dy
    if proj_sun > 0 and proj_sun < proj_tgt:
        closest_x = src_x + proj_sun * dx
        closest_y = src_y + proj_sun * dy
        d2 = (center[0] - closest_x) ** 2 + (center[1] - closest_y) ** 2
        if d2 < sun_radius * sun_radius + 1e-6:
            return True

    # ── Board edge check ──
    for coord, axis in ((board_size, 'x'), (0.0, 'x'), (board_size, 'y'), (0.0, 'y')):
        if axis == 'x':
            if abs(dx) < 1e-10:
                continue
            t = (coord - src_x) / dx
        else:
            if abs(dy) < 1e-10:
                continue
            t = (coord - src_y) / dy
        if 0 < t < proj_tgt:
            return True

    return False


def trajectory_hits_sun(
    src_x: float, src_y: float,
    angle: float = None,
    tgt_x: float = None, tgt_y: float = None,
    center: Tuple[float, float] = _SUN_CENTER,
    sun_radius: float = _SUN_RADIUS,
) -> bool:
    """Return True if the fleet will cross the sun **before** reaching the target.

    Computes the closest approach of the straight-line ray to the sun.
    If the sun's closest point lies **beyond** the target (i.e. the fleet
    would reach the target planet first), the launch is considered safe.
    """
    if angle is None:
        angle = math.atan2(tgt_y - src_y, tgt_x - src_x)
    dx, dy = math.cos(angle), math.sin(angle)

    # Closest approach to the sun along the ray
    to_cx, to_cy = center[0] - src_x, center[1] - src_y
    proj_sun = to_cx * dx + to_cy * dy
    if proj_sun <= 0:
        return False  # sun behind us

    # Distance from target along the ray
    if tgt_x is not None:
        proj_tgt = (tgt_x - src_x) * dx + (tgt_y - src_y) * dy
        if proj_sun > proj_tgt:
            return False  # target is closer than the sun danger zone

    closest_x, closest_y = src_x + proj_sun * dx, src_y + proj_sun * dy
    d2 = (center[0] - closest_x) ** 2 + (center[1] - closest_y) ** 2
    return d2 < sun_radius * sun_radius + 1e-6


def _get_field(obs, name, default=None):
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def _parse_planets(obs):
    return [Planet(*p) for p in _get_field(obs, "planets", [])]


def build_orbit_lookup(obs):
    """Build {planet_id: {orbital_r, initial_angle, rotates}} from raw obs.

    Uses ``initial_planets`` (list of [id, owner, x, y, radius, ...])
    to derive orbital parameters.  A planet rotates when
    ``orbital_radius + radius < ROTATION_RADIUS_LIMIT`` (50.0).
    """
    initial = _get_field(obs, "initial_planets", None)
    if initial is None:
        return {}
    lookup = {}
    cx, cy = _SUN_CENTER
    for p in initial:
        pid, _, px, py, pr = p[0], p[1], p[2], p[3], p[4]
        dx, dy = px - cx, py - cy
        r = math.hypot(dx, dy)
        lookup[pid] = {
            'orbital_r': r,
            'initial_angle': math.atan2(dy, dx),
            'rotates': (r + pr) < _ROTATION_RADIUS_LIMIT,
        }
    return lookup


def compute_intercept_angle(
    src_x: float, src_y: float,
    tgt_id: int,
    tgt_x: float, tgt_y: float,
    fleet_speed: float,
    orbit_lookup: dict,
    angular_velocity: float,
    src_radius: float = 0.0,
    return_pos: bool = False,
):
    """Compute firing angle to intercept a (possibly orbiting) target.

    For static planets the angle is simply the bearing to the current
    position.  For rotating planets the **secant method** solves for the
    flight time ``t`` that satisfies::

        distance(source, target_pos(θ₀ + ω·t)) - src_radius - 0.1 - fleet_speed · t = 0

    The ``− src_radius − 0.1`` term accounts for the engine's fleet
    spawn point (planet surface + 0.1), which is closer to the target
    than the planet center. Without this correction predicted arrival
    times are 2–5 steps too late, causing systematic misses.

    The secant method converges superlinearly and is robust across all
    parameter ranges, unlike the previous fixed-point iteration which
    diverged when ``ω·r / fleet_speed > 1`` (slow fleets vs. fast, far
    orbiting planets).

    When ``return_pos`` is True, also returns ``(future_tx, future_ty)``
    — the cartesian coordinates of the predicted intercept point.  This
    allows ``trajectory_hits_sun`` to correctly judge whether the fleet
    passes through the sun *before* reaching the actual intercept point.
    """
    info = orbit_lookup.get(tgt_id) if orbit_lookup else None
    if info is None or not info['rotates'] or angular_velocity == 0:
        angle = math.atan2(tgt_y - src_y, tgt_x - src_x)
        return (angle, tgt_x, tgt_y) if return_pos else angle

    r = info['orbital_r']
    cx, cy = _SUN_CENTER

    # Current angular position of the target, derived from the planet's
    # actual coordinates in the observation (not from ``step``, which is
    # managed by the framework and can be offset from the true position).
    theta_0 = math.atan2(tgt_y - cy, tgt_x - cx)

    # ── Helper: target cartesian position after *t* ticks ──────────
    def _target_pos(t: float):
        th = theta_0 + angular_velocity * t
        return (cx + r * math.cos(th), cy + r * math.sin(th))

    # ── f(t) = distance(center, target(t)) - fleet_speed · t - spawn_offset ──
    #     Fleet spawns at planet_surface + 0.1 along the launch ray, so the
    #     effective travel distance starts (src_radius + 0.1) shorter.
    spawn_offset = src_radius + 0.1

    def _f(t: float):
        tx, ty = _target_pos(t)
        d = math.hypot(ty - src_y, tx - src_x)
        return d - fleet_speed * t - spawn_offset

    # Initial guess: straight-line flight time to the current position
    t_cur = math.hypot(tgt_y - src_y, tgt_x - src_x) / max(fleet_speed, 0.01)
    f_cur = _f(t_cur)

    if f_cur <= 0:
        # Straight-line aim already reaches or overshoots — the minimal
        # positive root is between 0 and t_cur.  Bisect to find it.
        t_lo, f_lo = 0.0, _f(0.0)  # f(0) = distance - spawn_offset
        t_hi, f_hi = t_cur, f_cur
    else:
        # f(t_cur) > 0 — the planet "runs away".  Walk forward in steps
        # ≤ 1/8 of the orbit period (so we cannot miss an f < 0 window)
        # until we reach an f ≤ 0 point or the guaranteed upper bound
        # t_ub = (D + r) / v  where f(t_ub) ≤ 0 by triangle inequality.
        t_lo, f_lo = t_cur, f_cur

        src_dist_to_center = math.hypot(src_x - cx, src_y - cy)
        t_ub = (src_dist_to_center + r) / max(fleet_speed, 0.01) + 0.1
        orbit_period = 2.0 * math.pi / max(angular_velocity, 0.001)
        t_step = max(1.0, orbit_period / 10.0)

        t_hi = t_cur + t_step
        f_hi = _f(t_hi)
        while f_hi > 0 and t_hi < t_ub:
            t_lo, t_hi = t_hi, t_hi + t_step
            f_lo, f_hi = f_hi, _f(t_hi)

    # If the bracket endpoints are still same-sign, fall back to
    # the guaranteed upper bound (should never happen, but guard).
    if f_lo * f_hi > 0:
        t_hi = t_ub
        f_hi = _f(t_hi)

    # Already good enough — return immediately
    if abs(f_hi) < 1e-3:
        tx, ty = _target_pos(t_hi)
        angle = math.atan2(ty - src_y, tx - src_x)
        return (angle, tx, ty) if return_pos else angle

    # ── Secant method on guaranteed bracket [t_lo, t_hi] where ────
    #     f(t_lo) > 0 and f(t_hi) ≤ 0.
    t0, f0 = t_lo, f_lo
    t1, f1 = t_hi, f_hi

    # Secant method:  t_{n+1} = t_n - f(t_n) · (t_n - t_{n-1}) / (f(t_n) - f(t_{n-1}))
    for _ in range(15):
        denom = f1 - f0
        if abs(denom) < 1e-12:
            break          # near-flat — accept current estimate

        t2 = t1 - f1 * (t1 - t0) / denom
        # Guard against negative or unreasonably large steps
        if t2 < 0:
            t2 = t1 * 0.5
        elif t2 > t1 * 10:
            t2 = t1 * 2.0

        t0, t1 = t1, t2
        f0, f1 = f1, _f(t1)

        if abs(f1) < 1e-3:
            break
        if abs(t1 - t0) < 1e-4:
            break

    tx, ty = _target_pos(t1)
    angle = math.atan2(ty - src_y, tx - src_x)
    return (angle, tx, ty) if return_pos else angle


class ActionBuilder:
    def __init__(self, action_config: Optional[ActionSpaceConfig] = None):
        config = action_config or DEFAULT_CONFIG.action
        self.max_sources = config.max_sources
        self.max_launches = config.max_launches_per_source
        self.ship_fractions = config.ship_fractions

    def get_planet_data(self, obs):
        import math
        planets = _parse_planets(obs)
        player_id = _get_field(obs, "player", 0)
        # Sort planets *identically* to ``obs.encode_observation`` so
        # that Zp indices match this ``planets`` list.  Owned planets
        # come first, then enemy, then neutral; within each group
        # planets are ordered by distance to the sun.
        sun_x, sun_y = 50.0, 50.0
        planets.sort(key=lambda p: (
            -(1 if p.owner == player_id else 0),
            -(1 if (p.owner != player_id and p.owner != -1) else 0),
            math.hypot(p.x - sun_x, p.y - sun_y),
        ))
        my = [(i, p) for i, p in enumerate(planets) if p.owner == player_id]
        my.sort(key=lambda x: x[1].ships, reverse=True)
        # Exclude comets from valid targets — they move along fixed paths,
        # not circular orbits, so our intercept-angle prediction is wrong
        # for them.  Targeting a comet almost guarantees a miss.
        comet_ids = set(_get_field(obs, "comet_planet_ids", []))
        non_my = [i for i, p in enumerate(planets)
                  if p.owner != player_id and p.id not in comet_ids]
        return planets, [i for i, _ in my], non_my, player_id

    def decode(self, planet_by_idx, src_obs_idx, tgt_obs_idx, frac_idx, remaining,
               orbit_lookup=None, angular_velocity=0.0):
        src = planet_by_idx[src_obs_idx]
        tgt = planet_by_idx[tgt_obs_idx]
        frac = self.ship_fractions[frac_idx]
        ships = max(1, min(int(remaining * frac), remaining))

        # Intercept angle: compensate for target orbital motion.
        # Also retrieve the predicted intercept point so that the sun
        # collision check uses the *actual* arrival position, not the
        # target's current coordinates.
        speed = _fleet_speed(ships)
        angle, fut_x, fut_y = compute_intercept_angle(
            src_x=src.x, src_y=src.y,
            tgt_id=tgt.id, tgt_x=tgt.x, tgt_y=tgt.y,
            fleet_speed=speed,
            orbit_lookup=orbit_lookup or {},
            angular_velocity=angular_velocity,
            src_radius=src.radius,
            return_pos=True,
        )

        if trajectory_blocked(src.x, src.y, angle=angle,
                                tgt_x=fut_x, tgt_y=fut_y,
                                src_radius=src.radius):
            return None
        return [src.id, angle, ships]

    def decode_all(self, planets, action_indices, source_indices,
                   planet_ships, valid_targets,
                   orbit_lookup=None, angular_velocity=0.0):
        """Convert per-slot action indices into game moves."""
        moves = []
        remaining = dict(planet_ships)
        _S = source_indices.shape[0]
        n_slots = min(_S, self.max_sources)
        _ol = orbit_lookup or {}

        for s in range(n_slots):
            src_idx = int(source_indices[s].item())
            rem = remaining.get(src_idx, 0)
            if rem <= 0:
                continue
            for ll in range(self.max_launches):
                tgt_cat = int(action_indices[s, ll, 0].item())
                if tgt_cat < 0:
                    continue      # sun-blocked → skip to next launch
                if tgt_cat == 0:
                    break         # STOP → end this source's launches
                real_tgt = valid_targets[tgt_cat - 1]
                frac_idx = int(action_indices[s, ll, 1].item())
                move = self.decode(planets, src_idx, real_tgt, frac_idx, rem,
                                   _ol, angular_velocity)
                if move is None:
                    continue
                moves.append(move)
                rem -= move[2]
                if rem <= 0:
                    break
            remaining[src_idx] = rem
        return moves


# ── Per-slot source selection (kept as is, used by old callers) ────

def select_sources(source_logits, ownership_mask, k, deterministic=False):
    """Per-slot source selection."""
    if source_logits.shape[-1] <= 1:
        return None
    S, N = source_logits.shape
    device = source_logits.device
    source_indices = torch.zeros(S, dtype=torch.long, device=device)
    for s in range(S):
        scores = source_logits[s]
        if deterministic:
            masked = scores.masked_fill(~ownership_mask, float('-inf'))
            _, idx = torch.topk(masked, k=1, dim=-1)
        else:
            gumbel = -torch.log(-torch.log(torch.rand(N, device=device) + 1e-10) + 1e-10)
            gumbel_scores = scores + gumbel
            gumbel_scores = gumbel_scores.masked_fill(~ownership_mask, float('-inf'))
            _, idx = torch.topk(gumbel_scores, k=1, dim=-1)
        source_indices[s] = idx[0]
    return source_indices


def source_selection_logprob(source_logits, ownership_mask, source_indices):
    """Per-slot source logprob."""
    if source_indices is None or source_logits.shape[-1] <= 1:
        return torch.zeros((), device=source_logits.device)
    S = source_logits.shape[0]
    device = source_logits.device
    lp = torch.zeros((), device=device)
    for s in range(S):
        scores = source_logits[s].nan_to_num(0.0)
        scores = scores.masked_fill(~ownership_mask, float('-inf'))
        logp = scores.log_softmax(dim=-1).nan_to_num(0.0)
        idx = source_indices[s].item()
        lp = lp + logp[idx]
    return lp


# ── Rollout sampling (no_grad) ────────────────────────────────────

def _build_tgt_combined(target_scores, stop_logits, valid_indices_tensor, pad_to):
    """Build padded combined target/STOP logits for one env.

    Returns (S, PAD) tensor where col 0 = stop, cols 1..1+nv = target scores.
    """
    device = target_scores.device
    S = target_scores.shape[0]
    nv = valid_indices_tensor.shape[0]
    pad = max(2, 1 + nv, pad_to)
    combined = torch.full((S, pad), float('-inf'), device=device)
    combined[:, 0] = stop_logits.squeeze(-1).nan_to_num(0.0)
    if nv:
        combined[:, 1:1 + nv] = target_scores[:, valid_indices_tensor].nan_to_num(0.0)
    return combined


def sample_action_discrete(
    source_logits,
    ownership_mask,
    target_scores,
    stop_logits,
    frac_logits_all,
    valid_targets,
    planet_ships,          # {planet_idx: ships} dict for owned planets
    max_launches,
    deterministic,
    ship_fractions,
    epsilon=0.1,
    planets=None,          # list of Planet objects (for sun collision)
    orbit_lookup=None,     # {planet_id: {orbital_r, initial_angle, rotates}} — for intercept
    angular_velocity=0.0,  # radians per step
):
    """Per-slot action sampling: source → target → fraction.

    ε-greedy exploration: random uniform with prob ε, argmax with prob 1-ε.
    Logprobs use the model's softmax distribution.  Ships are consumed and
    logprobs recorded **only after** a sun-collision check with the actual
    fleet speed (computed from the chosen fraction), so blocked launches
    cost no probability and waste no ships — no pre-computed visibility
    needed.

    ``action_indices`` are initialised to -1 (invalid).  ``decode_all``
    treats negative ``tgt_cat`` as a no-op (sun-blocked launch) and
    ``tgt_cat == 0`` as STOP.

    **Sequential source masking**: once a planet is selected by a previous
    slot it is masked from all subsequent slots, forcing slot diversity.

    Returns
    -------
    action_indices  (S, L, 2) — (tgt_cat, frac_idx) per slot per launch
    source_indices  (S,)      — selected planet index per slot
    lp              scalar    — total log-probability (softmax)
    ent             scalar    — total entropy (softmax)
    """
    S, N = target_scores.shape
    L = max_launches
    device = target_scores.device
    n_fracs = len(ship_fractions)
    eps = 0.0 if deterministic else epsilon
    fill_val = float('-inf')

    action_indices = torch.full((S, L, 2), -1, dtype=torch.long, device=device)
    source_indices = torch.zeros(S, dtype=torch.long, device=device)
    lp = torch.zeros((), device=device)
    ent = torch.zeros((), device=device)

    # ── Target/STOP combined: pre-computed (per-slot, same every slot) ──
    valid_t = torch.tensor(valid_targets, dtype=torch.long, device=device)
    combined = _build_tgt_combined(target_scores, stop_logits, valid_t, 0)
    tgt_logp = combined.log_softmax(dim=-1).nan_to_num(0.0)  # (S, PAD)
    safe_tgt = tgt_logp.masked_fill(tgt_logp == fill_val, 0.0)
    tgt_ent = -(safe_tgt.exp() * safe_tgt).sum(dim=-1)       # (S,)

    # ── GPU random numbers (one shot) ──────────────────────────────
    owned_idx = ownership_mask.nonzero(as_tuple=False).squeeze(-1)
    n_owned = len(owned_idx)

    rand_source = torch.rand(S, device=device)
    rand_tgt    = torch.rand(S, L, device=device)
    rand_frac   = torch.rand(S, L, device=device)
    rand_source_int = torch.randint(0, max(n_owned, 1), (S,), device=device)
    rand_tgt_int    = torch.randint(0, max(len(valid_targets), 0) + 1, (S, L), device=device)
    rand_frac_int   = torch.randint(0, n_fracs, (S, L), device=device)

    # Base source logits — ownership + has_ships (drain mask added per slot)
    has_ships = torch.zeros(N, dtype=torch.bool, device=device)
    for pi, cnt in planet_ships.items():
        if cnt > 0:
            has_ships[pi] = True
    src_base = source_logits.masked_fill(
        ~ownership_mask.unsqueeze(0) | ~has_ships.unsqueeze(0), fill_val)

    # ── Ship state (Python) for sequential consumption ─────────────
    remaining_ships = dict(planet_ships)

    for s in range(min(S, len(planet_ships))):
        if n_owned == 0:
            continue

        # ── 0. Mask planets already selected by previous slots ──
        used_mask = torch.zeros(N, dtype=torch.bool, device=device)
        for prev_s in range(s):
            used_mask[source_indices[prev_s]] = True

        # ── 1. Source logprobs & entropy for THIS slot (recomputed) ──
        src_s = src_base[s].masked_fill(used_mask, fill_val)
        src_logp_s = src_s.log_softmax(dim=-1).nan_to_num(0.0)
        safe_src_s = src_logp_s.masked_fill(src_logp_s == fill_val, 0.0)
        src_ent_s = -(safe_src_s.exp() * safe_src_s).sum(dim=-1)

        # ── 2. Source selection ──
        #     Filter to planets with ships AND not already used.
        avail_source = owned_idx[has_ships[owned_idx] & ~used_mask[owned_idx]] if n_owned > 0 else owned_idx
        if len(avail_source) == 0:
            break

        if eps > 0 and rand_source[s].item() < eps:
            src_idx = int(avail_source[rand_source_int[s].item() % len(avail_source)].item())
        else:
            src_idx = int(torch.argmax(src_s).item())

        source_indices[s] = src_idx
        lp += src_logp_s[src_idx]
        ent += src_ent_s

        rem = remaining_ships.get(src_idx, 0)
        if rem <= 0:
            remaining_ships[src_idx] = 0
            continue

        src_p = planets[src_idx] if planets is not None else None

        for ll in range(L):
            # 3. Target / STOP selection
            if eps > 0 and rand_tgt[s, ll].item() < eps:
                tgt_cat = int(rand_tgt_int[s, ll].item())
            else:
                tgt_cat = int(torch.argmax(combined[s]).item())

            if tgt_cat == 0:
                lp += tgt_logp[s, tgt_cat]
                ent += tgt_ent[s]
                action_indices[s, ll, 0] = tgt_cat
                break

            real_tgt_idx = valid_targets[tgt_cat - 1]

            # 4. Fraction selection
            f_logits = frac_logits_all[s, real_tgt_idx]
            f_logp = f_logits.log_softmax(dim=-1).nan_to_num(0.0)

            if eps > 0 and rand_frac[s, ll].item() < eps:
                fa = int(rand_frac_int[s, ll].item())
            else:
                fa = int(torch.argmax(f_logits).item())

            _f = ship_fractions[fa]
            ships = max(1, min(int(rem * _f), rem))

            # 5. Sun check with REAL fleet speed from chosen fraction.
            #    Blocked → skip silently: no lp, no ship, no action record.
            if planets is not None and src_p is not None:
                tgt_p = planets[real_tgt_idx]
                speed = _fleet_speed(ships)
                int_angle, fut_x, fut_y = compute_intercept_angle(
                    src_x=src_p.x, src_y=src_p.y,
                    tgt_id=tgt_p.id, tgt_x=tgt_p.x, tgt_y=tgt_p.y,
                    fleet_speed=speed,
                    orbit_lookup=orbit_lookup or {},
                    angular_velocity=angular_velocity,
                    src_radius=src_p.radius,
                    return_pos=True,
                )
                if trajectory_blocked(src_p.x, src_p.y, angle=int_angle,
                                        tgt_x=fut_x, tgt_y=fut_y,
                                        src_radius=src_p.radius):
                    continue  # no lp, no consumption — ll advances

            # ── Commit: only after sun check passes ──
            lp += tgt_logp[s, tgt_cat]
            ent += tgt_ent[s]
            lp += f_logp[fa]
            safe_flp = f_logp.masked_fill(f_logp == fill_val, 0.0)
            ent -= (safe_flp.exp() * safe_flp).sum()
            action_indices[s, ll, 0] = tgt_cat
            action_indices[s, ll, 1] = fa

            rem -= ships
            if rem <= 0:
                break

        remaining_ships[src_idx] = rem

    return action_indices, source_indices, lp, ent


# ── PPO logprob (with gradients) ──────────────────────────────────

def logprob_batched_combined(
    src_logits,
    tgt_scores,
    stop_logits,
    frac_logits_all,
    ownership_mask,
    action_indices,           # (B, S, L, 2)
    source_indices,            # list of (S,) tensors, one per batch element
    all_valid_targets,
    all_planet_ships,          # list of dicts {planet_idx: ships}
    max_launches,
    ship_fractions,
    all_planets=None,          # list of Planet lists (for sun collision; optional)
    all_orbit_lookups=None,    # list of orbit lookup dicts
    all_ang_vels=None,         # list of float angular_velocities
):
    """Batched log-probability + entropy for PPO update.

    Ship consumption during replay matches actual execution:
    sun-colliding launches do NOT consume ships, using the same
    intercept-angle + future-position sun check as ``decode()``.

    Logprobs are only accumulated after the sun check passes, matching
    ``sample_action_discrete`` exactly.  Slots with ``tgt_cat < 0``
    are sun-blocked and skipped without contribution.
    """
    B, S, N = tgt_scores.shape
    L = max_launches
    device = tgt_scores.device
    fill_val = float('-inf')

    src_idx_batch = torch.stack(source_indices, dim=0).to(device)  # (B, S)
    src_base = src_logits.masked_fill(~ownership_mask.unsqueeze(1), fill_val)  # (B, S, N)
    # Per-batch has_ships: 0-ship planets are ineligible even before used_mask
    has_ships_batch = torch.zeros(B, N, dtype=torch.bool, device=device)
    for b in range(B):
        for pi, cnt in all_planet_ships[b].items():
            if cnt > 0:
                has_ships_batch[b, pi] = True
    src_base = src_base.masked_fill(~has_ships_batch.unsqueeze(1), fill_val)

    # ── 1. Source logprob & entropy — per-slot with sequential used mask ──
    src_lps = torch.zeros(B, device=device)
    src_ents = torch.zeros(B, device=device)

    # ── 2. Target/STOP combined — one batched tensor (B, S, PAD) ──
    maxV = max((len(vt) for vt in all_valid_targets), default=0)
    PAD = max(1, 1 + maxV)
    combined = torch.full((B, S, PAD), fill_val, device=device)
    combined[..., 0] = stop_logits.squeeze(-1).nan_to_num(0.0)
    for b in range(B):
        vt = all_valid_targets[b]
        nv = len(vt)
        if nv:
            vi = torch.tensor(vt, dtype=torch.long, device=device)
            combined[b, :, 1:1 + nv] = tgt_scores[b, :, vi].nan_to_num(0.0)

    all_logp = combined.log_softmax(dim=-1).nan_to_num(0.0)      # (B, S, PAD)
    safe_lp = all_logp.masked_fill(all_logp == fill_val, 0.0)
    all_ent = -(safe_lp.exp() * safe_lp).sum(dim=-1).nan_to_num(0.0)  # (B, S)

    # ── 3. Target + fraction replay (stateful per-slot loop) ──
    tgt_lps  = torch.zeros(B, device=device)
    tgt_ents = torch.zeros(B, device=device)
    frac_lps = torch.zeros(B, device=device)
    frac_ents = torch.zeros(B, device=device)
    source_count = torch.zeros(B, device=device)

    for b in range(B):
        valid = all_valid_targets[b]
        remaining = dict(all_planet_ships[b])              # mutable — tracks consumption
        act_b = action_indices[b]                            # (S, L, 2)
        tgt_cats = act_b[:, :, 0]                            # (S, L)
        frac_idxs = act_b[:, :, 1]                           # (S, L)
        planets = all_planets[b] if all_planets else None

        for s in range(S):
            # ── 0. Mask planets already selected by previous slots ──
            used_mask = torch.zeros(N, dtype=torch.bool, device=device)
            for prev_s in range(s):
                used_mask[src_idx_batch[b, prev_s]] = True

            # ── All sources with ships available → stop ──
            owned_avail = (ownership_mask[b] & has_ships_batch[b] & ~used_mask).sum().item()
            if owned_avail == 0:
                break

            src_bs = src_base[b, s].masked_fill(used_mask, fill_val)
            src_logp_bs = src_bs.log_softmax(dim=-1).nan_to_num(0.0)
            safe_src_bs = src_logp_bs.masked_fill(src_logp_bs == fill_val, 0.0)
            src_ents[b] += -(safe_src_bs.exp() * safe_src_bs).sum()

            si = src_idx_batch[b, s].item()
            src_lps[b] += src_logp_bs[si]

            rem = remaining.get(si, 0)
            if rem <= 0:
                continue

            committed = False  # track whether this source had ≥1 successful launch

            if planets is not None:
                src_p = planets[si]
            else:
                src_p = None

            for ll in range(L):
                tc = int(tgt_cats[s, ll].item())
                if tc < 0:
                    continue      # sun-blocked → skip, no lp
                if tc == 0:
                    tgt_lps[b] += all_logp[b, s, tc]
                    tgt_ents[b] += all_ent[b, s]
                    break
                ti = valid[tc - 1]
                fl = frac_logits_all[b, s, ti].nan_to_num(0.0)
                flp = fl.log_softmax(dim=-1).nan_to_num(0.0)
                fi = int(frac_idxs[s, ll].item())
                _f = ship_fractions[fi]
                ships = max(1, min(int(rem * _f), rem))

                # Sun collision guard — uses real chosen ships for speed.
                # Blocked → skip silently (no lp, no ship consumption).
                if planets is not None and src_p is not None:
                    tgt_p = planets[ti]
                    speed = _fleet_speed(ships)
                    ol = all_orbit_lookups[b] if all_orbit_lookups else {}
                    _av = all_ang_vels[b] if all_ang_vels else 0.0
                    int_angle, fut_x, fut_y = compute_intercept_angle(
                        src_x=src_p.x, src_y=src_p.y,
                        tgt_id=tgt_p.id, tgt_x=tgt_p.x, tgt_y=tgt_p.y,
                        fleet_speed=speed,
                        orbit_lookup=ol,
                        angular_velocity=_av,
                        src_radius=src_p.radius,
                        return_pos=True,
                    )
                    if trajectory_blocked(src_p.x, src_p.y, angle=int_angle,
                                            tgt_x=fut_x, tgt_y=fut_y,
                                            src_radius=src_p.radius):
                        continue  # no lp, no consumption

                # ── Commit: only after sun check passes ──
                if not committed:
                    source_count[b] += 1
                    committed = True
                tgt_lps[b] += all_logp[b, s, tc]
                tgt_ents[b] += all_ent[b, s]
                frac_lps[b] += flp[fi]
                safe_flp = flp.masked_fill(flp == fill_val, 0.0)
                frac_ents[b] -= (safe_flp.exp() * safe_flp).sum()

                rem -= ships
                if rem <= 0:
                    break

            remaining[si] = rem

    return (src_lps + tgt_lps + frac_lps,
            tgt_ents + frac_ents + src_ents,
            source_count)
