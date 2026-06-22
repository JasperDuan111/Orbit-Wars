"""Action space for Orbit Wars — per-slot source → offset → targets.

Offset scheme: each source picks an offset ∈ [0, 3, 5, 10, 20] that adds extra
ships on top of (target_ships + 1).  Ships are allocated to each target with
the full need (ts + 1 + offset), proportionally scaled if total exceeds available.
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


# ── Offset-based ship allocation ──────────────────────────────────

def _allocate_ships_offset(
    source_ships: int,
    offset: float,
    targets_with_ships: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Allocate ships to targets proportionally (no per-source pruning).

    - Enemy (ts > 0): ``target_ships + 1 + offset``
    - Neutral (ts = 0): ``max(5, offset + 1)``

    Per-source pruning is deliberately omitted — multiple sources can
    converge on the same target, and what matters is the *total* ships
    arriving.  Final pruning by total happens in ``decode_all``.
    """
    if source_ships <= 0 or not targets_with_ships:
        return [(t[0], 0) for t in targets_with_ships]

    n = len(targets_with_ships)
    needs = [
        max(3, int(offset) + 1) if ts == 0 else max(1, ts + 1 + int(offset))
        for _, ts in targets_with_ships
    ]
    total_need = sum(needs)

    if total_need <= source_ships:
        allocations = needs
    else:
        # Floor-based proportional allocation — never exceeds source_ships
        allocations = [int(n_need * source_ships / total_need) for n_need in needs]
        # Raise zeros to 1 (but cap at source_ships — drop targets if necessary)
        for i in range(n):
            if allocations[i] == 0:
                allocations[i] = 1
        total = sum(allocations)
        # Trim excess by dropping targets from the end (lowest priority)
        while total > source_ships and n > 0:
            n -= 1
            if allocations[n] > 0:
                total -= allocations[n]
                allocations[n] = 0
        # Distribute leftover safely
        leftover = source_ships - total
        for i in range(leftover):
            if allocations[i % max(n, 1)] > 0:
                allocations[i % max(n, 1)] += 1
        # Cap each allocation at its need
        for i in range(len(allocations)):
            if allocations[i] > needs[i]:
                allocations[i] = needs[i]

    return [(targets_with_ships[i][0], allocations[i]) for i in range(len(targets_with_ships))]


# ── ActionBuilder ─────────────────────────────────────────────────

class ActionBuilder:
    def __init__(self, action_config: Optional[ActionSpaceConfig] = None):
        config = action_config or DEFAULT_CONFIG.action
        self.max_sources = config.max_sources
        self.max_launches = config.max_launches_per_source
        self.offset_bins = config.offset_bins

    def get_planet_data(self, obs):
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

    def decode(self, planet_by_idx, src_obs_idx, tgt_obs_idx, ships,
               orbit_lookup=None, angular_velocity=0.0):
        """Decode one launch: source → target with *ships* ships."""
        src = planet_by_idx[src_obs_idx]
        tgt = planet_by_idx[tgt_obs_idx]

        # Intercept angle: compensate for target orbital motion.
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
                   valid_targets, offset_indices, offset_bins,
                   orbit_lookup=None, angular_velocity=0.0):
        """Two-phase decode: per-source allocation → per-target aggregation + prune.

        Phase 1 — each source allocates ships to its targets independently
        (no per-source pruning, so multiple sources can converge).

        Phase 2 — aggregate all allocations by target.  If the total ships
        across all sources heading to a target is less than ``target_ships + 1``
        (the bare minimum to flip it), drop all moves to that target.
        Leftover ships stay on their source planets for defence.
        """
        _S = source_indices.shape[0]
        n_slots = min(_S, self.max_sources)
        _ol = orbit_lookup or {}

        # Phase 1 — collect all (src_idx, tgt_idx, ships) allocations
        raw_allocations: List[Tuple[int, int, int]] = []  # (src, tgt, ships)

        for s in range(n_slots):
            src_idx = int(source_indices[s].item())
            src_p = planets[src_idx]
            src_ships = src_p.ships
            if src_ships <= 0:
                continue

            oi = int(offset_indices[s].item())
            offset_val = offset_bins[oi]

            selected_targets = []
            for ll in range(self.max_launches):
                tgt_cat = int(action_indices[s, ll].item())
                if tgt_cat < 0:
                    continue
                if tgt_cat == 0:
                    break
                real_tgt = valid_targets[tgt_cat - 1]
                tgt_p = planets[real_tgt]
                selected_targets.append((real_tgt, tgt_p.ships))

            if not selected_targets:
                continue

            allocations = _allocate_ships_offset(src_ships, offset_val, selected_targets)
            for tgt_idx, ships in allocations:
                if ships > 0:
                    raw_allocations.append((src_idx, tgt_idx, ships))

        # Phase 2 — aggregate by target and prune
        tgt_totals: Dict[int, List[Tuple[int, int]]] = {}  # tgt → [(src, ships), ...]

        for src_idx, tgt_idx, ships in raw_allocations:
            tgt_totals.setdefault(tgt_idx, []).append((src_idx, ships))

        moves = []
        for tgt_idx, contributions in tgt_totals.items():
            tgt_p = planets[tgt_idx]
            total_ships = sum(s for _, s in contributions)

            if tgt_p.ships > 0 and total_ships < tgt_p.ships + 1:
                # Not enough to flip this enemy planet — skip all moves
                continue
            # Neutral planet (ships=0): any amount is fine

            for src_idx, ships in contributions:
                move = self.decode(planets, src_idx, tgt_idx, ships,
                                   _ol, angular_velocity)
                if move is not None:
                    moves.append(move)

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
    offset_logits,
    valid_targets,
    planet_ships,          # {planet_idx: ships} dict for owned planets
    max_launches,
    offset_bins,
    deterministic,
    epsilon=0.1,
    planets=None,          # list of Planet objects (for sun collision)
    orbit_lookup=None,     # {planet_id: {orbital_r, initial_angle, rotates}} — for intercept
    angular_velocity=0.0,  # radians per step
):
    """Per-slot action sampling: source → offset → targets.

    Offset scheme:
    1. Select source planet.
    2. Sample offset from discrete bins [0, 3, 5, 10, 20].
    3. Collect targets until STOP (or max_launches).
    4. Allocate ships via ``_allocate_ships_offset(offset, ...)``:
       each target gets ``target_ships + 1 + offset``, proportionally
       scaled if total need exceeds available ships.

    Returns
    -------
    action_indices  (S, L)   — tgt_cat per slot per launch
    source_indices  (S,)     — selected planet index per slot
    offset_indices  (S,)     — selected offset bin index per slot
    lp              scalar   — total log-probability (softmax)
    ent             scalar   — total entropy (softmax)
    """
    S, N = target_scores.shape
    L = max_launches
    device = target_scores.device
    n_offs = len(offset_bins)
    eps = 0.0 if deterministic else epsilon
    fill_val = float('-inf')

    action_indices = torch.full((S, L), -1, dtype=torch.long, device=device)
    source_indices = torch.zeros(S, dtype=torch.long, device=device)
    offset_indices = torch.zeros(S, dtype=torch.long, device=device)
    lp = torch.zeros((), device=device)
    ent = torch.zeros((), device=device)

    # ── Target/STOP combined: pre-computed (per-slot, same every slot) ──
    valid_t = torch.tensor(valid_targets, dtype=torch.long, device=device)
    combined = _build_tgt_combined(target_scores, stop_logits, valid_t, 0)
    tgt_logp = combined.log_softmax(dim=-1).nan_to_num(0.0)  # (S, PAD)
    safe_tgt = tgt_logp.masked_fill(tgt_logp == fill_val, 0.0)
    tgt_ent = -(safe_tgt.exp() * safe_tgt).sum(dim=-1)       # (S,)

    # ── Offset logprobs & entropy ──
    offset_logp = offset_logits.log_softmax(dim=-1).nan_to_num(0.0)
    safe_off = offset_logp.masked_fill(offset_logp == fill_val, 0.0)
    offset_ent = -(safe_off.exp() * safe_off).sum(dim=-1)    # (S,)

    # ── GPU random numbers (one shot) ──────────────────────────────
    owned_idx = ownership_mask.nonzero(as_tuple=False).squeeze(-1)
    n_owned = len(owned_idx)

    rand_source = torch.rand(S, device=device)
    rand_offset = torch.rand(S, device=device)
    rand_tgt    = torch.rand(S, L, device=device)
    rand_source_int = torch.randint(0, max(n_owned, 1), (S,), device=device)
    rand_offset_int = torch.randint(0, n_offs, (S,), device=device)
    rand_tgt_int    = torch.randint(0, max(len(valid_targets), 0) + 1, (S, L), device=device)

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

        # ── 3. Offset selection (per source) ──
        if eps > 0 and rand_offset[s].item() < eps:
            oi = int(rand_offset_int[s].item())
        else:
            oi = int(torch.argmax(offset_logits[s]).item())

        offset_indices[s] = oi
        lp += offset_logp[s, oi]
        ent += offset_ent[s]
        offset_val = offset_bins[oi]

        src_p = planets[src_idx] if planets is not None else None

        # ── 4. Collect targets, then allocate ships ──
        selected = []  # (real_tgt_idx, tgt_ships)
        for ll in range(L):
            if eps > 0 and rand_tgt[s, ll].item() < eps:
                tgt_cat = int(rand_tgt_int[s, ll].item())
            else:
                tgt_cat = int(torch.argmax(combined[s]).item())

            if tgt_cat == 0:
                lp += tgt_logp[s, tgt_cat]
                ent += tgt_ent[s]
                action_indices[s, ll] = tgt_cat
                break

            real_tgt_idx = valid_targets[tgt_cat - 1]

            # Sun check — use a small probe fleet for collision test.
            if planets is not None and src_p is not None:
                tgt_p = planets[real_tgt_idx]
                probe_ships = max(1, rem // 2)
                speed = _fleet_speed(probe_ships)
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
                    action_indices[s, ll] = -1  # sun-blocked marker
                    continue

            lp += tgt_logp[s, tgt_cat]
            ent += tgt_ent[s]
            action_indices[s, ll] = tgt_cat
            selected.append((real_tgt_idx, tgt_p.ships if planets is not None else 0))

        # Source depleted for this step
        remaining_ships[src_idx] = 0

    return action_indices, source_indices, offset_indices, lp, ent


# ── PPO logprob (with gradients) ──────────────────────────────────

def logprob_batched_combined(
    src_logits,
    tgt_scores,
    stop_logits,
    offset_logits,
    ownership_mask,
    action_indices,            # (B, S, L) — tgt_cat per slot per launch
    source_indices,            # list of (S,) tensors, one per batch element
    offset_indices,            # list of (S,) tensors, one per batch element
    all_valid_targets,
    all_planet_ships,          # list of dicts {planet_idx: ships}
    max_launches,
    offset_bins,
    all_planets=None,          # list of Planet lists (for sun collision; optional)
    all_orbit_lookups=None,    # list of orbit lookup dicts
    all_ang_vels=None,         # list of float angular_velocities
):
    """Batched log-probability + entropy for PPO update with offset scheme.

    Ship allocation matches ``sample_action_discrete`` exactly:
    ``target_ships + 1 + offset`` per target, proportional scaling if
    total need exceeds available ships.
    """
    B, S, N = tgt_scores.shape
    L = max_launches
    device = tgt_scores.device
    fill_val = float('-inf')

    src_idx_batch = torch.stack(source_indices, dim=0).to(device)  # (B, S)
    off_idx_batch = torch.stack(offset_indices, dim=0).to(device)  # (B, S)
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

    # ── 3. Offset logprobs & entropy ──
    offset_logp = offset_logits.log_softmax(dim=-1).nan_to_num(0.0)  # (B, S, nO)
    safe_olp = offset_logp.masked_fill(offset_logp == fill_val, 0.0)
    offset_ents = -(safe_olp.exp() * safe_olp).sum(dim=-1).nan_to_num(0.0)  # (B, S)

    # ── 4. Target replay (stateful per-slot loop) ──
    tgt_lps  = torch.zeros(B, device=device)
    tgt_ents = torch.zeros(B, device=device)
    off_lps = torch.zeros(B, device=device)
    off_ents_sum = torch.zeros(B, device=device)
    source_count = torch.zeros(B, device=device)

    for b in range(B):
        valid = all_valid_targets[b]
        act_b = action_indices[b]                              # (S, L)
        tgt_cats = act_b                                        # (S, L)
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

            # ── Offset logprob ──
            oi = off_idx_batch[b, s].item()
            off_lps[b] += offset_logp[b, s, oi]
            off_ents_sum[b] += offset_ents[b, s]

            rem = all_planet_ships[b].get(si, 0)
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

                # Sun collision guard — uses a probe fleet for collision test.
                if planets is not None and src_p is not None:
                    tgt_p = planets[ti]
                    probe_ships = max(1, rem // 2)
                    speed = _fleet_speed(probe_ships)
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

    return (src_lps + off_lps + tgt_lps,
            off_ents_sum + tgt_ents + src_ents,
            source_count)
