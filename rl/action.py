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
    step: int,
) -> float:
    """Compute firing angle to intercept a (possibly orbiting) target.

    For static planets the angle is simply the bearing to the current
    position.  For rotating planets an iterative predictor–corrector
    loop (max 10 iterations) converges on the intercept angle.
    """
    info = orbit_lookup.get(tgt_id) if orbit_lookup else None
    if info is None or not info['rotates'] or angular_velocity == 0:
        return math.atan2(tgt_y - src_y, tgt_x - src_x)

    r = info['orbital_r']
    cx, cy = _SUN_CENTER

    # Current angular position of the target
    theta = info['initial_angle'] + angular_velocity * step

    # Current cartesian position (should match tgt_x, tgt_y)
    cur_tx = cx + r * math.cos(theta)
    cur_ty = cy + r * math.sin(theta)

    # Iterative refinement — usually converges in 2–3 steps
    angle = math.atan2(cur_ty - src_y, cur_tx - src_x)
    for _ in range(10):
        dist = math.hypot(cur_tx - src_x, cur_ty - src_y)
        flight_time = dist / max(fleet_speed, 0.01)

        future_theta = theta + angular_velocity * flight_time
        future_tx = cx + r * math.cos(future_theta)
        future_ty = cy + r * math.sin(future_theta)

        new_angle = math.atan2(future_ty - src_y, future_tx - src_x)
        if abs(new_angle - angle) < 1e-6:
            break
        angle = new_angle
        cur_tx, cur_ty = future_tx, future_ty

    return angle


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
        non_my = [i for i, p in enumerate(planets) if p.owner != player_id]
        return planets, [i for i, _ in my], non_my, player_id

    def decode(self, planet_by_idx, src_obs_idx, tgt_obs_idx, frac_idx, remaining,
               orbit_lookup=None, angular_velocity=0.0, step=0):
        src = planet_by_idx[src_obs_idx]
        tgt = planet_by_idx[tgt_obs_idx]
        frac = self.ship_fractions[frac_idx]
        ships = max(1, min(int(remaining * frac), remaining))

        # Intercept angle: compensate for target orbital motion
        speed = _fleet_speed(ships)
        angle = compute_intercept_angle(
            src_x=src.x, src_y=src.y,
            tgt_id=tgt.id, tgt_x=tgt.x, tgt_y=tgt.y,
            fleet_speed=speed,
            orbit_lookup=orbit_lookup or {},
            angular_velocity=angular_velocity,
            step=step,
        )

        if trajectory_hits_sun(src.x, src.y, angle=angle,
                                tgt_x=tgt.x, tgt_y=tgt.y):
            return None
        return [src.id, angle, ships]

    def decode_all(self, planets, action_indices, source_indices,
                   planet_ships, valid_targets,
                   orbit_lookup=None, angular_velocity=0.0, step=0):
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
                if tgt_cat <= 0:
                    break
                real_tgt = valid_targets[tgt_cat - 1]
                frac_idx = int(action_indices[s, ll, 1].item())
                move = self.decode(planets, src_idx, real_tgt, frac_idx, rem,
                                   _ol, angular_velocity, step)
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
):
    """Per-slot action sampling: source → target → fraction.

    ε-greedy exploration: random uniform with prob ε, argmax with prob 1-ε.
    Logprobs always use the model's softmax distribution.
    Ships are only consumed when the trajectory does NOT hit the sun,
    matching the actual execution in ``decode_all``.

    **Sequential source masking**: a planet is masked from subsequent slots
    when its remaining ships drop below 15% of its original count.  This
    prevents all slots from draining a single planet while allowing
    meaningful multi-slot use of a large source.

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
    _DRAIN_THRESHOLD = 0.15  # mask source when remaining < 15% of original

    action_indices = torch.zeros((S, L, 2), dtype=torch.long, device=device)
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

    # Base source logits — ownership mask only (drain mask added per slot)
    src_base = source_logits.masked_fill(~ownership_mask.unsqueeze(0), fill_val)

    # ── Ship state (Python) for sequential consumption ─────────────
    remaining_ships = dict(planet_ships)

    for s in range(min(S, len(planet_ships))):
        if n_owned == 0:
            continue

        # ── 0. Build drain mask: planets drained below threshold by prior slots ──
        drain_mask = torch.zeros(N, dtype=torch.bool, device=device)
        for prev_s in range(s):
            prev_si = source_indices[prev_s].item()
            initial = planet_ships.get(prev_si, 0)
            if initial <= 0:
                continue
            rem = remaining_ships.get(prev_si, 0)
            if rem < initial * _DRAIN_THRESHOLD:
                drain_mask[prev_si] = True

        # ── 1. Source logprobs & entropy for THIS slot (recomputed) ──
        src_s = src_base[s].masked_fill(drain_mask, fill_val)
        src_logp_s = src_s.log_softmax(dim=-1).nan_to_num(0.0)
        safe_src_s = src_logp_s.masked_fill(src_logp_s == fill_val, 0.0)
        src_ent_s = -(safe_src_s.exp() * safe_src_s).sum(dim=-1)

        # ── 2. Source selection ──
        #     If all owned planets are drained, stop assigning slots.
        n_avail_source = (~drain_mask[owned_idx]).sum().item() if n_owned > 0 else 0
        if n_avail_source == 0:
            break

        if eps > 0 and rand_source[s].item() < eps:
            avail = owned_idx[~drain_mask[owned_idx]]
            src_idx = int(avail[rand_source_int[s].item() % len(avail)].item())
        else:
            src_idx = int(torch.argmax(src_s).item())

        source_indices[s] = src_idx
        lp += src_logp_s[src_idx]
        ent += src_ent_s

        rem = remaining_ships.get(src_idx, 0)
        if rem <= 0:
            remaining_ships[src_idx] = 0  # mark depleted for later slots
            continue

        # Resolve source planet coordinates once
        src_p = planets[src_idx] if planets is not None else None

        for ll in range(L):
            # 3. Target / STOP selection
            if eps > 0 and rand_tgt[s, ll].item() < eps:
                tgt_cat = int(rand_tgt_int[s, ll].item())
            else:
                tgt_cat = int(torch.argmax(combined[s]).item())

            lp += tgt_logp[s, tgt_cat]
            ent += tgt_ent[s]

            action_indices[s, ll, 0] = tgt_cat
            if tgt_cat == 0:
                break

            real_tgt_idx = valid_targets[tgt_cat - 1]

            # 4. Fraction selection
            f_logits = frac_logits_all[s, real_tgt_idx]
            f_logp = f_logits.log_softmax(dim=-1).nan_to_num(0.0)

            if eps > 0 and rand_frac[s, ll].item() < eps:
                fa = int(rand_frac_int[s, ll].item())
            else:
                fa = int(torch.argmax(f_logits).item())

            lp += f_logp[fa]
            safe_flp = f_logp.masked_fill(f_logp == fill_val, 0.0)
            ent -= (safe_flp.exp() * safe_flp).sum()

            action_indices[s, ll, 1] = fa

            _f = ship_fractions[fa]
            ships = max(1, min(int(rem * _f), rem))

            # Sun collision guard — must use the *same intercept angle*
            # that ``decode`` will use.  Otherwise a launch that passes
            # the check here may still be cancelled in ``decode`` (or
            # vice-versa), causing mismatched ship counts.
            if planets is not None and src_p is not None:
                tgt_p = planets[real_tgt_idx]
                if trajectory_hits_sun(src_p.x, src_p.y, tgt_x=tgt_p.x, tgt_y=tgt_p.y):
                    continue  # ← keeps `rem` unchanged, ll still advances

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
):
    """Batched log-probability + entropy for PPO update.

    Ship consumption during replay matches actual execution:
    sun-colliding launches do NOT consume ships.

    Source logprobs are recomputed per-slot with sequential drain
    masking (15% threshold), matching the behaviour of ``sample_action_discrete``.
    """
    B, S, N = tgt_scores.shape
    L = max_launches
    device = tgt_scores.device
    fill_val = float('-inf')
    _DRAIN_THRESHOLD = 0.15

    src_idx_batch = torch.stack(source_indices, dim=0).to(device)  # (B, S)
    src_base = src_logits.masked_fill(~ownership_mask.unsqueeze(1), fill_val)  # (B, S, N)

    # ── 1. Source logprob & entropy — per-slot with sequential drain ──
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
        ships_initial = dict(all_planet_ships[b])     # immutable — used for drain threshold
        remaining = dict(all_planet_ships[b])          # mutable — tracks consumption
        act_b = action_indices[b]                            # (S, L, 2)
        tgt_cats = act_b[:, :, 0]                            # (S, L)
        frac_idxs = act_b[:, :, 1]                           # (S, L)
        planets = all_planets[b] if all_planets else None

        for s in range(S):
            # ── 0. Source logprob with sequential drain mask ──
            drain_mask = torch.zeros(N, dtype=torch.bool, device=device)
            for prev_s in range(s):
                prev_si = src_idx_batch[b, prev_s].item()
                initial = ships_initial.get(prev_si, 0)
                if initial <= 0:
                    continue
                if remaining.get(prev_si, 0) < initial * _DRAIN_THRESHOLD:
                    drain_mask[prev_si] = True

            # ── All sources drained → stop (matches sampling behaviour) ──
            owned_avail = (ownership_mask[b] & ~drain_mask).sum().item()
            if owned_avail == 0:
                break

            src_bs = src_base[b, s].masked_fill(drain_mask, fill_val)
            src_logp_bs = src_bs.log_softmax(dim=-1).nan_to_num(0.0)
            safe_src_bs = src_logp_bs.masked_fill(src_logp_bs == fill_val, 0.0)
            src_ents[b] += -(safe_src_bs.exp() * safe_src_bs).sum()

            si = src_idx_batch[b, s].item()
            src_lps[b] += src_logp_bs[si]

            rem = remaining.get(si, 0)
            if rem <= 0:
                continue
            source_count[b] += 1

            if planets is not None:
                src_p = planets[si]
            else:
                src_p = None

            for ll in range(L):
                tc = int(tgt_cats[s, ll].item())
                tgt_lps[b] += all_logp[b, s, tc]
                tgt_ents[b] += all_ent[b, s]
                if tc == 0:
                    break
                ti = valid[tc - 1]
                fl = frac_logits_all[b, s, ti].nan_to_num(0.0)
                flp = fl.log_softmax(dim=-1).nan_to_num(0.0)
                fi = int(frac_idxs[s, ll].item())
                frac_lps[b] += flp[fi]
                safe_flp = flp.masked_fill(flp == fill_val, 0.0)
                frac_ents[b] -= (safe_flp.exp() * safe_flp).sum()
                _f = ship_fractions[fi]
                ships = max(1, min(int(rem * _f), rem))

                # Sun collision guard — match decode_all behaviour
                if planets is not None and src_p is not None:
                    tgt_p = planets[ti]
                    if trajectory_hits_sun(src_p.x, src_p.y, tgt_x=tgt_p.x, tgt_y=tgt_p.y):
                        continue  # no ship consumption

                rem -= ships
                if rem <= 0:
                    break

            remaining[si] = rem

    return (src_lps + tgt_lps + frac_lps,
            tgt_ents + frac_ents + src_ents,
            source_count)
