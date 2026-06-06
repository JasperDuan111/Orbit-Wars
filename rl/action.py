import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch.distributions import Categorical
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
from .config import ActionSpaceConfig, DEFAULT_CONFIG


def _get_field(obs, name, default=None):
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def _parse_planets(obs):
    raw_planets = _get_field(obs, "planets", [])
    return [Planet(*p) for p in raw_planets]


class ActionBuilder:
    def __init__(self, action_config: Optional[ActionSpaceConfig] = None):
        self.config = action_config or DEFAULT_CONFIG.action
        self.actions_per_source = self.config.actions_per_source
        self.max_sources = self.config.max_sources

    def build(self, obs, source_planet_ids: Optional[Sequence[int]] = None):
        """Build action templates for source planets.

        If source_planet_ids is None, falls back to ship-count ordering
        (used by MLP model or when model doesn't output source logits).
        """
        player_id = _get_field(obs, "player", 0)
        planets = _parse_planets(obs)

        # Pre-split planets by ownership (done once, not per-source).
        my_planets_all = [p for p in planets if p.owner == player_id]
        my_planets_all.sort(key=lambda p: p.ships, reverse=True)
        enemy_neutral = [p for p in planets if p.owner != player_id]

        if source_planet_ids is not None and len(source_planet_ids) > 0:
            # Use model-selected source planets
            planet_by_id = {p.id: p for p in planets}
            my_planets = []
            used_ids = set()
            for pid in source_planet_ids:
                p = planet_by_id.get(int(pid))
                if p is not None and p.owner == player_id:
                    my_planets.append(p)
                    used_ids.add(p.id)
            # Pad with remaining owned planets (already ship-sorted)
            for p in my_planets_all:
                if p.id not in used_ids:
                    my_planets.append(p)
        else:
            # Fallback: ship-count ordering
            my_planets = my_planets_all

        max_sources = self.max_sources
        max_targets = self.config.max_targets
        aps = self.actions_per_source

        actions: List[List[Optional[ActionTemplate]]] = []
        source_ships: List[int] = []

        for i in range(max_sources):
            source_actions: List[Optional[ActionTemplate]] = [None] * aps

            if i >= len(my_planets):
                actions.append(source_actions)
                source_ships.append(0)
                continue

            src = my_planets[i]
            source_ships.append(int(src.ships))
            sx, sy = src.x, src.y

            # Two-group distance sort: own planets first, then enemy/neutral.
            own_distances = []
            other_distances = []
            for p in my_planets_all:
                if p.id != src.id:
                    own_distances.append((math.hypot(sx - p.x, sy - p.y), p))
            for p in enemy_neutral:
                other_distances.append((math.hypot(sx - p.x, sy - p.y), p))

            own_distances.sort(key=lambda x: x[0])
            other_distances.sort(key=lambda x: x[0])

            ordered_targets = [p for _, p in own_distances] + [p for _, p in other_distances]

            if not ordered_targets:
                actions.append(source_actions)
                continue

            # slot 0 = STOP, slots 1..max_targets = target selection (no fraction)
            for j in range(min(max_targets, len(ordered_targets))):
                tgt = ordered_targets[j]
                angle = math.atan2(tgt.y - sy, tgt.x - sx)
                source_actions[j + 1] = ActionTemplate(src.id, angle)

            actions.append(source_actions)

        return actions, source_ships

    def decode(
        self,
        action_indices,
        actions: Sequence[Sequence[Optional["ActionTemplate"]]],
        source_ships: Sequence[int],
        max_launches: Optional[int] = None,
        fractions: Optional[Sequence[Sequence[float]]] = None,
    ):
        """Decode action indices into game moves.

        Args:
            action_indices: (max_sources, max_launches) discrete target indices.
            actions: per-source list of ActionTemplate slots.
            source_ships: per-source available ship counts.
            max_launches: max launch steps per source.
            fractions: (max_sources, max_launches) continuous fraction values in [0, 1].
                       If None, defaults to 1.0 for all launches.
        """
        if action_indices is None:
            return []
        if hasattr(action_indices, "tolist"):
            action_indices = action_indices.tolist()

        if max_launches is None:
            max_launches = self.config.max_launches_per_source

        max_src = min(len(actions), self.max_sources)
        moves = []

        for src_idx in range(max_src):
            remaining = int(source_ships[src_idx]) if src_idx < len(source_ships) else 0
            if remaining <= 0:
                continue

            src_acts = action_indices[src_idx]
            source_actions = actions[src_idx]
            n_steps = min(max_launches, len(src_acts))

            for step_idx in range(n_steps):
                try:
                    action_idx = int(src_acts[step_idx])
                except (TypeError, ValueError, IndexError):
                    break
                if action_idx <= 0 or action_idx >= len(source_actions):
                    break
                action = source_actions[action_idx]
                if action is None:
                    break
                # Use continuous fraction from model output, default to 1.0
                frac = 1.0
                if fractions is not None and src_idx < len(fractions):
                    src_fracs = fractions[src_idx]
                    if step_idx < len(src_fracs):
                        frac = float(src_fracs[step_idx])
                ships = _ships_to_send(remaining, frac)
                if ships <= 0:
                    break
                moves.append([action.source_id, action.angle, ships])
                remaining -= ships
                if remaining <= 0:
                    break

        return moves


@dataclass(frozen=True, slots=True)
class ActionTemplate:
    source_id: int
    angle: float


# ── Source selection (Gumbel top-k for training, argmax for inference) ──

def select_sources(source_logits: torch.Tensor, ownership_mask: torch.Tensor,
                   k: int, deterministic: bool = False) -> Optional[torch.Tensor]:
    """Select k source planets from per-planet source scores.

    Args:
        source_logits: (S, N) per-slot logits over N planets, or (S, 1) for MLP fallback.
        ownership_mask: (N,) boolean mask, True = owned planets.
        k: number of sources to select.
        deterministic: if True, use argmax; otherwise Gumbel top-k.

    Returns:
        source_indices: (k,) long tensor of selected planet indices,
        or None if source_logits is not per-planet (MLP fallback).
    """
    # MLP fallback: source_logits has shape (S, 1) — no per-planet information
    if source_logits.shape[-1] <= 1:
        return None

    device = source_logits.device
    N = source_logits.shape[-1]
    k = min(k, N)

    # Average over slot dimension to get a single per-planet score
    planet_scores = source_logits.mean(dim=0)  # (N,)

    if deterministic:
        masked = planet_scores.masked_fill(~ownership_mask, float('-inf'))
        _, indices = torch.topk(masked, k=k, dim=-1)
    else:
        gumbel = -torch.log(-torch.log(torch.rand(N, device=device) + 1e-10) + 1e-10)
        gumbel_scores = planet_scores + gumbel
        gumbel_scores = gumbel_scores.masked_fill(~ownership_mask, float('-inf'))
        _, indices = torch.topk(gumbel_scores, k=k, dim=-1)

    return indices  # (k,)


def source_selection_logprob(source_logits: torch.Tensor, ownership_mask: torch.Tensor,
                             source_indices: Optional[torch.Tensor]) -> torch.Tensor:
    """Plackett-Luce log-probability of selecting source planets in order.

    Args:
        source_logits: (S, N) or (S, 1) for MLP fallback.
        ownership_mask: (N,) boolean mask.
        source_indices: (k,) long tensor of selected planet indices, or None.

    Returns:
        Scalar log-probability (sum over selected items).
    """
    if source_indices is None or source_logits.shape[-1] <= 1:
        return torch.zeros((), device=source_logits.device)

    device = source_logits.device
    planet_scores = source_logits.mean(dim=0)  # (N,)
    k = source_indices.shape[0]

    logprob = torch.zeros((), device=device)
    remaining_mask = ownership_mask.clone()

    for i in range(k):
        idx = source_indices[i].item()
        if not remaining_mask[idx]:
            continue  # shouldn't happen, but guard against invalid indices
        guarded_scores = planet_scores.nan_to_num(0.0)
        masked_logits = guarded_scores.masked_fill(~remaining_mask, float('-inf'))
        log_prob = masked_logits.log_softmax(dim=-1)
        logprob = logprob + log_prob[idx]
        remaining_mask[idx] = False

    return logprob


# ── Action sequence sampling / logprob ──

def _build_source_mask(
    source_actions: Sequence[Optional[ActionTemplate]],
    n_slots: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a static validity mask for a source's action slots.

    Slots with non-None templates are valid (mask = 1.0), invalid slots are
    masked out (0.0).  Slot 0 (STOP) is always valid.  The mask is identical
    for every launch step as long as *remaining > 0* — it only depends on
    which slots are filled, not on the concrete remaining-ship count.

    Returns a float32 tensor of shape (n_slots,).
    """
    mask = torch.zeros(n_slots, dtype=torch.float32, device=device)
    mask[0] = 1.0
    # Valid slots are contiguous from idx=1 (ActionBuilder fills them in order).
    count = 0
    for i in range(1, n_slots):
        if source_actions[i] is not None:
            count += 1
        else:
            break
    if count > 0:
        mask[1:count + 1] = 1.0
    return mask


def _mask_fill_value(dtype: torch.dtype) -> float:
    if dtype.is_floating_point:
        return torch.finfo(dtype).min
    return -1e9


def sample_action_sequence(
    logits: torch.Tensor,
    actions: Sequence[Sequence[Optional[ActionTemplate]]],
    source_ships: Sequence[int],
    max_launches: Optional[int] = None,
    deterministic: bool = False,
    action_config: Optional[ActionSpaceConfig] = None,
    fractions: Optional[torch.Tensor] = None,
):
    """Sample target actions and return action indices + logprobs.

    Args:
        logits: (S, L, A) per-slot per-launch target logits.
        actions: per-source action template lists.
        source_ships: per-source available ships.
        max_launches: max launch steps per source.
        deterministic: if True, use argmax instead of sampling.
        action_config: action space config.
        fractions: (S, L) continuous fraction tensor in [0, 1].
                   If None, defaults to 1.0.

    Returns:
        action_indices: (S, L) long tensor.
        logprob_sum: scalar sum of log-probs.
        entropy_sum: scalar sum of entropies.
        fractions_out: (S, L) fraction values used (same as input or defaults).
    """
    device = logits.device
    config = action_config or DEFAULT_CONFIG.action
    if max_launches is None:
        max_launches = config.max_launches_per_source

    S = config.max_sources
    L = max_launches
    action_indices = torch.zeros((S, L), dtype=torch.long, device=device)
    fractions_out = torch.ones((S, L), device=device)
    logprob_sum = torch.zeros((), device=device)
    entropy_sum = torch.zeros((), device=device)

    n_src = min(S, len(actions))
    for src_idx in range(n_src):
        remaining = int(source_ships[src_idx]) if src_idx < len(source_ships) else 0
        if remaining <= 0:
            continue
        source_actions = actions[src_idx]
        n_slots = len(source_actions)

        # Build mask ONCE per source: same validity for all launches
        mask = _build_source_mask(source_actions, n_slots, device)  # (A,)

        for step_idx in range(L):
            # Each launch step has its own logits
            src_logits_l = logits[src_idx, step_idx].to(torch.float32).nan_to_num(0.0)
            masked_logits = src_logits_l.masked_fill(mask == 0, float('-inf'))
            dist = Categorical(logits=masked_logits)
            entropy_sum = entropy_sum + dist.entropy()

            if deterministic:
                action = torch.argmax(masked_logits)
            else:
                action = dist.sample()
            action_indices[src_idx, step_idx] = action
            logprob_sum = logprob_sum + dist.log_prob(action)

            action_idx = int(action.item())
            if action_idx <= 0:
                break
            template = source_actions[action_idx]
            if template is None:
                break
            # Use continuous fraction from model output
            frac = 1.0
            if fractions is not None and src_idx < fractions.shape[0] and step_idx < fractions.shape[1]:
                frac = float(fractions[src_idx, step_idx].item())
            fractions_out[src_idx, step_idx] = frac
            ships = _ships_to_send(remaining, frac)
            if ships <= 0:
                break
            remaining -= ships
            if remaining <= 0:
                break

    return action_indices, logprob_sum, entropy_sum, fractions_out


def logprob_for_action_sequence(
    logits: torch.Tensor,
    actions: Sequence[Sequence[Optional[ActionTemplate]]],
    source_ships: Sequence[int],
    action_indices: torch.Tensor,
    max_launches: Optional[int] = None,
    action_config: Optional[ActionSpaceConfig] = None,
    fractions: Optional[torch.Tensor] = None,
):
    """Compute log-probability and entropy for a given action sequence.

    Fraction values do not contribute to logprob (treated as deterministic).
    
    Args:
        logits: (S, L, A) per-slot per-launch target logits.
    """
    device = logits.device
    config = action_config or DEFAULT_CONFIG.action
    if max_launches is None:
        max_launches = config.max_launches_per_source
    logprob_sum = torch.zeros((), device=device)
    entropy_sum = torch.zeros((), device=device)

    if hasattr(action_indices, "to"):
        action_indices = action_indices.to(device)

    n_src = min(config.max_sources, len(actions))
    for src_idx in range(n_src):
        remaining = int(source_ships[src_idx]) if src_idx < len(source_ships) else 0
        if remaining <= 0:
            continue
        source_actions = actions[src_idx]
        n_slots = len(source_actions)

        mask = _build_source_mask(source_actions, n_slots, device)

        for step_idx in range(max_launches):
            try:
                action_idx = int(action_indices[src_idx, step_idx].item())
            except (TypeError, ValueError, IndexError):
                break
            # Each launch step has its own logits
            src_logits_l = logits[src_idx, step_idx].to(torch.float32).nan_to_num(0.0)
            masked_logits = src_logits_l.masked_fill(mask == 0, float('-inf'))
            dist = Categorical(logits=masked_logits)
            entropy_sum = entropy_sum + dist.entropy()
            logprob_sum = logprob_sum + dist.log_prob(action_indices[src_idx, step_idx])

            if action_idx <= 0:
                break
            template = source_actions[action_idx]
            if template is None:
                break
            frac = 1.0
            if fractions is not None and src_idx < fractions.shape[0] and step_idx < fractions.shape[1]:
                frac = float(fractions[src_idx, step_idx].item())
            ships = _ships_to_send(remaining, frac)
            if ships <= 0:
                break
            remaining -= ships
            if remaining <= 0:
                break

    return logprob_sum, entropy_sum


def _ships_to_send(remaining_ships: int, fraction: float) -> int:
    if remaining_ships <= 0:
        return 0
    ships = int(remaining_ships * float(fraction))
    return max(1, min(ships, remaining_ships))


# ── Batched logprob computation (PPO update) ──

def _compute_max_valid_idx(
    templates_list: Sequence[Sequence[Sequence[Optional[ActionTemplate]]]],
    n_sources: int,
    n_slots: int,
) -> torch.Tensor:
    """Compute max valid action index per (batch, source) from template lists.

    Returns:
        (B, S) LongTensor on CPU.  Value = last valid index (0-based), or 0 for empty sources.
    """
    B = len(templates_list)
    result = torch.zeros(B, n_sources, dtype=torch.long)
    for b in range(B):
        for s in range(n_sources):
            src = templates_list[b][s] if s < len(templates_list[b]) else []
            count = 1  # slot 0 (STOP) always valid
            for i in range(1, min(len(src), n_slots)):
                if src[i] is not None:
                    count += 1
                else:
                    break
            result[b, s] = count - 1
    return result


def _compute_step_valid(
    action_indices: torch.Tensor,
    source_ships_list: Sequence[Sequence[int]],
    fractions: torch.Tensor,
    max_launches: int,
) -> torch.Tensor:
    """CPU simulation of autoregressive break logic for every (batch, source).

    Args:
        fractions: (B, S, L) continuous fraction tensor in [0, 1].

    Returns:
        (B, S, L) BoolTensor on CPU.  True = this launch step actually executed.
    """
    B, n_sources, L = action_indices.shape
    result = torch.zeros(B, n_sources, L, dtype=torch.bool)

    for b in range(B):
        for s in range(n_sources):
            remaining = source_ships_list[b][s] if s < len(source_ships_list[b]) else 0
            if remaining <= 0:
                continue
            for l in range(L):
                result[b, s, l] = True

                aidx = int(action_indices[b, s, l].item())
                if aidx <= 0:
                    break
                frac = float(fractions[b, s, l].item()) if fractions is not None else 1.0
                if frac <= 0:
                    break
                ships = max(1, min(int(remaining * frac), remaining))
                if ships <= 0:
                    break
                remaining -= ships
                if remaining <= 0:
                    break

    return result


def logprob_for_action_sequence_batched(
    slot_logits: torch.Tensor,
    max_valid_idx: torch.Tensor,
    action_indices: torch.Tensor,
    step_valid: torch.Tensor,
):
    """Batch log-probability and entropy for all samples in a single GPU pass.

    Note: fraction values do NOT contribute to logprob (treated as deterministic).
    Only target selection contributes to the policy gradient.

    Args:
        slot_logits:   (B, S, L, A) per-slot per-launch logits (target selection only).
        max_valid_idx: (B, S)   max valid action index per source (CPU, moved to GPU).
        action_indices:(B, S, L) discrete action indices.
        step_valid:    (B, S, L) bool mask: which steps actually executed (CPU→GPU).

    Returns:
        logprob_sum: (B,)  sum of log-probs over valid steps.
        entropy_sum: (B,)  sum of entropies over active sources.
        source_active_count: (B,) number of active sources per sample.
    """
    B, S, L, A = slot_logits.shape
    device = slot_logits.device
    needs_cast = slot_logits.dtype == torch.float16

    max_valid_idx = max_valid_idx.to(device).clamp(min=0)  # (B, S)
    step_valid = step_valid.to(device)                       # (B, S, L)

    # Build mask: position <= max_valid_idx → valid
    # Expand max_valid_idx from (B, S) to (B, S, 1, 1) for proper broadcasting
    positions = torch.arange(A, device=device).view(1, 1, 1, A)          # (1, 1, 1, A)
    max_valid_4d = max_valid_idx[:, :, None, None]                        # (B, S, 1, 1)
    mask = (positions <= max_valid_4d).expand(-1, -1, L, -1)             # (B, S, L, A)

    # Force float32 and guard against any upstream NaN in logits.
    logits_f32 = slot_logits.to(torch.float32).nan_to_num(0.0)
    fill_value = _mask_fill_value(torch.float32)
    masked_logits = logits_f32.masked_fill(~mask, fill_value)

    # One Categorical call for the entire batch: (B*S*L, A)
    flat_logits = masked_logits.reshape(B * S * L, A)
    dist = Categorical(logits=flat_logits)

    flat_actions = action_indices.reshape(B * S * L)  # (B*S*L,)
    all_logprobs = dist.log_prob(flat_actions).reshape(B, S, L)  # (B, S, L)
    entropy = dist.entropy().reshape(B, S, L)                     # (B, S, L)

    # Mask entropy to only active (source, launch) pairs
    step_valid_float = step_valid.float()               # (B, S, L)
    logprob_sum = (all_logprobs * step_valid_float).sum(dim=(1, 2))  # (B,)
    if needs_cast:
        logprob_sum = logprob_sum.half()

    # Per-source active count for entropy normalization
    source_active = step_valid.any(dim=-1).float()      # (B, S)
    entropy_sum = (entropy * step_valid_float).sum(dim=(1, 2))  # (B,)
    source_active_count = source_active.sum(dim=1)      # (B,)

    return logprob_sum, entropy_sum, source_active_count
