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

        if source_planet_ids is not None and len(source_planet_ids) > 0:
            # Use model-selected source planets
            planet_by_id = {p.id: p for p in planets}
            my_planets = []
            for pid in source_planet_ids:
                p = planet_by_id.get(int(pid))
                if p is not None and p.owner == player_id:
                    my_planets.append(p)
            # Pad with ship-count sorted remaining owned planets
            used_ids = {p.id for p in my_planets}
            remaining = sorted(
                [p for p in planets if p.owner == player_id and p.id not in used_ids],
                key=lambda p: p.ships, reverse=True,
            )
            my_planets.extend(remaining)
        else:
            # Fallback: ship-count ordering
            my_planets = [p for p in planets if p.owner == player_id]
            my_planets.sort(key=lambda p: p.ships, reverse=True)

        actions: List[List[Optional[ActionTemplate]]] = []
        source_ships: List[int] = []

        for i in range(self.max_sources):
            source_actions: List[Optional[ActionTemplate]] = [None] * self.actions_per_source

            if i >= len(my_planets):
                actions.append(source_actions)
                source_ships.append(0)
                continue

            src = my_planets[i]
            source_ships.append(int(src.ships))

            def _target_priority(tgt):
                is_own = 1 if tgt.owner == player_id else 0
                dist = math.hypot(src.x - tgt.x, src.y - tgt.y)
                return (-is_own, dist)

            ordered_targets = sorted(
                (p for p in planets if p.id != src.id),
                key=_target_priority,
            )
            if not ordered_targets:
                actions.append(source_actions)
                continue

            idx = 1
            for j in range(self.config.max_targets):
                if j >= len(ordered_targets):
                    idx += len(self.config.ship_fractions)
                    continue
                tgt = ordered_targets[j]
                angle = math.atan2(tgt.y - src.y, tgt.x - src.x)
                for frac in self.config.ship_fractions:
                    source_actions[idx] = ActionTemplate(src.id, angle, float(frac))
                    idx += 1

            actions.append(source_actions)

        return actions, source_ships

    def decode(
        self,
        action_indices,
        actions: Sequence[Sequence[Optional["ActionTemplate"]]],
        source_ships: Sequence[int],
        max_launches: Optional[int] = None,
    ):
        if action_indices is None:
            return []
        if hasattr(action_indices, "tolist"):
            action_indices = action_indices.tolist()

        if max_launches is None:
            max_launches = self.config.max_launches_per_source

        moves = []
        for src_idx in range(min(len(actions), self.max_sources)):
            remaining = int(source_ships[src_idx]) if src_idx < len(source_ships) else 0
            if remaining <= 0:
                continue
            for step_idx in range(min(max_launches, len(action_indices[src_idx]))):
                try:
                    action_idx = int(action_indices[src_idx][step_idx])
                except (TypeError, ValueError, IndexError):
                    break
                if action_idx <= 0:
                    break
                source_actions = actions[src_idx]
                if action_idx >= len(source_actions):
                    break
                action = source_actions[action_idx]
                if action is None:
                    break
                ships = _ships_to_send(remaining, action.fraction)
                if ships <= 0:
                    break
                moves.append([action.source_id, action.angle, ships])
                remaining -= ships
                if remaining <= 0:
                    break

        return moves


@dataclass(frozen=True)
class ActionTemplate:
    source_id: int
    angle: float
    fraction: float


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
        masked_logits = planet_scores.masked_fill(~remaining_mask, float('-inf'))
        log_prob = masked_logits.log_softmax(dim=-1)
        logprob = logprob + log_prob[idx]
        remaining_mask[idx] = False

    return logprob


# ── Action sequence sampling / logprob (unmodified slot-level logic) ──

def sample_action_sequence(
    logits: torch.Tensor,
    actions: Sequence[Sequence[Optional[ActionTemplate]]],
    source_ships: Sequence[int],
    max_launches: Optional[int] = None,
    deterministic: bool = False,
    action_config: Optional[ActionSpaceConfig] = None,
):
    device = logits.device
    config = action_config or DEFAULT_CONFIG.action
    if max_launches is None:
        max_launches = config.max_launches_per_source
    action_indices = torch.zeros(
        (config.max_sources, max_launches), dtype=torch.long, device=device
    )
    logprob_sum = torch.zeros((), device=device)
    entropy_sum = torch.zeros((), device=device)

    for src_idx in range(min(config.max_sources, len(actions))):
        remaining = int(source_ships[src_idx]) if src_idx < len(source_ships) else 0
        if remaining <= 0:
            continue
        source_actions = actions[src_idx]
        fractions = [
            (a.fraction if a is not None else -1.0)
            for a in source_actions
        ]
        frac_tensor = torch.tensor(fractions, dtype=torch.float32, device=device)
        for step_idx in range(max_launches):
            mask = _mask_tensor(frac_tensor, remaining, device)
            masked_logits = logits[src_idx].masked_fill(
                mask == 0, _mask_fill_value(logits[src_idx].dtype)
            )
            dist = Categorical(logits=masked_logits)
            if deterministic:
                action = torch.argmax(masked_logits)
            else:
                action = dist.sample()
            action_indices[src_idx, step_idx] = action
            logprob_sum = logprob_sum + dist.log_prob(action)
            entropy_sum = entropy_sum + dist.entropy()

            action_idx = int(action.item())
            if action_idx <= 0:
                break
            template = source_actions[action_idx]
            if template is None:
                break
            ships = _ships_to_send(remaining, template.fraction)
            if ships <= 0:
                break
            remaining -= ships
            if remaining <= 0:
                break

    return action_indices, logprob_sum, entropy_sum


def logprob_for_action_sequence(
    logits: torch.Tensor,
    actions: Sequence[Sequence[Optional[ActionTemplate]]],
    source_ships: Sequence[int],
    action_indices: torch.Tensor,
    max_launches: Optional[int] = None,
    action_config: Optional[ActionSpaceConfig] = None,
):
    device = logits.device
    config = action_config or DEFAULT_CONFIG.action
    if max_launches is None:
        max_launches = config.max_launches_per_source
    logprob_sum = torch.zeros((), device=device)
    entropy_sum = torch.zeros((), device=device)

    if hasattr(action_indices, "to"):
        action_indices = action_indices.to(device)

    for src_idx in range(min(config.max_sources, len(actions))):
        remaining = int(source_ships[src_idx]) if src_idx < len(source_ships) else 0
        if remaining <= 0:
            continue
        source_actions = actions[src_idx]
        fractions = [
            (a.fraction if a is not None else -1.0)
            for a in source_actions
        ]
        frac_tensor = torch.tensor(fractions, dtype=torch.float32, device=device)
        for step_idx in range(max_launches):
            try:
                action_idx = int(action_indices[src_idx, step_idx].item())
            except (TypeError, ValueError, IndexError):
                break
            mask = _mask_tensor(frac_tensor, remaining, device)
            masked_logits = logits[src_idx].masked_fill(
                mask == 0, _mask_fill_value(logits[src_idx].dtype)
            )
            dist = Categorical(logits=masked_logits)
            action_tensor = torch.tensor(action_idx, device=device)
            logprob_sum = logprob_sum + dist.log_prob(action_tensor)
            entropy_sum = entropy_sum + dist.entropy()
            if action_idx <= 0:
                break
            template = source_actions[action_idx]
            if template is None:
                break
            ships = _ships_to_send(remaining, template.fraction)
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


def _mask_tensor(
    source_actions: torch.Tensor,
    remaining_ships: int,
    device: torch.device,
) -> torch.Tensor:
    if remaining_ships <= 0:
        mask = torch.zeros(len(source_actions), dtype=torch.float32, device=device)
        mask[0] = 1.0
        return mask

    mask = (remaining_ships * source_actions > 0.0).to(torch.float32)
    mask[0] = 1.0
    return mask


def _mask_fill_value(dtype: torch.dtype) -> float:
    if dtype.is_floating_point:
        return torch.finfo(dtype).min
    return -1e9
