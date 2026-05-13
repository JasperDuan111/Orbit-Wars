import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch.distributions import Categorical
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
from .config import (
    ACTIONS_PER_SOURCE,
    MAX_LAUNCHES_PER_SOURCE,
    MAX_SOURCES,
    MAX_TARGETS,
    SHIP_FRACTIONS,
)


def _get_field(obs, name, default=None):
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def _parse_planets(obs):
    raw_planets = _get_field(obs, "planets", [])
    return [Planet(*p) for p in raw_planets]


class ActionBuilder:
    def __init__(self):
        self.actions_per_source = ACTIONS_PER_SOURCE

    def build(self, obs):
        player_id = _get_field(obs, "player", 0)
        planets = _parse_planets(obs)
        my_planets = [p for p in planets if p.owner == player_id]
        my_planets.sort(key=lambda p: p.ships, reverse=True)

        actions: List[List[Optional[ActionTemplate]]] = []
        source_ships: List[int] = []

        for i in range(MAX_SOURCES):
            source_actions: List[Optional[ActionTemplate]] = [None] * self.actions_per_source
            source_actions[0] = None

            if i >= len(my_planets):
                actions.append(source_actions)
                source_ships.append(0)
                continue

            src = my_planets[i]
            source_ships.append(int(src.ships))
            def _target_priority(tgt):
                is_own = 1 if tgt.owner == player_id else 0
                dist = math.hypot(src.x - tgt.x, src.y - tgt.y)
                return (is_own, dist)

            ordered_targets = sorted(
                (p for p in planets if p.id != src.id),
                key=_target_priority,
            )
            if not ordered_targets:
                actions.append(source_actions)
                continue

            idx = 1
            for j in range(MAX_TARGETS):
                if j >= len(ordered_targets):
                    idx += len(SHIP_FRACTIONS)
                    continue
                tgt = ordered_targets[j]
                angle = math.atan2(tgt.y - src.y, tgt.x - src.x)
                for frac in SHIP_FRACTIONS:
                    source_actions[idx] = ActionTemplate(src.id, angle, float(frac))
                    idx += 1

            actions.append(source_actions)

        return actions, source_ships

    def decode(
        self,
        action_indices,
        actions: Sequence[Sequence[Optional["ActionTemplate"]]],
        source_ships: Sequence[int],
        max_launches: int = MAX_LAUNCHES_PER_SOURCE,
    ):
        if action_indices is None:
            return []
        if hasattr(action_indices, "tolist"):
            action_indices = action_indices.tolist()

        moves = []
        for src_idx in range(min(len(actions), MAX_SOURCES)):
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


def sample_action_sequence(
    logits: torch.Tensor,
    actions: Sequence[Sequence[Optional[ActionTemplate]]],
    source_ships: Sequence[int],
    max_launches: int = MAX_LAUNCHES_PER_SOURCE,
    deterministic: bool = False,
):
    device = logits.device
    action_indices = torch.zeros(
        (MAX_SOURCES, max_launches), dtype=torch.long, device=device
    )
    logprob_sum = torch.zeros((), device=device)
    entropy_sum = torch.zeros((), device=device)

    for src_idx in range(min(MAX_SOURCES, len(actions))):
        remaining = int(source_ships[src_idx]) if src_idx < len(source_ships) else 0
        if remaining <= 0:
            continue
        source_actions = actions[src_idx]
        for step_idx in range(max_launches):
            mask = _mask_tensor(source_actions, remaining, device)
            masked_logits = logits[src_idx].masked_fill(mask == 0, -1e9)
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
    max_launches: int = MAX_LAUNCHES_PER_SOURCE,
):
    device = logits.device
    logprob_sum = torch.zeros((), device=device)
    entropy_sum = torch.zeros((), device=device)

    if hasattr(action_indices, "to"):
        action_indices = action_indices.to(device)

    for src_idx in range(min(MAX_SOURCES, len(actions))):
        remaining = int(source_ships[src_idx]) if src_idx < len(source_ships) else 0
        if remaining <= 0:
            continue
        source_actions = actions[src_idx]
        for step_idx in range(max_launches):
            try:
                action_idx = int(action_indices[src_idx, step_idx].item())
            except (TypeError, ValueError, IndexError):
                break
            mask = _mask_tensor(source_actions, remaining, device)
            masked_logits = logits[src_idx].masked_fill(mask == 0, -1e9)
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
    ships = int(math.ceil(remaining_ships * float(fraction)))
    return max(0, min(ships, remaining_ships))


def _mask_tensor(
    source_actions: Sequence[Optional[ActionTemplate]],
    remaining_ships: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros((ACTIONS_PER_SOURCE,), dtype=torch.float32, device=device)
    mask[0] = 1.0
    if remaining_ships <= 0:
        return mask
    for idx in range(1, len(source_actions)):
        template = source_actions[idx]
        if template is None:
            continue
        ships = _ships_to_send(remaining_ships, template.fraction)
        if ships >= 1:
            mask[idx] = 1.0
    return mask
