import math
import random
import torch
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
from .action import ActionBuilder, sample_action_sequence
from .config import MAX_LAUNCHES_PER_SOURCE
from .obs import encode_observation
from .models import ActorCritic


def _get_field(obs, name, default=None):
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def _parse_planets(obs):
    raw_planets = _get_field(obs, "planets", [])
    return [Planet(*p) for p in raw_planets]


class NearestPlanetOpponent:
    def act(self, obs):
        moves = []
        player_id = _get_field(obs, "player", 0)
        planets = _parse_planets(obs)
        my_planets = [p for p in planets if p.owner == player_id]
        targets = [p for p in planets if p.owner != player_id]
        if not targets:
            return moves
        for mine in my_planets:
            nearest = min(targets, key=lambda t: math.hypot(mine.x - t.x, mine.y - t.y))
            ships_needed = nearest.ships + 1
            if mine.ships >= ships_needed:
                angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
                moves.append([mine.id, angle, ships_needed])
        return moves


class RandomOpponent:
    def act(self, obs):
        return []


class PolicyOpponent:
    def __init__(self, policy, device="cpu"):
        self.policy = policy.to(device)
        self.policy.eval()
        self.device = device
        self._action_builder = ActionBuilder()

    def act(self, obs):
        actions, source_ships = self._action_builder.build(obs)
        obs_vector = encode_observation(obs)
        obs_tensor = torch.from_numpy(obs_vector).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.policy(obs_tensor)
            action_indices, _, _ = sample_action_sequence(
                logits.squeeze(0),
                actions,
                source_ships,
                max_launches=MAX_LAUNCHES_PER_SOURCE,
                deterministic=True,
            )
        return self._action_builder.decode(
            action_indices,
            actions,
            source_ships,
            max_launches=MAX_LAUNCHES_PER_SOURCE,
        )


class OpponentPool:
    def __init__(self, policy_factory, capacity=5, device="cpu"):
        self.policy_factory = policy_factory
        self.capacity = capacity
        self.device = device
        self.snapshots = []

    def add(self, state_dict):
        self.snapshots.append({key: value.cpu() for key, value in state_dict.items()})
        if len(self.snapshots) > self.capacity:
            self.snapshots.pop(0)

    def sample(self):
        if not self.snapshots:
            return NearestPlanetOpponent()
        state_dict = random.choice(self.snapshots)
        policy = self.policy_factory().to(self.device)
        policy.load_state_dict(state_dict)
        return PolicyOpponent(policy, device=self.device)
