import math
import random
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
from .action import (ActionBuilder, sample_action_sequence,
                        select_sources, source_selection_logprob)
from .config import ActionSpaceConfig, DEFAULT_CONFIG, GameConfig, ObsConfig
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
    def __init__(
        self,
        policy,
        device="cpu",
        action_config: Optional[ActionSpaceConfig] = None,
        obs_config: Optional[ObsConfig] = None,
        game_config: Optional[GameConfig] = None,
    ):
        self.policy = policy.to(device)
        self.policy.eval()
        self.device = device
        self.action_config = action_config or DEFAULT_CONFIG.action
        self.obs_config = obs_config or DEFAULT_CONFIG.obs
        self.game_config = game_config or DEFAULT_CONFIG.game
        self._action_builder = ActionBuilder(self.action_config)

    def act(self, obs):
        obs_vector = encode_observation(
            obs, obs_config=self.obs_config, game_config=self.game_config
        )
        obs_tensor = torch.from_numpy(obs_vector).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            source_logits, slot_logits, _, ownership_mask = self.policy(obs_tensor)
            source_logits = source_logits.squeeze(0)
            slot_logits = slot_logits.squeeze(0)
            ownership_mask = ownership_mask.squeeze(0)
        # Select sources deterministically
        src_indices = select_sources(
            source_logits, ownership_mask,
            self.action_config.max_sources, deterministic=True,
        )
        # Build templates with selected sources
        actions, source_ships = self._action_builder.build(
            obs, source_planet_ids=src_indices,
        )
        with torch.no_grad():
            action_indices, _, _ = sample_action_sequence(
                slot_logits,
                actions,
                source_ships,
                max_launches=self.action_config.max_launches_per_source,
                deterministic=True,
                action_config=self.action_config,
            )
        return self._action_builder.decode(
            action_indices,
            actions,
            source_ships,
            max_launches=self.action_config.max_launches_per_source,
        )

    @staticmethod
    def batch_act(opponents_with_obs, device, action_config=None, obs_config=None,
                  game_config=None, episode_steps=500):
        """Batch inference for multiple PolicyOpponents on GPU.

        Groups opponents by policy identity so that opponents sharing the
        same model checkpoint run in a single batched forward pass.
        """
        action_config = action_config or DEFAULT_CONFIG.action
        obs_config = obs_config or DEFAULT_CONFIG.obs
        game_config = game_config or DEFAULT_CONFIG.game
        max_launches = action_config.max_launches_per_source

        groups = defaultdict(list)
        for idx, (opponent, _obs) in enumerate(opponents_with_obs):
            groups[id(opponent.policy)].append(idx)

        results = [None] * len(opponents_with_obs)

        for _policy_id, indices in groups.items():
            first = opponents_with_obs[indices[0]][0]
            policy = first.policy

            obs_vectors = []
            raw_obs_list = []
            for idx in indices:
                _, obs = opponents_with_obs[idx]
                raw_obs_list.append(obs)
                obs_vec = encode_observation(
                    obs, obs_config=obs_config, game_config=game_config,
                    episode_steps=episode_steps,
                )
                obs_vectors.append(obs_vec)

            obs_tensor = torch.from_numpy(np.stack(obs_vectors)).float().to(device)
            with torch.no_grad():
                source_logits_batch, slot_logits_batch, _, ownership_masks = policy(obs_tensor)

            for i, idx in enumerate(indices):
                _, obs = opponents_with_obs[idx]
                # Select sources deterministically
                src_indices = select_sources(
                    source_logits_batch[i], ownership_masks[i],
                    action_config.max_sources, deterministic=True,
                )
                # Build templates with selected sources
                templates, ships = first._action_builder.build(
                    obs, source_planet_ids=src_indices,
                )
                # Sample slot actions
                action_indices, _, _ = sample_action_sequence(
                    slot_logits_batch[i],
                    templates,
                    ships,
                    max_launches=max_launches,
                    deterministic=True,
                    action_config=action_config,
                )
                results[idx] = first._action_builder.decode(
                    action_indices,
                    templates,
                    ships,
                    max_launches=max_launches,
                )

        return results


class OpponentPool:
    def __init__(
        self,
        policy_factory,
        capacity=5,
        device="cpu",
        action_config: Optional[ActionSpaceConfig] = None,
        obs_config: Optional[ObsConfig] = None,
        game_config: Optional[GameConfig] = None,
    ):
        self.policy_factory = policy_factory
        self.capacity = capacity
        self.device = device
        self.action_config = action_config or DEFAULT_CONFIG.action
        self.obs_config = obs_config or DEFAULT_CONFIG.obs
        self.game_config = game_config or DEFAULT_CONFIG.game
        self._snapshots = []
        self._policy_instances = []

    def add(self, state_dict):
        cpu_dict = {key: value.cpu() for key, value in state_dict.items()}
        self._snapshots.append(cpu_dict)
        policy = self.policy_factory().to(self.device)
        policy.load_state_dict(cpu_dict)
        policy.eval()
        self._policy_instances.append(policy)
        if len(self._snapshots) > self.capacity:
            self._snapshots.pop(0)
            old = self._policy_instances.pop(0)
            del old

    def restore_snapshots(self, snapshots):
        """Restore pool from a list of CPU state dicts (e.g. from checkpoint)."""
        self._snapshots = []
        self._policy_instances = []
        for cpu_dict in snapshots:
            self._snapshots.append(cpu_dict)
            policy = self.policy_factory().to(self.device)
            policy.load_state_dict(cpu_dict)
            policy.eval()
            self._policy_instances.append(policy)

    @property
    def snapshots(self):
        return self._snapshots

    def sample(self):
        if not self._policy_instances:
            return random.choice([NearestPlanetOpponent(), RandomOpponent()])
        policy = random.choice(self._policy_instances)
        return PolicyOpponent(
            policy,
            device=self.device,
            action_config=self.action_config,
            obs_config=self.obs_config,
            game_config=self.game_config,
        )
