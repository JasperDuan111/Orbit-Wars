import math
import random
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from kaggle_environments.envs.orbit_wars.orbit_wars import CENTER, Planet, ROTATION_RADIUS_LIMIT
from .action import (ActionBuilder, sample_action_discrete,
                        build_orbit_lookup)
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
        moves = []
        player = obs.get("player", 0)
        planets = [Planet(*p) for p in obs.get("planets", [])]
        for p in planets:
            if p.owner == player and p.ships > 0:
                angle = random.uniform(0, 2 * math.pi)
                ships = p.ships // 2
                if ships >= 20:
                    moves.append([p.id, angle, ships])
        return moves
    
class RuleBasedStarter:
    """Simple scripted opponent that targets static planets first."""
    def act(self, obs):
        from kaggle_environments.envs.orbit_wars.orbit_wars import ROTATION_RADIUS_LIMIT, CENTER
        moves = []
        player_id = _get_field(obs, "player", 0)
        planets = [Planet(*p) for p in _get_field(obs, "planets", [])]

        static_targets = [p for p in planets
                         if math.hypot(p.x - CENTER, p.y - CENTER) + p.radius >= ROTATION_RADIUS_LIMIT
                         and p.owner != player_id]
        kids = [p for p in planets if p.owner == player_id]
        for mp in kids:
            if mp.ships <= 10: continue
            closest = None; best_d = float('inf')
            for t in static_targets:
                d = math.hypot(mp.x - t.x, mp.y - t.y)
                if d < best_d: best_d = d; closest = t
            if closest:
                ships = min(mp.ships // 2, closest.ships + 30)
                if ships > 10:
                    angle = math.atan2(closest.y - mp.y, closest.x - mp.x)
                    moves.append([mp.id, angle, ships])
        return moves

class DoNothingOpponent:
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
            sg, tgt, stop, frac, _, omask = self.policy(obs_tensor)
            sg = sg.squeeze(0); tgt = tgt.squeeze(0); stop = stop.squeeze(0)
            frac = frac.squeeze(0); omask = omask.squeeze(0)

        planets, my_idx, non_my_idx, player_id = self._action_builder.get_planet_data(obs)
        planet_ships_dict = {
            idx: planets[idx].ships
            for idx in my_idx if planets[idx].owner == player_id
        }

        orbit_lookup = build_orbit_lookup(obs)
        angular_velocity = float(obs.get("angular_velocity", 0.0))
        step = int(obs.get("step", 0))

        action_indices, src_indices, _, _ = sample_action_discrete(
            source_logits=sg, ownership_mask=omask,
            target_scores=tgt, stop_logits=stop, frac_logits_all=frac,
            valid_targets=non_my_idx, planet_ships=planet_ships_dict,
            max_launches=self.action_config.max_launches_per_source,
            deterministic=True, ship_fractions=self.action_config.ship_fractions,
            planets=planets,
            orbit_lookup=orbit_lookup,
            angular_velocity=angular_velocity,
            step=step,
        )
        return self._action_builder.decode_all(
            planets, action_indices, src_indices, planet_ships_dict, non_my_idx,
            orbit_lookup=orbit_lookup,
            angular_velocity=angular_velocity,
            step=step)

    @staticmethod
    def batch_act(opponents_with_obs, device, action_config=None, obs_config=None,
                  game_config=None, episode_steps=500):
        """Batch inference for multiple PolicyOpponents on GPU."""
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

            obs_vectors, raw_obs_list = [], []
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
                sg_b, tgt_b, stop_b, frac_b, _, omask_b = policy(obs_tensor)

            for i, idx in enumerate(indices):
                _, obs = opponents_with_obs[idx]
                pbi, my_i, non_i, player_id = first._action_builder.get_planet_data(obs)
                planet_ships_dict = {
                    pi: pbi[pi].ships
                    for pi in my_i if pbi[pi].owner == player_id
                }

                ol = build_orbit_lookup(obs)
                av = float(obs.get("angular_velocity", 0.0))
                st = int(obs.get("step", 0))

                acts, src_i, _, _ = sample_action_discrete(
                    source_logits=sg_b[i], ownership_mask=omask_b[i],
                    target_scores=tgt_b[i], stop_logits=stop_b[i],
                    frac_logits_all=frac_b[i],
                    valid_targets=non_i, planet_ships=planet_ships_dict,
                    max_launches=max_launches,
                    deterministic=True, ship_fractions=action_config.ship_fractions,
                    planets=pbi,
                    orbit_lookup=ol, angular_velocity=av, step=st,
                )
                results[idx] = first._action_builder.decode_all(
                    pbi, acts, src_i, planet_ships_dict, non_i,
                    orbit_lookup=ol, angular_velocity=av, step=st)

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
        self._static_opponents = []   # loaded via load_checkpoint, always in the rule mix

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

    def load_checkpoint(self, path: str):
        """Load a .pt checkpoint and add it to the static opponent lineup.

        Static opponents are always mixed into the rule-based branch of
        ``sample()``, so they act like persistent scripted foes rather
        than transient self-play snapshots.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("policy_state_dict", ckpt)
        policy = self.policy_factory().to(self.device)
        policy.load_state_dict(state_dict)
        policy.eval()
        self._static_opponents.append(PolicyOpponent(
            policy,
            device=self.device,
            action_config=self.action_config,
            obs_config=self.obs_config,
            game_config=self.game_config,
        ))

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

    def __len__(self):
        return len(self._policy_instances)

    def sample(self, only_economy, rule_prob: float = 0.0):
        """Sample an opponent.

        ``rule_prob`` gives the chance of returning a non-policy opponent
        even when policy snapshots are available.  This prevents the
        self-play from becoming too homogeneous.

        Rule-based opponents: ``NearestPlanetOpponent`` (aggressive), a
        pure random sender, or the built-in ``starter`` agent.
        """
        if only_economy:
            return random.choice([
                DoNothingOpponent(),
                RandomOpponent(),
            ])

        rule_pool = [
            NearestPlanetOpponent(),
            DoNothingOpponent(),
            RandomOpponent(),
            RuleBasedStarter(),
        ] + self._static_opponents

        if not self._policy_instances or (rule_prob > 0 and random.random() < rule_prob):
            return random.choice(rule_pool)

        policy = random.choice(self._policy_instances)
        return PolicyOpponent(
            policy,
            device=self.device,
            action_config=self.action_config,
            obs_config=self.obs_config,
            game_config=self.game_config,
        )



