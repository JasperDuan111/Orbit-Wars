"""
Orbit Wars — PPO agent submission.

Kaggle calls ``agent(obs)`` each step and expects a list of moves
``[[from_planet_id, angle_in_radians, num_ships], ...]``.

Setup before submitting:
    1. Copy your trained checkpoint to this folder as ``model.pt``
    2. Make sure ``config.yaml`` matches the checkpoint (model_type, action space, etc.)
    3. Tar the folder:  tar -czf submission.tar.gz -C submission .
    4. Submit:  kaggle competitions submit orbit-wars -f submission.tar.gz -m "PPO agent"
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch

from rl.config import OrbitWarsConfig
from rl.models import ActorCritic, ActorCriticGNN
from rl.obs import encode_observation
from rl.action import ActionBuilder, sample_action_discrete, build_orbit_lookup

# ── Module-level init (runs once at import time) ────────────────────────

DEVICE = torch.device("cpu")  # Kaggle evaluation uses CPU

_CONFIG_PATH = os.path.join(_HERE, "config.yaml")
if os.path.exists(_CONFIG_PATH):
    _CONFIG = OrbitWarsConfig.from_yaml(_CONFIG_PATH)
else:
    _CONFIG = OrbitWarsConfig()

_MODEL_PATH = os.path.join(_HERE, "model.pt")

_ACTION_BUILDER = ActionBuilder(_CONFIG.action)
_MAX_LAUNCHES = _CONFIG.action.max_launches_per_source


def _build_model(config: OrbitWarsConfig):
    """Create a model instance matching *config*."""
    if config.model.model_type == "gnn":
        return ActorCriticGNN(
            obs_config=config.obs,
            actions_per_source=config.action.actions_per_source,
            max_sources=config.action.max_sources,
            n_offsets=config.action.n_offsets,
            model_config=config.model,
        ).to(DEVICE)
    else:
        return ActorCritic(
            config.obs.obs_dim,
            config.action.actions_per_source,
            config.action.max_sources,
            model_config=config.model,
        ).to(DEVICE)


# Load model weights once
_MODEL = None
_MODEL_LOAD_ERROR = None

if os.path.exists(_MODEL_PATH):
    try:
        _MODEL = _build_model(_CONFIG)
        ckpt = torch.load(_MODEL_PATH, map_location=DEVICE, weights_only=True)
        state_dict = ckpt.get("policy_state_dict", ckpt)
        _MODEL.load_state_dict(state_dict)
        _MODEL.eval()
        _update = ckpt.get("update", "?") if isinstance(ckpt, dict) else "?"
        print(f"[orbit-wars-agent] Loaded model (update={_update}, "
              f"type={_CONFIG.model.model_type}, "
              f"params={sum(p.numel() for p in _MODEL.parameters()):,})",
              file=sys.stderr)
    except Exception as exc:
        _MODEL = None
        _MODEL_LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        print(f"[orbit-wars-agent] Model load FAILED: {_MODEL_LOAD_ERROR}", file=sys.stderr)
else:
    _MODEL_LOAD_ERROR = f"model.pt not found at {_MODEL_PATH}"
    print(f"[orbit-wars-agent] WARNING: {_MODEL_LOAD_ERROR}", file=sys.stderr)


# ── Helpers ─────────────────────────────────────────────────────────

def _get(obs, key, default=None):
    """Read *key* from a dict or Struct, returning *default* if absent."""
    if isinstance(obs, dict):
        return obs.get(key, default)
    try:
        return getattr(obs, key)
    except AttributeError:
        return default


# ── Agent function ──────────────────────────────────────────────────

def agent(obs):
    """Called by Kaggle each step. Returns a list of moves.

    Each move: ``[from_planet_id, angle_in_radians, num_ships]``
    """

    # ── Fallback if model didn't load ──
    if _MODEL is None:
        import math
        from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

        moves = []
        player = _get(obs, "player", 0)
        planets = [Planet(*p) for p in _get(obs, "planets", [])]
        my_planets = [p for p in planets if p.owner == player]
        targets = [p for p in planets if p.owner != player]
        if not targets:
            return moves
        for mine in my_planets:
            nearest = min(targets, key=lambda t: math.hypot(mine.x - t.x, mine.y - t.y))
            ships_needed = nearest.ships + 1
            if mine.ships >= ships_needed:
                angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
                moves.append([mine.id, angle, ships_needed])
        return moves

    # ── Main inference path ──
    obs_vec = encode_observation(
        obs, obs_config=_CONFIG.obs, game_config=_CONFIG.game,
        episode_steps=_CONFIG.env.episode_steps,
    )

    obs_tensor = torch.from_numpy(obs_vec).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        sg, tgt, stop, offset, _, omask = _MODEL(obs_tensor)
    sg = sg.squeeze(0); tgt = tgt.squeeze(0); stop = stop.squeeze(0)
    offset = offset.squeeze(0); omask = omask.squeeze(0)

    planets, my_idx, non_my_idx, player_id = _ACTION_BUILDER.get_planet_data(obs)
    planet_ships_dict = {
        idx: planets[idx].ships
        for idx in my_idx if planets[idx].owner == player_id
    }

    orbit_lookup = build_orbit_lookup(obs)
    angular_velocity = float(_get(obs, "angular_velocity", 0.0))

    action_indices, src_indices, offset_idx, _, _ = sample_action_discrete(
        source_logits=sg, ownership_mask=omask,
        target_scores=tgt, stop_logits=stop, offset_logits=offset,
        valid_targets=non_my_idx, planet_ships=planet_ships_dict,
        max_launches=_MAX_LAUNCHES, deterministic=True,
        offset_bins=_CONFIG.action.offset_bins,
        planets=planets, orbit_lookup=orbit_lookup,
        angular_velocity=angular_velocity,
    )

    return _ACTION_BUILDER.decode_all(
        planets, action_indices, src_indices, non_my_idx,
        offset_indices=offset_idx, offset_bins=_CONFIG.action.offset_bins,
        orbit_lookup=orbit_lookup, angular_velocity=angular_velocity,
    )
