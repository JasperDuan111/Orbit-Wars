"""
Orbit Wars — pit trained policies and rule-based agents against each other.

Usage:
  # Policy vs rule-based opponent
  python main.py --player0 checkpoints/ppo_orbit_wars_500.pt --player1 nearest

  # Two different policies fight each other (same config)
  python main.py --player0 checkpoints/model_v1.pt --player1 checkpoints/model_v2.pt

  # Two policies with DIFFERENT structures — each reads its own config
  python main.py --player0 checkpoints/mlp_model.pt --config0 configs/mlp_config.yaml \
                 --player1 checkpoints/gnn_model.pt --config1 configs/gnn_config.yaml

  # Rule-based vs rule-based (spectate)
  python main.py --player0 nearest --player1 starter --render

Player specs:
  nearest  — NearestPlanetOpponent (targets closest enemy planet)
  random   — RandomOpponent (sends random fleets)
  starter  — RuleBasedStarter (targets static planets first)
  nothing  — DoNothingOpponent (never moves)
  <path>   — path to a .pt checkpoint (loads trained PPO policy)
"""

import argparse
import os
import sys

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch

from rl.action import ActionBuilder, sample_action_discrete, build_orbit_lookup
from rl.config import OrbitWarsConfig
from rl.models import ActorCritic, ActorCriticGNN
from rl.obs import encode_observation

# ── Shared device ──────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════
# Per-player agent context (config + action builder + model)
# ══════════════════════════════════════════════════════════════════════════

class AgentContext:
    """Holds the config, action builder, and optional model for one player."""

    __slots__ = ("config", "action_builder", "model", "max_launches")

    def __init__(self, config: OrbitWarsConfig):
        self.config = config
        self.action_builder = ActionBuilder(config.action)
        self.max_launches = config.action.max_launches_per_source
        self.model = None  # populated by _load_model for policy agents


def _resolve_config(config_path, fallback_path: str) -> OrbitWarsConfig:
    """Load config from *config_path*, or fall back to *fallback_path*, or use defaults."""
    path = config_path or fallback_path
    if path and os.path.exists(path):
        return OrbitWarsConfig.from_yaml(path)
    return OrbitWarsConfig()


# ══════════════════════════════════════════════════════════════════════════
# Model factory (per-player)
# ══════════════════════════════════════════════════════════════════════════

def _build_model(config: OrbitWarsConfig):
    """Create a fresh model instance on the configured device."""
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


def _load_model(checkpoint_path: str, config: OrbitWarsConfig):
    """Load weights from *checkpoint_path*, return ``(model, update_number)``."""
    model = _build_model(config)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    state_dict = ckpt.get("policy_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    update = ckpt.get("update", "?") if isinstance(ckpt, dict) else "?"
    return model, update


# ══════════════════════════════════════════════════════════════════════════
# Agent functions (per-player context)
# ══════════════════════════════════════════════════════════════════════════

def _policy_forward(obs, ctx: AgentContext):
    """Core inference for one PPO policy → ``[[from_id, angle, ships], ...]``."""
    config = ctx.config

    obs_vec = encode_observation(
        obs, obs_config=config.obs, game_config=config.game,
        episode_steps=config.env.episode_steps,
    )

    obs_tensor = torch.from_numpy(obs_vec).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        sg, tgt, stop, offset, _, omask = ctx.model(obs_tensor)
    sg = sg.squeeze(0); tgt = tgt.squeeze(0); stop = stop.squeeze(0)
    offset = offset.squeeze(0); omask = omask.squeeze(0)

    planets, my_idx, non_my_idx, player_id = ctx.action_builder.get_planet_data(obs)
    planet_ships_dict = {
        idx: planets[idx].ships
        for idx in my_idx if planets[idx].owner == player_id
    }

    orbit_lookup = build_orbit_lookup(obs)
    angular_velocity = float(obs.get("angular_velocity", 0.0))

    action_indices, src_indices, offset_idx, _, _ = sample_action_discrete(
        source_logits=sg, ownership_mask=omask,
        target_scores=tgt, stop_logits=stop, offset_logits=offset,
        valid_targets=non_my_idx, planet_ships=planet_ships_dict,
        max_launches=ctx.max_launches, deterministic=True,
        offset_bins=config.action.offset_bins,
        planets=planets, orbit_lookup=orbit_lookup,
        angular_velocity=angular_velocity,
    )

    return ctx.action_builder.decode_all(
        planets, action_indices, src_indices, non_my_idx,
        offset_indices=offset_idx, offset_bins=config.action.offset_bins,
        orbit_lookup=orbit_lookup, angular_velocity=angular_velocity)


def _make_policy_agent(checkpoint_path: str, ctx: AgentContext):
    """Return ``(display_name, agent_fn)`` for a checkpoint-loaded policy."""
    model, update = _load_model(checkpoint_path, ctx.config)
    ctx.model = model
    ckpt_name = os.path.basename(checkpoint_path)
    name = f"Policy({ckpt_name}, update={update})"
    return name, lambda obs: _policy_forward(obs, ctx)


def _make_rule_agent(strategy: str):
    """Return ``(display_name, agent_fn)`` for a named rule-based strategy."""
    from rl.opponents import (
        NearestPlanetOpponent, RandomOpponent,
        RuleBasedStarter, DoNothingOpponent,
    )
    registry = {
        "nearest": (NearestPlanetOpponent, "NearestPlanet"),
        "random":  (RandomOpponent,        "Random"),
        "starter": (RuleBasedStarter,       "RuleBasedStarter"),
        "nothing": (DoNothingOpponent,      "DoNothing"),
    }
    if strategy not in registry:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from: {list(registry.keys())}, or a checkpoint path."
        )
    cls, label = registry[strategy]
    instance = cls()
    return label, lambda obs: instance.act(obs)


def _create_agent(spec: str, ctx = None):
    """Parse a player spec → ``(display_name, agent_fn)``.

    ``spec`` can be:
      - ``nearest``, ``random``, ``starter``, ``nothing``  → rule-based
      - a path to a ``.pt`` file                            → trained PPO policy

    For policy agents, *ctx* must be an ``AgentContext`` with the appropriate config.
    For rule-based agents, *ctx* is optional (ignored).
    """
    # Rule-based strategies by name
    if spec in ("nearest", "random", "starter", "nothing"):
        return _make_rule_agent(spec)

    # Otherwise treat as checkpoint path
    if not os.path.exists(spec):
        raise FileNotFoundError(
            f"Checkpoint not found: '{spec}'\n"
            f"Valid strategy names: nearest, random, starter, nothing"
        )
    if ctx is None:
        raise ValueError("AgentContext is required for policy agents (pass --config0 / --config1).")
    return _make_policy_agent(spec, ctx)


# ══════════════════════════════════════════════════════════════════════════
# Kaggle env compatibility patch
# ══════════════════════════════════════════════════════════════════════════

def _patch_orbit_wars_struct():
    """Monkey-patch ``Struct()`` so scalars pass through as-is.

    kaggle_environments' ``update_props`` calls ``Struct(scalar)`` for
    leaf fields like ``step``, ``player``, ``angular_velocity``, which
    crashes on the default ``Struct``.  This makes those pass through.
    """
    from kaggle_environments.utils import Struct

    _orig_new = Struct.__new__

    @staticmethod
    def _safe_new(cls, entries=None):
        if entries is None or isinstance(entries, dict):
            return _orig_new(cls)
        return entries

    Struct.__new__ = _safe_new


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Orbit Wars — pit policies and rule-based agents against each other.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --player0 checkpoints/model.pt --player1 nearest
  python main.py --player0 checkpoints/v1.pt --player1 checkpoints/v2.pt
  python main.py --player0 checkpoints/mlp.pt --config0 configs/mlp.yaml \\
                 --player1 checkpoints/gnn.pt --config1 configs/gnn.yaml
  python main.py --player0 nearest --player1 starter --render

Player specs:
  nearest  — NearestPlanetOpponent (targets closest enemy planet)
  random   — RandomOpponent (sends random fleets)
  starter  — RuleBasedStarter (targets static planets first)
  nothing  — DoNothingOpponent (never moves)
  <path>   — path to a .pt checkpoint
        """,
    )

    # ── Player specs ──
    parser.add_argument(
        "--player0", type=str, default="nearest",
        help="Agent spec for Player 0 (bottom-right).  Strategy name or checkpoint path.",
    )
    parser.add_argument(
        "--player1", type=str, default="random",
        help="Agent spec for Player 1 (top-left).  Strategy name or checkpoint path.",
    )

    # ── Config paths (per-player) ──
    parser.add_argument(
        "--config", type=str, default="configs/action_change.yaml",
        help="Shared config for both players (fallback when --config0/--config1 not set).",
    )
    parser.add_argument(
        "--config0", type=str, default=None,
        help="Config for Player 0 (overrides --config). Required when loading a .pt policy.",
    )
    parser.add_argument(
        "--config1", type=str, default=None,
        help="Config for Player 1 (overrides --config). Required when loading a .pt policy.",
    )

    # ── Game settings ──
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for board generation.",
    )
    parser.add_argument(
        "--episode-steps", type=int, default=500,
        help="Maximum steps per episode.",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Save replay HTML to replay.html.",
    )
    args = parser.parse_args()

    _patch_orbit_wars_struct()

    from kaggle_environments import make
    from kaggle_environments.utils import structify

    # ── Resolve config per player ──
    config0 = _resolve_config(args.config0, args.config)
    config1 = _resolve_config(args.config1, args.config)

    # ── Build both agents with their own context ──
    ctx0 = AgentContext(config0)
    ctx1 = AgentContext(config1)

    name0, agent0 = _create_agent(args.player0, ctx0)
    name1, agent1 = _create_agent(args.player1, ctx1)

    print(f"Player 0 (bottom-right): {name0}")
    if ctx0.model is not None:
        print(f"  config: model_type={config0.model.model_type}, "
              f"max_sources={config0.action.max_sources}, "
              f"max_targets={config0.action.max_targets}")
    print(f"Player 1 (top-left): {name1}")
    if ctx1.model is not None:
        print(f"  config: model_type={config1.model.model_type}, "
              f"max_sources={config1.action.max_sources}, "
              f"max_targets={config1.action.max_targets}")

    env = make(
        "orbit_wars",
        configuration=structify({
            "seed": args.seed,
            "episodeSteps": args.episode_steps,
        }),
        debug=True,
    )
    env.run([agent0, agent1])

    print()
    for i, state in enumerate(env.steps[-1]):
        label = name0 if i == 0 else name1
        print(f"Player {i} [{label}]: reward={state.reward}, status={state.status}")

    if args.render:
        path = os.path.join(_project_root, "replay.html")
        html = env.render(mode="html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\nReplay saved to: {path}")


if __name__ == "__main__":
    main()
