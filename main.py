"""
Orbit Wars — pit trained policies and rule-based agents against each other.

Usage:
  # Policy vs rule-based opponent
  python main.py --player0 checkpoints/ppo_orbit_wars_500.pt --player1 nearest

  # Two different policies fight each other
  python main.py --player0 checkpoints/model_v1.pt --player1 checkpoints/model_v2.pt

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

# ── Shared config & device ───────────────────────────────────────────
CONFIG_PATH = os.path.join(_project_root, "configs", "larger_model.yaml")
CONFIG = OrbitWarsConfig.from_yaml(CONFIG_PATH) if os.path.exists(CONFIG_PATH) else OrbitWarsConfig()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_BUILDER = ActionBuilder(CONFIG.action)
MAX_LAUNCHES = CONFIG.action.max_launches_per_source


# ══════════════════════════════════════════════════════════════════════
# Model factory
# ══════════════════════════════════════════════════════════════════════

def _build_model():
    """Create a fresh model instance on the configured device."""
    if CONFIG.model.model_type == "gnn":
        return ActorCriticGNN(
            obs_config=CONFIG.obs,
            actions_per_source=CONFIG.action.actions_per_source,
            max_sources=CONFIG.action.max_sources,
            n_fractions=len(CONFIG.action.ship_fractions),
            model_config=CONFIG.model,
        ).to(DEVICE)
    else:
        return ActorCritic(
            CONFIG.obs.obs_dim,
            CONFIG.action.actions_per_source,
            CONFIG.action.max_sources,
            model_config=CONFIG.model,
        ).to(DEVICE)


def _load_model(checkpoint_path: str):
    """Load weights from *checkpoint_path*, return ``(model, update_number)``."""
    model = _build_model()
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    state_dict = ckpt.get("policy_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    update = ckpt.get("update", "?") if isinstance(ckpt, dict) else "?"
    return model, update


# ══════════════════════════════════════════════════════════════════════
# Agent functions
# ══════════════════════════════════════════════════════════════════════

def _policy_forward(obs, model):
    """Core inference for one PPO policy → ``[[from_id, angle, ships], ...]``."""
    obs_vec = encode_observation(
        obs, obs_config=CONFIG.obs, game_config=CONFIG.game,
        episode_steps=CONFIG.env.episode_steps,
    )

    obs_tensor = torch.from_numpy(obs_vec).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        sg, tgt, stop, frac, _, omask = model(obs_tensor)
    sg = sg.squeeze(0); tgt = tgt.squeeze(0); stop = stop.squeeze(0)
    frac = frac.squeeze(0); omask = omask.squeeze(0)

    planets, my_idx, non_my_idx, player_id = ACTION_BUILDER.get_planet_data(obs)
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
        max_launches=MAX_LAUNCHES, deterministic=True,
        ship_fractions=CONFIG.action.ship_fractions,
        planets=planets, orbit_lookup=orbit_lookup,
        angular_velocity=angular_velocity, step=step,
    )

    return ACTION_BUILDER.decode_all(
        planets, action_indices, src_indices, planet_ships_dict, non_my_idx,
        orbit_lookup=orbit_lookup, angular_velocity=angular_velocity, step=step)


def _make_policy_agent(checkpoint_path: str):
    """Return ``(display_name, agent_fn)`` for a checkpoint-loaded policy."""
    model, update = _load_model(checkpoint_path)
    ckpt_name = os.path.basename(checkpoint_path)
    name = f"Policy({ckpt_name}, update={update})"
    return name, lambda obs: _policy_forward(obs, model)


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


def _create_agent(spec: str):
    """Parse a player spec → ``(display_name, agent_fn)``.

    ``spec`` can be:
      - ``nearest``, ``random``, ``starter``, ``nothing``  → rule-based
      - a path to a ``.pt`` file                            → trained PPO policy
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
    return _make_policy_agent(spec)


# ══════════════════════════════════════════════════════════════════════
# Kaggle env compatibility patch
# ══════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Orbit Wars — pit policies and rule-based agents against each other.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --player0 checkpoints/model.pt --player1 nearest
  python main.py --player0 checkpoints/v1.pt --player1 checkpoints/v2.pt
  python main.py --player0 nearest --player1 starter --render

Player specs:
  nearest  — NearestPlanetOpponent (targets closest enemy planet)
  random   — RandomOpponent (sends random fleets)
  starter  — RuleBasedStarter (targets static planets first)
  nothing  — DoNothingOpponent (never moves)
  <path>   — path to a .pt checkpoint
        """,
    )
    parser.add_argument(
        "--player0", type=str, default="nearest",
        help="Agent spec for Player 0 (top-left).  Strategy name or checkpoint path.",
    )
    parser.add_argument(
        "--player1", type=str, default="random",
        help="Agent spec for Player 1 (bottom-right).  Strategy name or checkpoint path.",
    )
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

    # ── Build both agents ──
    name0, agent0 = _create_agent(args.player0)
    name1, agent1 = _create_agent(args.player1)

    print(f"Player 0 (bottom-right): {name0}")
    print(f"Player 1 (top-left): {name1}")

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
