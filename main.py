"""
Orbit Wars — PPO-trained model agent.

Pipeline:
  Observation → encode → model.forward → select_sources → build templates → sample actions → decode → moves

Usage:
  python -m rl.main --checkpoint checkpoints/ppo_orbit_wars_500.pt
"""

import argparse
import os
import sys

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch

from rl.action import ActionBuilder, sample_action_discrete, select_sources
from rl.config import OrbitWarsConfig
from rl.models import ActorCritic, ActorCriticGNN
from rl.obs import encode_observation

# ── Load config ──
CONFIG_PATH = os.path.join(_project_root, "configs", "default.yaml")
CONFIG = OrbitWarsConfig.from_yaml(CONFIG_PATH) if os.path.exists(CONFIG_PATH) else OrbitWarsConfig()

# ── Device ──
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Build model ──
if CONFIG.model.model_type == "gnn":
    MODEL = ActorCriticGNN(
        obs_config=CONFIG.obs,
        actions_per_source=CONFIG.action.actions_per_source,
        max_sources=CONFIG.action.max_sources,
        n_fractions=len(CONFIG.action.ship_fractions),
        model_config=CONFIG.model,
    ).to(DEVICE)
else:
    MODEL = ActorCritic(
        CONFIG.obs.obs_dim,
        CONFIG.action.actions_per_source,
        CONFIG.action.max_sources,
        model_config=CONFIG.model,
    ).to(DEVICE)

MODEL.eval()

# ── Action builder ──
ACTION_BUILDER = ActionBuilder(CONFIG.action)
MAX_LAUNCHES = CONFIG.action.max_launches_per_source


def load_checkpoint(path: str):
    """Load trained weights into MODEL. Call before using agent()."""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    state_dict = ckpt.get("policy_state_dict", ckpt)
    MODEL.load_state_dict(state_dict)


def agent(obs):
    """Kaggle-compatible entry point. Returns [[from_id, angle, ships], ...]."""

    # 1. Encode observation
    obs_vec = encode_observation(
        obs,
        obs_config=CONFIG.obs,
        game_config=CONFIG.game,
        episode_steps=CONFIG.env.episode_steps,
    )
    obs_tensor = torch.from_numpy(obs_vec).float().unsqueeze(0).to(DEVICE)

    # 2. Model forward
    with torch.no_grad():
        sg, tgt, stop, frac, _, omask = MODEL(obs_tensor)
    sg = sg.squeeze(0); tgt = tgt.squeeze(0); stop = stop.squeeze(0)
    frac = frac.squeeze(0); omask = omask.squeeze(0)

    # 3. Select source planets
    src_indices = select_sources(sg, omask, CONFIG.action.max_sources, deterministic=True)

    # 4. Planet metadata
    planets, my_idx, non_my_idx, _ = ACTION_BUILDER.get_planet_data(obs)
    if src_indices is not None:
        ordered = [int(idx.item()) for idx in src_indices]
    else:
        ordered = my_idx[:CONFIG.action.max_sources]
    ordered = ordered[:CONFIG.action.max_sources]
    ships = [planets[si].ships if si < len(planets) else 0 for si in ordered]

    # 5. Sample discrete actions (two-step: target → fraction)
    action_indices, _, _ = sample_action_discrete(
        target_scores=tgt, stop_logits=stop, frac_logits_all=frac,
        valid_targets=non_my_idx, source_ships=ships,
        max_launches=MAX_LAUNCHES, deterministic=True,
        ship_fractions=CONFIG.action.ship_fractions,
    )

    # 6. Decode to moves
    return ACTION_BUILDER.decode_all(planets, action_indices, ordered, ships, non_my_idx)


# ── CLI ──

def _patch_orbit_wars_struct():
    """Monkey-patch ``Struct()`` so it passes scalars through as-is.

    kaggle_environments' ``update_props`` does::

        state[k] = Struct(shared_state[k])

    for every shared field.  Leaf fields (step, player, angular_velocity…)
    are plain int/float, so ``Struct(42)`` crashes.  This patch makes
    ``Struct(42)`` return ``42`` directly.
    """
    from kaggle_environments.utils import Struct

    _orig_new = Struct.__new__

    @staticmethod
    def _safe_new(cls, entries=None):
        if entries is None or isinstance(entries, dict):
            return _orig_new(cls)
        # scalar or list — just return as-is so state[k] keeps the raw value
        return entries

    Struct.__new__ = _safe_new


def main():
    parser = argparse.ArgumentParser(description="Run the PPO agent locally.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--opponent", type=str, default="random", choices=["random", "nearest", "starter"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    load_checkpoint(args.checkpoint)
    _patch_orbit_wars_struct()

    from kaggle_environments import make
    from kaggle_environments.utils import structify

    env = make("orbit_wars", configuration=structify({"seed": args.seed, "episodeSteps": 500}), debug=True)

    if args.opponent == "random":
        from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent as opp_agent
    elif args.opponent == "nearest":
        from rl.opponents import NearestPlanetOpponent
        _near = NearestPlanetOpponent()
        opp_agent = lambda obs, _cfg=None: _near.act(obs)
    elif args.opponent == "starter":
        from rl.opponents import RuleBasedStarter
        _st = RuleBasedStarter()
        opp_agent = lambda obs, _cfg=None: _st.act(obs)
    else:
        opp_agent = agent

    env.run([agent, opp_agent])

    for i, state in enumerate(env.steps[-1]):
        print(f"Player {i}: reward={state.reward}, status={state.status}")

    if args.render:
        html = env.render(mode="html")
        path = os.path.join(_project_root, "replay.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Replay saved to: {path}")


if __name__ == "__main__":
    main()
