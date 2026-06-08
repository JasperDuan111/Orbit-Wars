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

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch

from rl.action import ActionBuilder, sample_action_sequence, select_sources
from rl.config import OrbitWarsConfig
from rl.models import ActorCritic, ActorCriticGNN
from rl.obs import encode_observation

# ── Load config ──
CONFIG = OrbitWarsConfig()

# ── Device ──
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Build model ──
if CONFIG.model.model_type == "gnn":
    MODEL = ActorCriticGNN(
        obs_config=CONFIG.obs,
        actions_per_source=CONFIG.action.actions_per_source,
        max_sources=CONFIG.action.max_sources,
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
        source_logits, slot_logits, _, ownership_mask = MODEL(obs_tensor)

    source_logits = source_logits.squeeze(0)
    slot_logits = slot_logits.squeeze(0)
    ownership_mask = ownership_mask.squeeze(0)

    # 3. Select source planets
    src_indices = select_sources(
        source_logits, ownership_mask,
        CONFIG.action.max_sources, deterministic=True,
    )

    # 4. Build action templates
    templates, source_ships = ACTION_BUILDER.build(obs, source_planet_ids=src_indices)

    # 5. Sample discrete actions
    action_indices, _, _ = sample_action_sequence(
        slot_logits, templates, source_ships,
        max_launches=MAX_LAUNCHES, deterministic=True, action_config=CONFIG.action,
    )

    # 6. Decode to moves
    return ACTION_BUILDER.decode(action_indices, templates, source_ships, max_launches=MAX_LAUNCHES)


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description="Run the PPO agent locally.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--opponent", type=str, default="random", choices=["random", "nearest"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    load_checkpoint(args.checkpoint)

    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": args.seed, "episodeSteps": 500}, debug=True)
    opponent = "random" if args.opponent == "random" else __file__
    env.run([agent, opponent])

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
