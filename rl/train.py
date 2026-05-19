import argparse
import os
import random
from dataclasses import replace
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .action import sample_action_sequence
from .config import OrbitWarsConfig
from .envs.orbit_wars_env import OrbitWarsSelfPlayEnv
from .models import ActorCritic
from .obs import encode_observation
from .opponents import OpponentPool, NearestPlanetOpponent, PolicyOpponent
from .ppo import RolloutBuffer, PPOTrainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    config = OrbitWarsConfig()
    parser = argparse.ArgumentParser(description="Orbit Wars PPO self-play training")
    parser.add_argument("--total-updates", type=int, default=config.train.total_updates)
    parser.add_argument("--rollout-steps", type=int, default=config.train.rollout_steps)
    parser.add_argument("--seed", type=int, default=config.train.seed)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=config.train.num_envs)
    parser.add_argument(
        "--max-launches-per-source",
        type=int,
        default=config.action.max_launches_per_source,
    )
    args = parser.parse_args()

    config.train.total_updates = args.total_updates
    config.train.rollout_steps = args.rollout_steps
    config.train.seed = args.seed
    config.train.num_envs = args.num_envs
    config.action = replace(
        config.action, max_launches_per_source=args.max_launches_per_source
    )

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    set_seed(config.train.seed)

    log_dir = args.log_dir or os.path.join(
        "runs", "OrbitWars", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    writer = SummaryWriter(log_dir=log_dir)

    envs = [
        OrbitWarsSelfPlayEnv(
            opponent=NearestPlanetOpponent(),
            env_config=replace(config.env, seed=config.train.seed + i),
            reward_config=config.reward,
            action_config=config.action,
        )
        for i in range(config.train.num_envs)
    ]

    policy = ActorCritic(
        config.obs.obs_dim,
        config.action.actions_per_source,
        config.action.max_sources,
        model_config=config.model,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.train.learning_rate)
    trainer = PPOTrainer(policy, optimizer, config.train, device,
                         action_config=config.action, use_amp=use_amp)
    buffer = RolloutBuffer(
        config.train.rollout_steps,
        config.train.num_envs,
        config.obs.obs_dim,
        config.action.max_sources,
        config.action.max_launches_per_source,
        device,
    )

    pool = OpponentPool(
        lambda: ActorCritic(
            config.obs.obs_dim,
            config.action.actions_per_source,
            config.action.max_sources,
            model_config=config.model,
        ),
        capacity=5,
        device=device,
        action_config=config.action,
        obs_config=config.obs,
        game_config=config.game,
    )

    start_update = 1
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "policy_state_dict" in ckpt:
            # New format: full checkpoint with metadata
            policy.load_state_dict(ckpt["policy_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_update = ckpt.get("update", 0) + 1
            if "pool_snapshots" in ckpt:
                pool.restore_snapshots(ckpt["pool_snapshots"])
            print(f"  Restored policy, optimizer, pool.  Starting at update {start_update}")
        else:
            # Old format: raw state_dict only (no optimizer / pool / update)
            policy.load_state_dict(ckpt)
            print("  Old-format checkpoint — only policy restored.  Starting at update 1")

    for env in envs:
        env.set_opponent(pool.sample())

    obs_list = [env.reset() for env in envs]

    try:
        for update in range(start_update, config.train.total_updates + 1):
            buffer.clear()
            for _ in range(config.train.rollout_steps):
                obs_vec_batch = np.stack(
                    [
                        encode_observation(
                            obs, obs_config=config.obs, game_config=config.game,
                            episode_steps=config.env.episode_steps,
                        )
                        for obs in obs_list
                    ]
                )
                obs_tensor = torch.from_numpy(obs_vec_batch).float().to(device)

                with torch.no_grad():
                    logits_batch, values_batch = policy(obs_tensor)
                values_batch = values_batch.squeeze(-1)

                actions_batch = torch.zeros(
                    (
                        config.train.num_envs,
                        config.action.max_sources,
                        config.action.max_launches_per_source,
                    ),
                    dtype=torch.long,
                    device=device,
                )
                logprobs_batch = torch.zeros((config.train.num_envs,), device=device)

                # --- Batch opponent inference ---
                opponent_actions_per_env = []
                policy_opponent_items = []

                for i, env in enumerate(envs):
                    env_actions = {}
                    for opp_idx_in_list, opponent, opp_obs in env.get_opponents_data():
                        if isinstance(opponent, PolicyOpponent):
                            policy_opponent_items.append(
                                (i, opp_idx_in_list, opponent, opp_obs)
                            )
                        else:
                            env_actions[opp_idx_in_list] = opponent.act(opp_obs)
                    opponent_actions_per_env.append(env_actions)

                if policy_opponent_items:
                    batch_results = PolicyOpponent.batch_act(
                        [(opp, obs) for _, _, opp, obs in policy_opponent_items],
                        device=device,
                        action_config=config.action,
                        obs_config=config.obs,
                        game_config=config.game,
                        episode_steps=config.env.episode_steps,
                    )
                    for (env_i, opp_idx, _, _), action in zip(
                        policy_opponent_items, batch_results
                    ):
                        opponent_actions_per_env[env_i][opp_idx] = action

                next_obs_list = []
                rewards = torch.zeros((config.train.num_envs,), device=device)
                dones = torch.zeros((config.train.num_envs,), device=device)
                obs_snapshots = []

                for i, env in enumerate(envs):
                    action_templates = env.last_actions
                    source_ships = env.last_source_ships
                    action_indices, logprob, _ = sample_action_sequence(
                        logits_batch[i],
                        action_templates,
                        source_ships,
                        max_launches=config.action.max_launches_per_source,
                        deterministic=False,
                        action_config=config.action,
                    )
                    actions_batch[i] = action_indices
                    logprobs_batch[i] = logprob

                    obs_snapshots.append(dict(obs_list[i]))
                    next_obs, reward, done, info = env.step(
                        action_indices,
                        opponent_actions=opponent_actions_per_env[i],
                    )
                    rewards[i] = reward
                    dones[i] = float(done)

                    if done:
                        env.set_opponent(pool.sample())
                        next_obs = env.reset()

                    next_obs_list.append(next_obs)

                buffer.add_batch(
                    obs_tensor,
                    obs_snapshots,
                    actions_batch,
                    logprobs_batch,
                    rewards,
                    dones,
                    values_batch,
                )

                obs_list = next_obs_list

            last_obs_vec = np.stack(
                [
                    encode_observation(
                        obs, obs_config=config.obs, game_config=config.game,
                        episode_steps=config.env.episode_steps,
                    )
                    for obs in obs_list
                ]
            )
            last_obs_tensor = torch.from_numpy(last_obs_vec).float().to(device)
            with torch.no_grad():
                _, last_values = policy(last_obs_tensor)
            buffer.compute_returns_and_advantages(
                last_values.squeeze(-1), config.train.gamma, config.train.gae_lambda
            )

            stats = trainer.update(buffer)

            mean_reward = float(buffer.rewards.mean().item())
            writer.add_scalar("train/reward_mean", mean_reward, update)
            writer.add_scalar("train/policy_loss", stats["policy_loss"], update)
            writer.add_scalar("train/value_loss", stats["value_loss"], update)
            writer.add_scalar("train/entropy", stats["entropy"], update)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], update)

            if update % config.train.save_every == 0:
                os.makedirs(args.save_dir, exist_ok=True)
                ckpt_path = os.path.join(args.save_dir, f"ppo_orbit_wars_{update}.pt")
                torch.save(
                    {
                        "policy_state_dict": policy.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "update": update,
                        "pool_snapshots": list(pool.snapshots),
                    },
                    ckpt_path,
                )
                pool.add(policy.state_dict())

            if update % config.train.opponent_refresh == 0:
                for env in envs:
                    env.set_opponent(pool.sample())

            if update % 10 == 0:
                print(
                    f"update={update} policy_loss={stats['policy_loss']:.4f} "
                    f"value_loss={stats['value_loss']:.4f} entropy={stats['entropy']:.4f}"
                )
    finally:
        writer.close()


if __name__ == "__main__":
    main()
