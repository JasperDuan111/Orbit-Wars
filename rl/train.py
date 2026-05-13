import argparse
import os
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .action import sample_action_sequence
from .config import ACTIONS_PER_SOURCE, MAX_SOURCES, OBS_DIM, PPOConfig
from .envs.orbit_wars_env import OrbitWarsSelfPlayEnv
from .models import ActorCritic
from .obs import encode_observation
from .opponents import OpponentPool, NearestPlanetOpponent
from .ppo import RolloutBuffer, PPOTrainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    config = PPOConfig()
    parser = argparse.ArgumentParser(description="Orbit Wars PPO self-play training")
    parser.add_argument("--total-updates", type=int, default=config.total_updates)
    parser.add_argument("--rollout-steps", type=int, default=config.rollout_steps)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default="runs/orbit_wars")
    parser.add_argument("--num-envs", type=int, default=config.num_envs)
    parser.add_argument(
        "--max-launches-per-source",
        type=int,
        default=config.max_launches_per_source,
    )
    args = parser.parse_args()

    config.total_updates = args.total_updates
    config.rollout_steps = args.rollout_steps
    config.seed = args.seed
    config.num_envs = args.num_envs
    config.max_launches_per_source = args.max_launches_per_source

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(config.seed)

    writer = SummaryWriter(log_dir=args.log_dir)

    envs = [
        OrbitWarsSelfPlayEnv(
            opponent=NearestPlanetOpponent(),
            seed=config.seed + i,
            reward_scale=config.reward_scale,
            invalid_action_penalty=config.invalid_action_penalty,
            terminal_reward_scale=config.terminal_reward_scale,
            max_launches_per_source=config.max_launches_per_source,
        )
        for i in range(config.num_envs)
    ]

    policy = ActorCritic(OBS_DIM, ACTIONS_PER_SOURCE, MAX_SOURCES).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    trainer = PPOTrainer(policy, optimizer, config, device)
    buffer = RolloutBuffer(
        config.rollout_steps,
        config.num_envs,
        OBS_DIM,
        MAX_SOURCES,
        config.max_launches_per_source,
        device,
    )

    pool = OpponentPool(
        lambda: ActorCritic(OBS_DIM, ACTIONS_PER_SOURCE, MAX_SOURCES),
        capacity=5,
        device=device,
    )
    for env in envs:
        env.set_opponent(pool.sample())

    obs_list = [env.reset() for env in envs]

    try:
        for update in range(1, config.total_updates + 1):
            buffer.clear()
            for _ in range(config.rollout_steps):
                obs_vec_batch = np.stack([encode_observation(obs) for obs in obs_list])
                obs_tensor = torch.from_numpy(obs_vec_batch).float().to(device)

                with torch.no_grad():
                    logits_batch, values_batch = policy(obs_tensor)
                values_batch = values_batch.squeeze(-1)

                actions_batch = torch.zeros(
                    (config.num_envs, MAX_SOURCES, config.max_launches_per_source),
                    dtype=torch.long,
                    device=device,
                )
                logprobs_batch = torch.zeros((config.num_envs,), device=device)

                next_obs_list = []
                rewards = torch.zeros((config.num_envs,), device=device)
                dones = torch.zeros((config.num_envs,), device=device)
                obs_snapshots = []

                for i, env in enumerate(envs):
                    action_templates = env.last_actions
                    source_ships = env.last_source_ships
                    action_indices, logprob, _ = sample_action_sequence(
                        logits_batch[i],
                        action_templates,
                        source_ships,
                        max_launches=config.max_launches_per_source,
                        deterministic=False,
                    )
                    actions_batch[i] = action_indices
                    logprobs_batch[i] = logprob

                    obs_snapshots.append(dict(obs_list[i]))
                    next_obs, reward, done, info = env.step(action_indices)
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

            last_obs_vec = np.stack([encode_observation(obs) for obs in obs_list])
            last_obs_tensor = torch.from_numpy(last_obs_vec).float().to(device)
            with torch.no_grad():
                _, last_values = policy(last_obs_tensor)
            buffer.compute_returns_and_advantages(
                last_values.squeeze(-1), config.gamma, config.gae_lambda
            )

            stats = trainer.update(buffer)

            mean_reward = float(buffer.rewards.mean().item())
            writer.add_scalar("train/reward_mean", mean_reward, update)
            writer.add_scalar("train/policy_loss", stats["policy_loss"], update)
            writer.add_scalar("train/value_loss", stats["value_loss"], update)
            writer.add_scalar("train/entropy", stats["entropy"], update)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], update)

            if update % config.save_every == 0:
                os.makedirs(args.save_dir, exist_ok=True)
                ckpt_path = os.path.join(args.save_dir, f"ppo_orbit_wars_{update}.pt")
                torch.save(policy.state_dict(), ckpt_path)
                pool.add(policy.state_dict())

            if update % config.opponent_refresh == 0:
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
