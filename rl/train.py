import argparse
import os
import random
import sys
import time
from dataclasses import replace
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from kaggle_environments.utils import Struct

from .action import sample_action_sequence
from .config import OrbitWarsConfig
from .envs.orbit_wars_env import OrbitWarsSelfPlayEnv
from .models import ActorCritic, ActorCriticGNN
from .obs import encode_observation
from .opponents import OpponentPool, NearestPlanetOpponent, PolicyOpponent
from .ppo import RolloutBuffer, PPOTrainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_time(seconds: float) -> str:
    """Format seconds as *h*min*s."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h}h{m:02d}min{s:.1f}s"
    elif m > 0:
        return f"{m}min{s:.1f}s"
    else:
        return f"{s:.1f}s"

def _format_result(r: float) -> str:
    return f"{r:.4f}"


class Logger:
    """Write to both console and log file."""
    def __init__(self, log_file: str):
        self.console = sys.stdout
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.file = open(log_file, "w", encoding="utf-8")

    def write(self, message):
        self.console.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def main():
    # Parse config
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None,
                                help="Path to YAML config file")
    config_args, _ = config_parser.parse_known_args()

    config = OrbitWarsConfig()
    if config_args.config:
        config = OrbitWarsConfig.from_yaml(config_args.config)

    parser = argparse.ArgumentParser(
        description="Orbit Wars PPO self-play training",
        parents=[config_parser],
    )
    parser.add_argument("--total-updates", type=int, default=config.train.total_updates)
    parser.add_argument("--rollout-steps", type=int, default=config.train.rollout_steps)
    parser.add_argument("--seed", type=int, default=config.train.seed)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=config.train.num_envs)
    parser.add_argument("--model-type", type=str, default=config.model.model_type,
                        choices=["mlp", "gnn"],
                        help="Model type: mlp (baseline) or gnn (GNN+Attention)")
    parser.add_argument("--epochs", type=int, default=config.train.epochs)
    parser.add_argument("--batch-size", type=int, default=config.train.batch_size)
    parser.add_argument("--save-every", type=int, default=config.train.save_every)
    parser.add_argument("--opponent-refresh", type=int, default=config.train.opponent_refresh)
    parser.add_argument("--learning-rate", type=float, default=config.train.learning_rate)
    parser.add_argument("--save-final-dir", type=str, default=None,
                        help="Save final model to this directory (auto-named as YYYYMMDD_{model_type}.pt)")
    parser.add_argument("--cleanup-checkpoints", action="store_true", default=False,
                        help="Delete checkpoint dir after saving final model")
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
    config.train.epochs = args.epochs
    config.train.batch_size = args.batch_size
    config.train.save_every = args.save_every
    config.train.opponent_refresh = args.opponent_refresh
    config.train.learning_rate = args.learning_rate
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

    # Setup logging
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, "log")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")
    logger = Logger(log_file)
    sys.stdout = logger

    print(f"Log file: {log_file}")
    print(f"Model type: {args.model_type}")
    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"Config: total_updates={config.train.total_updates}, "
          f"num_envs={config.train.num_envs}, "
          f"rollout_steps={config.train.rollout_steps}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # TensorBoard
    tb_log_dir = args.log_dir or os.path.join(
        "runs", "OrbitWars", timestamp
    )
    writer = SummaryWriter(log_dir=tb_log_dir)

    # Create envs
    envs = [
        OrbitWarsSelfPlayEnv(
            opponent=NearestPlanetOpponent(),
            env_config=replace(config.env, seed=config.train.seed + i),
            reward_config=config.reward,
            action_config=config.action,
        )
        for i in range(config.train.num_envs)
    ]

    # Create model
    if args.model_type == "gnn":
        policy = ActorCriticGNN(
            obs_config=config.obs,
            actions_per_source=config.action.actions_per_source,
            max_sources=config.action.max_sources,
            model_config=config.model,
        ).to(device)
    else:
        policy = ActorCritic(
            config.obs.obs_dim,
            config.action.actions_per_source,
            config.action.max_sources,
            model_config=config.model,
        ).to(device)

    # Parameter count
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print()

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

    # Opponent pool
    def _make_policy():
        if args.model_type == "gnn":
            return ActorCriticGNN(
                obs_config=config.obs,
                actions_per_source=config.action.actions_per_source,
                max_sources=config.action.max_sources,
                model_config=config.model,
            )
        else:
            return ActorCritic(
                config.obs.obs_dim,
                config.action.actions_per_source,
                config.action.max_sources,
                model_config=config.model,
            )

    pool = OpponentPool(
        _make_policy,
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
            policy.load_state_dict(ckpt["policy_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_update = ckpt.get("update", 0) + 1
            if "pool_snapshots" in ckpt:
                pool.restore_snapshots(ckpt["pool_snapshots"])
            print(f"  Restored policy, optimizer, pool.  Starting at update {start_update}")
        else:
            policy.load_state_dict(ckpt)
            print("  Old-format checkpoint — only policy restored.  Starting at update 1")

    for env in envs:
        env.set_opponent(pool.sample())

    obs_list = [env.reset() for env in envs]

    # Training loop
    total_start_time = time.time()

    try:
        for update in range(start_update, config.train.total_updates + 1):
            ct = datetime.now()
            update_start = rollout_start = time.time()

            buffer.clear()
            for roll in range(config.train.rollout_steps):
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

                # Batch opponent inference
                opponent_actions_per_env = []
                policy_opponent_items = []

                for i, env in enumerate(envs):
                    env_actions = Struct({})
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
                    # rendering_html = env._env.render(mode = "html")
                    # with open(f"./tmp/render_result/new-update-{update}-roll-{roll}-env-{i}.html", "w") as f:
                    #     f.write(rendering_html)
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

            rollout_time = time.time() - rollout_start

            # -- GAE computation --
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

            # -- Training --
            train_start = time.time()
            stats = trainer.update(buffer)
            train_time = time.time() - train_start

            update_time = time.time() - update_start

            # Logging & TensorBoard
            mean_reward = float(buffer.rewards.mean().item())
            writer.add_scalar("train/reward_mean", mean_reward, update)
            writer.add_scalar("train/policy_loss", stats["policy_loss"], update)
            writer.add_scalar("train/value_loss", stats["value_loss"], update)
            writer.add_scalar("train/entropy", stats["entropy"], update)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], update)

            train_ratio = f"{train_time/update_time*100:.2f}"
            # Log per-step timing
            log_msg = (
                f"==============================    Update {update}    ==================================\n"
                f"rollout{_format_time(rollout_time):>10s} | "
                f"train{_format_time(train_time):>10s} | "
                f"total{_format_time(update_time):>11s} | "
                f"train/total{train_ratio:>7s}% \n"
                f"reward{_format_result(mean_reward):>11s} | "
                f"p_loss{_format_result(stats['policy_loss']):>9s} | "
                f"v_loss{_format_result(stats['value_loss']):>10s} | "
                f"ent{_format_result(stats['entropy']):>16s} \n"
            )
            print(log_msg)

            # Save & opponent refresh
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

    finally:
        total_time = time.time() - total_start_time
        avg_time = total_time / max(update - start_update + 1, 1)

        print()
        print("-" * 60)
        print(f"Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total updates: {update}")
        print(f"Total time: {_format_time(total_time)}")
        print(f"Average per update: {_format_time(avg_time)}")
        print("-" * 60)

        writer.close()
        sys.stdout = logger.console
        logger.close()
        print(f"Log saved to: {log_file}")

        # Save final model and optionally clean up checkpoints
        if args.save_final_dir:
            os.makedirs(args.save_final_dir, exist_ok=True)
            date_str = datetime.now().strftime("%Y%m%d")
            final_name = f"{date_str}_{args.model_type}.pt"
            final_path = os.path.join(args.save_final_dir, final_name)
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "update": update,
                    "pool_snapshots": list(pool.snapshots),
                },
                final_path,
            )
            print(f"Final model saved to: {final_path}")

        if args.cleanup_checkpoints:
            save_dir = os.path.abspath(args.save_dir)
            if os.path.isdir(save_dir):
                import shutil
                shutil.rmtree(save_dir)
                print(f"Cleaned up checkpoints: {save_dir}")


if __name__ == "__main__":
    main()
