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

from .action import (ActionBuilder, sample_action_discrete,
                        build_orbit_lookup, _get_field)
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
    parser.add_argument(
        "--load-opponents",
        type=str, nargs="*", default=None,
        help="Pre-load checkpoint .pt files into the opponent pool before training.",
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

    # Create envs — fixed initial seed from config, +1 per episode.
    # Reproducible: same config seed → same sequence of planet layouts.
    base_seed = config.train.seed
    envs = [
        OrbitWarsSelfPlayEnv(
            opponent=NearestPlanetOpponent(),
            env_config=replace(config.env, seed=base_seed + i),
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
            n_fractions=len(config.action.ship_fractions),
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
                         action_config=config.action)
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
                n_fractions=len(config.action.ship_fractions),
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
        capacity=config.train.opponent_pool_capacity,
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

    # Pre-load opponents from checkpoint files
    if args.load_opponents:
        for path in args.load_opponents:
            pool.load_checkpoint(path)
            print(f"  Loaded static opponent from {path}")
        print(f"  Static opponents: {len(pool._static_opponents)}")

    for env in envs:
        env.set_opponent(pool.sample(config.reward.only_economy, rule_prob=config.train.opponent_rule_prob))

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
                    out = policy(obs_tensor)
                    source_logits, target_scores, stop_logits, frac_logits_all, values_batch, ownership_masks = out
                values_batch = values_batch.squeeze(-1)

                actions_batch = torch.zeros(
                    (
                        config.train.num_envs,
                        config.action.max_sources,
                        config.action.max_launches_per_source,
                        2,
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
                valid_targets_snapshots = []
                ships_snapshots = []
                source_indices_snapshots = []
                planets_snapshots = []
                orbit_snapshots = []
                ang_vel_snapshots = []

                action_builder = ActionBuilder(config.action)

                for i, env in enumerate(envs):
                    # 1. Planet metadata — unified obs indexing
                    planets, my_idx, non_my_idx, player_id = \
                        action_builder.get_planet_data(obs_list[i])

                    # 1b. Orbit data for intercept-angle computation
                    orbit_lookup = build_orbit_lookup(obs_list[i])
                    angular_velocity = float(
                        _get_field(obs_list[i], "angular_velocity", 0.0))

                    # 2. Per-slot action sampling: source → target → fraction
                    #    Build {planet_idx: ships} for all owned planets
                    planet_ships_dict = {
                        idx: planets[idx].ships
                        for idx in my_idx if planets[idx].owner == player_id
                    }
                    action_indices, src_indices, lp_val, _ = sample_action_discrete(
                        source_logits=source_logits[i],
                        ownership_mask=ownership_masks[i],
                        target_scores=target_scores[i],
                        stop_logits=stop_logits[i],
                        frac_logits_all=frac_logits_all[i],
                        valid_targets=non_my_idx,
                        planet_ships=planet_ships_dict,
                        max_launches=config.action.max_launches_per_source,
                        deterministic=False,
                        ship_fractions=config.action.ship_fractions,
                        epsilon=config.action.epsilon,
                        planets=planets,
                        orbit_lookup=orbit_lookup,
                        angular_velocity=angular_velocity,
                    )
                    actions_batch[i] = action_indices
                    logprobs_batch[i] = lp_val

                    # 3. Decode to moves (intercept angle + sun collision)
                    my_moves = action_builder.decode_all(
                        planets, action_indices,
                        src_indices, planet_ships_dict, non_my_idx,
                        orbit_lookup=orbit_lookup,
                        angular_velocity=angular_velocity,
                    )

                    # ── diagnostic: first env, first step of each update ──
                    if i == 0 and roll == 0 and update % 1 == 0:
                        n_own = int(ownership_masks[0].sum().item())
                        stop_v = float(stop_logits[0, 0].item())
                        tgt_max = float(target_scores[0].max().item())
                        tgt_min = float(target_scores[0].min().item())
                        src_list = [int(src_indices[s].item()) for s in range(min(5, len(src_indices)))]
                        src_ships = {s: planet_ships_dict.get(s, 0) for s in src_list}
                        tgt00 = int(action_indices[0, 0, 0].item())
                        tgt01 = int(action_indices[0, 0, 1].item()) if action_indices.shape[2] > 1 else -1
                        obs_step = int(_get_field(obs_list[i], "step", 0))
                        print(f"[diag] upd={update} step={obs_step} pid={player_id} n_own={n_own} "
                              f"stop={stop_v:.2f} tgt_max={tgt_max:.2f} tgt_min={tgt_min:.2f} "
                              f"srcs={src_list} ships={src_ships} "
                              f"tgt_cat0={tgt00} frac0={tgt01} "
                              f"moves={len(my_moves)} lp={lp_val:.3f}")

                    orbit_snapshots.append(orbit_lookup)
                    ang_vel_snapshots.append(angular_velocity)

                    obs_snapshots.append(dict(obs_list[i]))
                    valid_targets_snapshots.append(non_my_idx)
                    ships_snapshots.append(planet_ships_dict)
                    source_indices_snapshots.append(src_indices)
                    planets_snapshots.append(planets)

                    # 6. Step env
                    next_obs, reward, done, _info = env.step(
                        config.reward.only_economy,
                        opponent_actions=opponent_actions_per_env[i],
                        my_action_override=my_moves,
                        update_count = update,
                    )
                    rewards[i] = reward

                    n_own_next = sum(
                        1 for p in next_obs.get("planets", [])
                        if p[1] == player_id)
                    if done or n_own_next == 0:
                        dones[i] = 1.0
                        env.set_opponent(pool.sample(config.reward.only_economy, rule_prob=config.train.opponent_rule_prob))
                        next_obs = env.reset()
                    else:
                        dones[i] = float(done)

                    next_obs_list.append(next_obs)

                buffer.add_batch(
                    obs_tensor,
                    obs_snapshots,
                    actions_batch,
                    logprobs_batch,
                    rewards,
                    dones,
                    values_batch,
                    valid_targets_list=valid_targets_snapshots,
                    source_ships_list=ships_snapshots,
                    source_indices_list=source_indices_snapshots,
                    planet_ships_list=ships_snapshots,
                    planets_list=planets_snapshots,
                    orbit_lookups_list=orbit_snapshots,
                    ang_vels_list=ang_vel_snapshots,
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
                _, _, _, _, last_values, _ = policy(last_obs_tensor)
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
            # if update % 10 == 0 or update == start_update:
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
                    env.set_opponent(pool.sample(config.reward.only_economy, rule_prob=config.train.opponent_rule_prob))

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
