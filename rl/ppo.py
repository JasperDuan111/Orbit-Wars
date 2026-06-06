from typing import Optional

import numpy as np
import torch
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

from .action import (
    _compute_max_valid_idx,
    _compute_step_valid,
    logprob_for_action_sequence_batched,
    source_selection_logprob,
)
from .config import ActionSpaceConfig, DEFAULT_CONFIG


class RolloutBuffer:
    def __init__(self, rollout_steps, num_envs, obs_dim, max_sources, max_launches, device):
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.device = device
        self.obs = torch.zeros((rollout_steps, num_envs, obs_dim), device=device)
        self.actions = torch.zeros(
            (rollout_steps, num_envs, max_sources, max_launches),
            dtype=torch.long,
            device=device,
        )
        self.fractions = torch.zeros(
            (rollout_steps, num_envs, max_sources, max_launches),
            device=device,
        )
        self.logprobs = torch.zeros((rollout_steps, num_envs), device=device)
        self.rewards = torch.zeros((rollout_steps, num_envs), device=device)
        self.dones = torch.zeros((rollout_steps, num_envs), device=device)
        self.values = torch.zeros((rollout_steps, num_envs), device=device)
        self.advantages = torch.zeros((rollout_steps, num_envs), device=device)
        self.returns = torch.zeros((rollout_steps, num_envs), device=device)
        self.raw_obs = [[None for _ in range(num_envs)] for _ in range(rollout_steps)]
        self.action_templates = [[None for _ in range(num_envs)] for _ in range(rollout_steps)]
        self.source_ships = [[None for _ in range(num_envs)] for _ in range(rollout_steps)]
        self.source_indices = [[None for _ in range(num_envs)] for _ in range(rollout_steps)]
        self.pos = 0

    def add_batch(self, obs, raw_obs, actions, logprobs, rewards, dones, values,
                  action_templates_list=None, source_ships_list=None,
                  source_indices_list=None, fractions=None):
        if self.pos >= self.rollout_steps:
            raise RuntimeError("Rollout buffer is full.")
        self.obs[self.pos].copy_(obs)
        self.actions[self.pos].copy_(actions)
        self.logprobs[self.pos].copy_(logprobs)
        self.rewards[self.pos].copy_(rewards)
        self.dones[self.pos].copy_(dones)
        self.values[self.pos].copy_(values)
        if fractions is not None:
            self.fractions[self.pos].copy_(fractions)
        for i in range(self.num_envs):
            self.raw_obs[self.pos][i] = raw_obs[i]
            if action_templates_list is not None:
                self.action_templates[self.pos][i] = action_templates_list[i]
            if source_ships_list is not None:
                self.source_ships[self.pos][i] = source_ships_list[i]
            if source_indices_list is not None:
                self.source_indices[self.pos][i] = source_indices_list[i]
        self.pos += 1

    def compute_returns_and_advantages(self, last_values, gamma, gae_lambda):
        gae = torch.zeros((self.num_envs,), device=self.device)
        for step in reversed(range(self.rollout_steps)):
            next_values = last_values if step == self.rollout_steps - 1 else self.values[step + 1]
            next_non_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[step] = gae
        self.returns = self.advantages + self.values

    def get(self, batch_size):
        total = self.rollout_steps * self.num_envs
        indices = np.random.permutation(total)
        flat_obs = self.obs.reshape(total, -1)
        flat_actions = self.actions.reshape(total, *self.actions.shape[2:])
        flat_fractions = self.fractions.reshape(total, *self.fractions.shape[2:])
        flat_logprobs = self.logprobs.reshape(total)
        flat_returns = self.returns.reshape(total)
        flat_advantages = self.advantages.reshape(total)
        flat_raw_obs = [self.raw_obs[t][e] for t in range(self.rollout_steps) for e in range(self.num_envs)]
        flat_templates = [self.action_templates[t][e] for t in range(self.rollout_steps) for e in range(self.num_envs)]
        flat_ships = [self.source_ships[t][e] for t in range(self.rollout_steps) for e in range(self.num_envs)]
        flat_source_indices = [self.source_indices[t][e] for t in range(self.rollout_steps) for e in range(self.num_envs)]

        for start in range(0, total, batch_size):
            batch_idx = indices[start : start + batch_size]
            yield (
                batch_idx,
                flat_obs[batch_idx],
                flat_actions[batch_idx],
                flat_logprobs[batch_idx],
                flat_returns[batch_idx],
                flat_advantages[batch_idx],
                [flat_raw_obs[i] for i in batch_idx],
                [flat_templates[i] for i in batch_idx],
                [flat_ships[i] for i in batch_idx],
                [flat_source_indices[i] for i in batch_idx],
                flat_fractions[batch_idx],
            )

    def clear(self):
        self.pos = 0


class PPOTrainer:
    def __init__(self, policy, optimizer, config, device,
                 action_config: Optional[ActionSpaceConfig] = None,
                 use_amp: bool = False):
        self.policy = policy
        self.optimizer = optimizer
        self.clip_range = config.clip_range
        self.ent_coef = config.ent_coef
        self.vf_coef = config.vf_coef
        self.max_grad_norm = config.max_grad_norm
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.device = device
        self.use_amp = use_amp
        self.scaler = GradScaler("cuda") if use_amp else None
        self.action_config = action_config or DEFAULT_CONFIG.action
        self.max_launches = self.action_config.max_launches_per_source

    def update(self, buffer):
        advantages = buffer.advantages.reshape(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        old_values = buffer.values.reshape(-1)
        # Use raw returns directly — standard PPO practice.
        # The value head needs to learn actual return magnitudes so that
        # GAE deltas accurately capture state-to-state variation.
        # Advantages get normalized; returns do not.
        flat_returns = buffer.returns.reshape(-1)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        updates = 0

        # Pre-compute static CPU structures for all (T*E) samples — invariant across epochs.
        total = buffer.rollout_steps * buffer.num_envs
        flat_all_actions = buffer.actions.reshape(total, *buffer.actions.shape[2:])  # (T*E, S, L)
        flat_all_fractions = buffer.fractions.reshape(total, *buffer.fractions.shape[2:])  # (T*E, S, L)
        flat_all_templates = [
            buffer.action_templates[t][e]
            for t in range(buffer.rollout_steps) for e in range(buffer.num_envs)
        ]
        flat_all_ships = [
            buffer.source_ships[t][e]
            for t in range(buffer.rollout_steps) for e in range(buffer.num_envs)
        ]
        max_valid_idx_all = _compute_max_valid_idx(
            flat_all_templates,
            self.action_config.max_sources,
            self.action_config.actions_per_source,
        )
        step_valid_all = _compute_step_valid(
            flat_all_actions, flat_all_ships, flat_all_fractions, self.max_launches,
        )

        for _ in range(self.epochs):
            for batch_idx, obs, actions, old_logprobs, returns, adv, raw_obs, \
                _templates_list, _ships_list, source_indices_list, _fractions_batch in buffer.get(
                self.batch_size
            ):
                if self.use_amp:
                    with autocast("cuda"):
                        source_logits, slot_logits, _fractions, values, ownership_mask = self.policy(obs)
                else:
                    source_logits, slot_logits, _fractions, values, ownership_mask = self.policy(obs)
                values = values.squeeze(-1)

                adv = advantages[batch_idx]

                # Batched slot logprob — single Categorical call for the whole batch.
                slot_lps, entropies_sum, src_counts = logprob_for_action_sequence_batched(
                    slot_logits,
                    max_valid_idx_all[batch_idx],
                    actions,
                    step_valid_all[batch_idx],
                )

                # Source-selection logprob (per-sample Plackett-Luce; k ≤ 20, lightweight).
                src_lps = []
                for i in range(len(raw_obs)):
                    src_lps.append(source_selection_logprob(
                        source_logits[i], ownership_mask[i], source_indices_list[i],
                    ))
                src_lps = torch.stack(src_lps)

                new_logprobs = src_lps + slot_lps

                ratios = torch.exp(new_logprobs - old_logprobs)
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = old_values[batch_idx] + torch.clamp(
                    values - old_values[batch_idx], -self.clip_range, self.clip_range
                )
                rets = flat_returns[batch_idx]
                value_loss_1 = (rets - values).pow(2)
                value_loss_2 = (rets - value_pred_clipped).pow(2)
                value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()

                # Per-source entropy (not summed across sources) so that the
                # entropy bonus is proportional to the average randomness per
                # decision, not the raw number of active sources.
                avg_entropy_per_source = entropies_sum / (src_counts + 1)  # (B,)
                entropy = avg_entropy_per_source.mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                updates += 1

        if updates > 0:
            stats = {key: value / updates for key, value in stats.items()}
        return stats
