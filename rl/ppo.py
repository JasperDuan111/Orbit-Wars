from typing import Optional

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .action import ActionBuilder, logprob_for_action_sequence
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
        self.logprobs = torch.zeros((rollout_steps, num_envs), device=device)
        self.rewards = torch.zeros((rollout_steps, num_envs), device=device)
        self.dones = torch.zeros((rollout_steps, num_envs), device=device)
        self.values = torch.zeros((rollout_steps, num_envs), device=device)
        self.advantages = torch.zeros((rollout_steps, num_envs), device=device)
        self.returns = torch.zeros((rollout_steps, num_envs), device=device)
        self.raw_obs = [[None for _ in range(num_envs)] for _ in range(rollout_steps)]
        self.pos = 0

    def add_batch(self, obs, raw_obs, actions, logprobs, rewards, dones, values):
        if self.pos >= self.rollout_steps:
            raise RuntimeError("Rollout buffer is full.")
        self.obs[self.pos].copy_(obs)
        self.actions[self.pos].copy_(actions)
        self.logprobs[self.pos].copy_(logprobs)
        self.rewards[self.pos].copy_(rewards)
        self.dones[self.pos].copy_(dones)
        self.values[self.pos].copy_(values)
        for i in range(self.num_envs):
            self.raw_obs[self.pos][i] = raw_obs[i]
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
        flat_logprobs = self.logprobs.reshape(total)
        flat_returns = self.returns.reshape(total)
        flat_advantages = self.advantages.reshape(total)
        flat_raw_obs = [self.raw_obs[t][e] for t in range(self.rollout_steps) for e in range(self.num_envs)]

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
            )

    def clear(self):
        self.pos = 0


class PPOTrainer:
    def __init__(self, policy, optimizer, config, device, action_config: Optional[ActionSpaceConfig] = None):
        self.policy = policy
        self.optimizer = optimizer
        self.clip_range = config.clip_range
        self.ent_coef = config.ent_coef
        self.vf_coef = config.vf_coef
        self.max_grad_norm = config.max_grad_norm
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.device = device
        self.action_config = action_config or DEFAULT_CONFIG.action
        self.action_builder = ActionBuilder(self.action_config)
        self.max_launches = self.action_config.max_launches_per_source

    def update(self, buffer):
        advantages = buffer.advantages.reshape(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        old_values = buffer.values.reshape(-1)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        updates = 0

        for _ in range(self.epochs):
            for batch_idx, obs, actions, old_logprobs, returns, adv, raw_obs in buffer.get(
                self.batch_size
            ):
                logits, values = self.policy(obs)
                values = values.squeeze(-1)

                adv = advantages[batch_idx]

                new_logprobs = []
                entropies = []
                for i in range(len(raw_obs)):
                    action_templates, source_ships = self.action_builder.build(raw_obs[i])
                    logprob, entropy = logprob_for_action_sequence(
                        logits[i],
                        action_templates,
                        source_ships,
                        actions[i],
                        max_launches=self.max_launches,
                        action_config=self.action_config,
                    )
                    new_logprobs.append(logprob)
                    entropies.append(entropy)

                new_logprobs = torch.stack(new_logprobs)
                entropies = torch.stack(entropies)

                ratios = torch.exp(new_logprobs - old_logprobs)
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = old_values[batch_idx] + torch.clamp(
                    values - old_values[batch_idx], -self.clip_range, self.clip_range
                )
                value_loss_1 = (returns - values).pow(2)
                value_loss_2 = (returns - value_pred_clipped).pow(2)
                value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()

                entropy = entropies.mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
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
