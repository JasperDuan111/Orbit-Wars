from typing import Optional

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .action import logprob_batched_combined
from .config import ActionSpaceConfig, DEFAULT_CONFIG


class RolloutBuffer:
    def __init__(self, rollout_steps, num_envs, obs_dim, max_sources, max_launches, device):
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.device = device
        self.obs = torch.zeros((rollout_steps, num_envs, obs_dim), device=device)
        self.actions = torch.zeros(
            (rollout_steps, num_envs, max_sources, max_launches, 2),
            dtype=torch.long, device=device,
        )
        self.logprobs = torch.zeros((rollout_steps, num_envs), device=device)
        self.rewards = torch.zeros((rollout_steps, num_envs), device=device)
        self.dones = torch.zeros((rollout_steps, num_envs), device=device)
        self.values = torch.zeros((rollout_steps, num_envs), device=device)
        self.advantages = torch.zeros((rollout_steps, num_envs), device=device)
        self.returns = torch.zeros((rollout_steps, num_envs), device=device)
        self.raw_obs = [[None for _ in range(num_envs)] for _ in range(rollout_steps)]
        self.valid_targets = [[None for _ in range(num_envs)] for _ in range(rollout_steps)]
        self.planet_ships = [[None for _ in range(num_envs)] for _ in range(rollout_steps)]
        self.source_ships = [[None for _ in range(num_envs)] for _ in range(rollout_steps)]
        self.source_indices = [[None for _ in range(num_envs)] for _ in range(rollout_steps)]
        self.planets = [[None for _ in range(num_envs)] for _ in range(rollout_steps)]
        self.orbit_lookups = [[None for _ in range(num_envs)] for _ in range(rollout_steps)]
        self.steps = [[0 for _ in range(num_envs)] for _ in range(rollout_steps)]
        self.ang_vels = [[0.0 for _ in range(num_envs)] for _ in range(rollout_steps)]
        self.pos = 0

    def add_batch(self, obs, raw_obs, actions, logprobs, rewards, dones, values,
                  valid_targets_list=None, source_ships_list=None, source_indices_list=None,
                  planet_ships_list=None, planets_list=None,
                  orbit_lookups_list=None, steps_list=None, ang_vels_list=None):
        self.obs[self.pos].copy_(obs)
        self.actions[self.pos].copy_(actions)
        self.logprobs[self.pos].copy_(logprobs)
        self.rewards[self.pos].copy_(rewards)
        self.dones[self.pos].copy_(dones)
        self.values[self.pos].copy_(values)
        for i in range(self.num_envs):
            self.raw_obs[self.pos][i] = raw_obs[i]
            if valid_targets_list is not None: self.valid_targets[self.pos][i] = valid_targets_list[i]
            if source_ships_list is not None: self.source_ships[self.pos][i] = source_ships_list[i]
            if source_indices_list is not None: self.source_indices[self.pos][i] = source_indices_list[i]
            if planet_ships_list is not None: self.planet_ships[self.pos][i] = planet_ships_list[i]
            if planets_list is not None: self.planets[self.pos][i] = planets_list[i]
            if orbit_lookups_list is not None: self.orbit_lookups[self.pos][i] = orbit_lookups_list[i]
            if steps_list is not None: self.steps[self.pos][i] = steps_list[i]
            if ang_vels_list is not None: self.ang_vels[self.pos][i] = ang_vels_list[i]
        self.pos += 1

    def compute_returns_and_advantages(self, last_values, gamma, gae_lambda):
        gae = torch.zeros((self.num_envs,), device=self.device)
        for step in reversed(range(self.rollout_steps)):
            nv = last_values if step == self.rollout_steps - 1 else self.values[step + 1]
            nnt = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * nv * nnt - self.values[step]
            gae = delta + gamma * gae_lambda * nnt * gae
            self.advantages[step] = gae
        self.returns = self.advantages + self.values
        # Normalize returns once per buffer so value head targets have ~O(1) scale
        r = self.returns.reshape(-1)
        self.ret_mean = r.mean(); self.ret_std = r.std() + 1e-8
        self.returns = (self.returns - self.ret_mean) / self.ret_std

    def get(self, batch_size):
        total = self.rollout_steps * self.num_envs
        indices = np.random.permutation(total)
        flat_obs = self.obs.reshape(total, -1)
        flat_act = self.actions.reshape(total, *self.actions.shape[2:])
        flat_lp = self.logprobs.reshape(total)
        flat_ret = self.returns.reshape(total)
        flat_adv = self.advantages.reshape(total)
        flat_raw = [self.raw_obs[t][e] for t in range(self.rollout_steps) for e in range(self.num_envs)]
        flat_vt = [self.valid_targets[t][e] for t in range(self.rollout_steps) for e in range(self.num_envs)]
        flat_sh = [self.source_ships[t][e] for t in range(self.rollout_steps) for e in range(self.num_envs)]
        flat_si = [self.source_indices[t][e] for t in range(self.rollout_steps) for e in range(self.num_envs)]
        flat_ps = [self.planet_ships[t][e] for t in range(self.rollout_steps) for e in range(self.num_envs)]
        flat_pl = [self.planets[t][e] for t in range(self.rollout_steps) for e in range(self.num_envs)]
        flat_ol = [self.orbit_lookups[t][e] for t in range(self.rollout_steps) for e in range(self.num_envs)]
        flat_st = [self.steps[t][e] for t in range(self.rollout_steps) for e in range(self.num_envs)]
        flat_av = [self.ang_vels[t][e] for t in range(self.rollout_steps) for e in range(self.num_envs)]
        for start in range(0, total, batch_size):
            bi = indices[start:start + batch_size]
            yield bi, flat_obs[bi], flat_act[bi], flat_lp[bi], flat_ret[bi], flat_adv[bi], \
                [flat_raw[i] for i in bi], [flat_vt[i] for i in bi], [flat_sh[i] for i in bi], \
                [flat_si[i] for i in bi], [flat_ps[i] for i in bi], [flat_pl[i] for i in bi], \
                [flat_ol[i] for i in bi], [flat_st[i] for i in bi], [flat_av[i] for i in bi]

    def clear(self): self.pos = 0


class PPOTrainer:
    def __init__(self, policy, optimizer, config, device,
                 action_config: Optional[ActionSpaceConfig] = None):
        self.policy = policy
        self.optimizer = optimizer
        self.clip_range = config.clip_range
        self.ent_coef = config.ent_coef
        self.vf_coef = config.vf_coef
        self.value_reg = getattr(config, 'value_reg', 0.0)
        self.max_grad_norm = config.max_grad_norm
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.device = device
        self.action_config = action_config or DEFAULT_CONFIG.action
        self.max_launches = self.action_config.max_launches_per_source
        self.ship_fractions = self.action_config.ship_fractions

    def update(self, buffer):
        adv_flat = buffer.advantages.reshape(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        old_values = buffer.values.reshape(-1)
        returns = buffer.returns.reshape(-1)  # already normalized at buffer level
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        updates = 0

        for _ in range(self.epochs):
            for bi, obs, acts, old_lps, _, _, _, vt_list, sh_list, si_list, ps_list, pl_list, ol_list, st_list, av_list in buffer.get(self.batch_size):
                src, tgt, stop, frac, val, om = self.policy(obs)
                val = val.squeeze(-1)

                new_lps, ents, scnt = logprob_batched_combined(
                    src_logits=src, tgt_scores=tgt, stop_logits=stop,
                    frac_logits_all=frac, ownership_mask=om,
                    action_indices=acts, source_indices=si_list,
                    all_valid_targets=vt_list, all_planet_ships=ps_list,
                    max_launches=self.max_launches, ship_fractions=self.ship_fractions,
                    all_planets=pl_list,
                    all_orbit_lookups=ol_list, all_steps=st_list,
                    all_ang_vels=av_list,
                )

                ratios = torch.exp(new_lps - old_lps)
                surr1 = ratios * adv_flat[bi]
                surr2 = torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * adv_flat[bi]
                policy_loss = -torch.min(surr1, surr2).mean()

                rets = returns[bi]; ov = old_values[bi]
                r_mean, r_std = buffer.ret_mean, buffer.ret_std
                ov = (ov - r_mean) / r_std; vn = (val - r_mean) / r_std
                vc = ov + torch.clamp(vn - ov, -self.clip_range, self.clip_range)
                value_loss = 0.5 * torch.max((rets - vn).pow(2), (rets - vc).pow(2)).mean()

                # L2 penalty on raw value output — prevents collapse to constant
                value_l2 = val.squeeze(-1).pow(2).mean()

                entropy = (ents / (scnt + 1)).mean()
                loss = (policy_loss + self.vf_coef * value_loss
                        + self.value_reg * value_l2 - self.ent_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                # ── NaN detection: catch exploding gradients ──
                grad_norms = []
                has_nan = False
                for name, p in self.policy.named_parameters():
                    if p.grad is not None:
                        gn = p.grad.norm().item()
                        grad_norms.append(gn)
                        if gn != gn:  # NaN check
                            has_nan = True
                            print(f"  [!] NaN grad in {name}")
                if has_nan:
                    max_gn = max(grad_norms) if grad_norms else 0
                    print(f"  [!] NaN DETECTED — skipping optimizer step. max_grad={max_gn:.1f}")
                    continue  # skip this batch
                clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                # Check for NaN in weights after clipping   
                for name, p in self.policy.named_parameters():
                    if p.data.ne(p.data).any():  # NaN check
                        print(f"  [!] NaN in weights: {name}")
                        has_nan = True
                        break
                if has_nan:
                    print(f"  [!] Skipping optimizer step due to NaN weights")
                    continue
                self.optimizer.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                updates += 1

        return {k: v / max(updates, 1) for k, v in stats.items()}
