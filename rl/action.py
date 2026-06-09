"""Action space for Orbit Wars — two-step discrete: target → fraction."""

import math
from typing import List, Optional, Sequence, Tuple

import torch
from torch.distributions import Categorical
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from .config import ActionSpaceConfig, DEFAULT_CONFIG


def _get_field(obs, name, default=None):
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def _parse_planets(obs):
    return [Planet(*p) for p in _get_field(obs, "planets", [])]


class ActionBuilder:
    def __init__(self, action_config: Optional[ActionSpaceConfig] = None):
        config = action_config or DEFAULT_CONFIG.action
        self.max_sources = config.max_sources
        self.max_launches = config.max_launches_per_source
        self.ship_fractions = config.ship_fractions

    def get_planet_data(self, obs):
        planets = _parse_planets(obs)
        player_id = _get_field(obs, "player", 0)
        my = [(i, p) for i, p in enumerate(planets) if p.owner == player_id]
        my.sort(key=lambda x: x[1].ships, reverse=True)
        non_my = [i for i, p in enumerate(planets) if p.owner != player_id]
        return planets, [i for i, _ in my], non_my, player_id

    def decode(self, planet_by_idx, src_obs_idx, tgt_obs_idx, frac_idx, remaining):
        src = planet_by_idx[src_obs_idx]; tgt = planet_by_idx[tgt_obs_idx]
        angle = math.atan2(tgt.y - src.y, tgt.x - src.x)
        frac = self.ship_fractions[frac_idx]
        ships = max(1, min(int(remaining * frac), remaining))
        return [src.id, angle, ships]

    def decode_all(self, planets, action_indices, source_obs_indices, source_ships, valid_targets):
        moves = []
        n_src = min(len(source_obs_indices), self.max_sources)
        for s in range(n_src):
            rem = source_ships[s]
            if rem <= 0: continue
            for l in range(self.max_launches):
                tgt_cat = int(action_indices[s, l, 0].item())
                if tgt_cat <= 0: break
                real_tgt = valid_targets[tgt_cat - 1]
                frac_idx = int(action_indices[s, l, 1].item())
                move = self.decode(planets, source_obs_indices[s], real_tgt, frac_idx, rem)
                moves.append(move)
                rem -= move[2]
                if rem <= 0: break
        return moves


# ── Source selection (unchanged) ───────────────────────────────────

def select_sources(source_logits, ownership_mask, k, deterministic=False):
    if source_logits.shape[-1] <= 1:
        return None
    device = source_logits.device
    N = source_logits.shape[-1]
    k = min(k, N)
    scores = source_logits.mean(dim=0)
    if deterministic:
        masked = scores.masked_fill(~ownership_mask, float('-inf'))
        _, indices = torch.topk(masked, k=k, dim=-1)
    else:
        gumbel = -torch.log(-torch.log(torch.rand(N, device=device) + 1e-10) + 1e-10)
        gumbel_scores = scores + gumbel
        gumbel_scores = gumbel_scores.masked_fill(~ownership_mask, float('-inf'))
        _, indices = torch.topk(gumbel_scores, k=k, dim=-1)
    return indices


def source_selection_logprob(source_logits, ownership_mask, source_indices):
    if source_indices is None or source_logits.shape[-1] <= 1:
        return torch.zeros((), device=source_logits.device)
    device = source_logits.device
    scores = source_logits.mean(dim=0)
    lp = torch.zeros((), device=device)
    mask = ownership_mask.clone()
    for i in range(len(source_indices)):
        idx = source_indices[i].item()
        if not mask[idx]: continue
        logp = scores.nan_to_num(0).masked_fill(~mask, float('-inf')).log_softmax(dim=-1)
        lp = lp + logp[idx]
        mask[idx] = False
    return lp


# ── Rollout sampling (no_grad, uses Categorical) ───────────────────

def sample_action_discrete(
    target_scores, stop_logits, frac_logits_all,
    valid_targets, source_ships, max_launches, deterministic, ship_fractions,
):
    S, N = target_scores.shape; L = max_launches
    device = target_scores.device
    action_indices = torch.zeros((S, L, 2), dtype=torch.long, device=device)
    lp = torch.zeros((), device=device); ent = torch.zeros((), device=device)
    for s in range(min(S, len(source_ships))):
        rem = source_ships[s]
        if rem <= 0: continue
        for l in range(L):
            vs = target_scores[s, valid_targets]
            combined = torch.cat([stop_logits[s], vs])
            dist = Categorical(logits=combined)
            act = torch.argmax(combined) if deterministic else dist.sample()
            lp += dist.log_prob(act); ent += dist.entropy().nan_to_num(0.0)
            tgt_cat = int(act.item()); action_indices[s, l, 0] = tgt_cat
            if tgt_cat == 0: break
            f_dist = Categorical(logits=frac_logits_all[s, valid_targets[tgt_cat - 1]])
            fa = torch.argmax(f_dist.logits) if deterministic else f_dist.sample()
            lp += f_dist.log_prob(fa); ent += f_dist.entropy().nan_to_num(0.0)
            action_indices[s, l, 1] = fa
            _f = ship_fractions[int(fa.item())]
            rem -= max(1, min(int(rem * _f), rem))
            if rem <= 0: break
    return action_indices, lp, ent


# ── PPO logprob (with gradients) ───────────────────────────────────

def logprob_batched_combined(
    src_logits, tgt_scores, stop_logits, frac_logits_all, ownership_mask,
    action_indices, source_indices, all_valid_targets, all_source_ships,
    max_launches, ship_fractions,
):
    B, S, N = tgt_scores.shape; L = max_launches
    device = tgt_scores.device

    # 1. Source logprob
    src_mean = src_logits.mean(dim=1).nan_to_num(0.0)
    fill_val = float('-inf') if src_mean.dtype == torch.float32 else -1e9
    src_mean = src_mean.masked_fill(~ownership_mask, fill_val)
    # Grace: if all masked (no owned planets), log_softmax produces NaN — clamp
    if not ownership_mask.any():
        return (torch.zeros(B, device=device),
                torch.zeros(B, device=device),
                torch.zeros(B, device=device))
    src_logp = src_mean.log_softmax(dim=-1).nan_to_num(0.0)
    src_lps = torch.zeros(B, device=device)
    for b in range(B):
        si = source_indices[b]
        if si is not None and len(si) > 0:
            src_lps[b] = src_logp[b, si.to(device)].sum()

    # 2. Target/STOP
    maxV = max((len(vt) for vt in all_valid_targets), default=0)
    PAD = max(1, 1 + maxV)  # at least 2 slots (STOP + nothing)
    combined = torch.full((B, S, PAD), fill_val, device=device)
    combined[..., 0] = stop_logits.squeeze(-1).nan_to_num(0.0)
    for b in range(B):
        vt = all_valid_targets[b]
        nv = len(vt)
        if nv:
            combined[b, :, 1:1 + nv] = tgt_scores[b, :, vt].nan_to_num(0.0)
    all_logp = combined.log_softmax(dim=-1).nan_to_num(0.0)
    safe_lp = all_logp.masked_fill(all_logp == float('-inf'), 0.0)
    all_ent = -(safe_lp.exp() * safe_lp).sum(dim=-1).nan_to_num(0.0)

    # 3. Target + fraction logprobs
    source_count = torch.zeros(B, device=device)
    tgt_lps = torch.zeros(B, device=device)
    tgt_ents = torch.zeros(B, device=device)
    frac_lps = torch.zeros(B, device=device)
    frac_ents = torch.zeros(B, device=device)

    for b in range(B):
        ships = list(all_source_ships[b])
        valid = all_valid_targets[b]
        for s in range(S):
            rem = ships[s] if s < len(ships) else 0
            if rem <= 0: continue
            source_count[b] += 1
            for l in range(L):
                tc = int(action_indices[b, s, l, 0].item())
                tgt_lps[b] += all_logp[b, s, tc]
                tgt_ents[b] += all_ent[b, s]
                if tc == 0: break
                ti = valid[tc - 1]
                fl = frac_logits_all[b, s, ti].nan_to_num(0.0)          # (F,)
                flp = fl.log_softmax(dim=-1).nan_to_num(0.0)
                fi = int(action_indices[b, s, l, 1].item())
                frac_lps[b] += flp[fi]
                safe_flp = flp.masked_fill(flp == float('-inf'), 0.0)
                frac_ents[b] -= (safe_flp.exp() * safe_flp).sum()
                _f = ship_fractions[fi]
                rem -= max(1, min(int(rem * _f), rem))
                if rem <= 0: break

    return src_lps + tgt_lps + frac_lps, tgt_ents + frac_ents, source_count
