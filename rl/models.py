import torch
import torch.nn as nn
from .config import DEFAULT_CONFIG, ModelConfig, GNNConfig, ObsConfig


def _build_mlp(input_dim, hidden_sizes, output_dim, dropout):
    layers = []
    last_dim = input_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(last_dim, size))
        layers.append(nn.LayerNorm(size))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        last_dim = size
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


# MLP baseline model (preserved for comparison)
class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim,
        actions_per_source,
        max_sources,
        hidden_sizes=None,
        dropout=None,
        model_config: ModelConfig = None,
    ):
        super().__init__()
        config = model_config or DEFAULT_CONFIG.model
        if hidden_sizes is None:
            hidden_sizes = config.hidden_sizes
        if dropout is None:
            dropout = config.dropout
        self.actions_per_source = actions_per_source
        self.max_sources = max_sources
        self.body = _build_mlp(obs_dim, hidden_sizes, hidden_sizes[-1], dropout)
        self.slot_policy_head = nn.Linear(hidden_sizes[-1], max_sources * actions_per_source)
        self.value_head = nn.Linear(hidden_sizes[-1], 1)
        # Bias STOP action (index 0) negative so the model defaults to
        # launching rather than idling — critical for breaking the initial
        # deadlock where random attacks never succeed and STOP looks optimal.
        with torch.no_grad():
            for s in range(max_sources):
                self.slot_policy_head.bias[s * actions_per_source] = -2.0
        # Fallback: learnable, non-observation-dependent source logits (equivalent to fixed ordering)
        self.fallback_source_logits = nn.Parameter(
            torch.zeros(max_sources, 1)
        )

    def forward(self, obs):
        B = obs.shape[0]
        features = self.body(obs)
        slot_logits = self.slot_policy_head(features)
        slot_logits = slot_logits.view(B, self.max_sources, self.actions_per_source)
        value = self.value_head(features)
        # MLP lacks per-planet structure: source_logits are non-spatial (zeros);
        # ActionBuilder falls back to ship-count ordering when no planet-dim source logits.
        source_logits = self.fallback_source_logits.expand(B, self.max_sources, 1)
        # ownership_mask: dummy True for all "planets" (MLP model doesn't use real planets)
        ownership_mask = torch.ones(B, 1, dtype=torch.bool, device=obs.device)
        # NaN guard: corrupted weights can produce NaN in any head output.
        # Intercept here before NaN leaks into logprobs, value loss, and GAE.
        source_logits = source_logits.nan_to_num(0.0)
        slot_logits = slot_logits.nan_to_num(0.0)
        value = value.nan_to_num(0.0)
        return source_logits, slot_logits, value, ownership_mask


# GNN + Self-Attention + Cross-Attention model
class ActorCriticGNN(nn.Module):
    def __init__(
        self,
        obs_config: ObsConfig,
        actions_per_source: int,
        max_sources: int,
        n_fractions: int = 5,
        model_config: ModelConfig = None,
        gnn_config: GNNConfig = None,
    ):
        super().__init__()
        config = model_config or DEFAULT_CONFIG.model
        gnn_cfg = gnn_config or config.gnn

        self.max_planets = obs_config.max_planets
        self.max_fleets = obs_config.max_fleets
        self.planet_features = obs_config.planet_features
        self.fleet_features = obs_config.fleet_features
        self.global_features = obs_config.global_features
        self.actions_per_source = actions_per_source
        self.max_sources = max_sources

        self.gcn_dims = list(gnn_cfg.gcn_dims)
        self.fleet_attn_dims = list(gnn_cfg.fleet_attn_dims)
        self.cross_attn_dims = list(gnn_cfg.cross_attn_dims)
        self.hg = self.gcn_dims[-1]                     # final GCN output dim
        self.hf = self.fleet_attn_dims[-1]               # final fleet attention output dim
        self.dropout = config.dropout

        # 2.1 Learnable adjacency matrix weight
        self.Wa = nn.Linear(self.planet_features, self.planet_features, bias=False)

        # 2.2 Graph convolution layers
        self.gcn_layers = nn.ModuleList()
        in_dim = self.planet_features
        for out_dim in self.gcn_dims:
            self.gcn_layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
            ))
            in_dim = out_dim

        # 3. Fleet self-attention (stacked layers, each with own QKV)
        self.fleet_attn_layers = nn.ModuleList()
        in_dim = self.fleet_features
        for out_dim in self.fleet_attn_dims:
            self.fleet_attn_layers.append(nn.ModuleDict({
                'q': nn.Linear(in_dim, out_dim),
                'k': nn.Linear(in_dim, out_dim),
                'v': nn.Linear(in_dim, out_dim),
            }))
            in_dim = out_dim

        # 4. Cross-attention (stacked layers, each with own QKV + residual projection)
        self.cross_attn_layers = nn.ModuleList()
        for out_dim in self.cross_attn_dims:
            self.cross_attn_layers.append(nn.ModuleDict({
                'q': nn.Linear(self.hg, out_dim),
                'k': nn.Linear(self.hf, out_dim),
                'v': nn.Linear(self.hf, out_dim),
                'proj': nn.Linear(out_dim, self.hg),
            }))

        # 5. Per-slot source selection via learned query vectors
        self.source_query = nn.Parameter(torch.randn(max_sources, self.hg) * 0.02)
        self.source_key = nn.Linear(self.hg, self.hg, bias=False)

        # 5b. Per-slot target scoring (separate from source — avoids
        #      autograd graph conflict from multiple log_softmax calls)
        self.target_query = nn.Parameter(torch.randn(max_sources, self.hg) * 0.02)

        # 6. Two-step action heads: STOP (independent) + fraction (target-dependent)
        self.stop_head = nn.Linear(self.hg, 1)
        with torch.no_grad():
            self.stop_head.bias[0] = -1.0  # bias toward launching

        self.fraction_head = nn.Sequential(
            nn.Linear(self.hg * 2, self.hg),
            nn.LayerNorm(self.hg),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hg, n_fractions),
        )
        self.n_fractions = n_fractions

        # 7. Value head — pools planet, fleet, and global features together.
        # Fleet features are mean-pooled (masked), concatenated with planet
        # mean-pool and global features, then fed through a small MLP.
        value_input_dim = self.hg + self.hf + self.global_features
        self.value_head = nn.Sequential(
            nn.Linear(value_input_dim, self.hg),
            nn.LayerNorm(self.hg),
            nn.GELU(),
            nn.Dropout(0.3),                         # heavier dropout → prevents overfitting
            nn.Linear(self.hg, self.hg // 2),
            nn.LayerNorm(self.hg // 2),
            nn.GELU(),
            nn.Linear(self.hg // 2, 1),
        )

    def forward(self, obs_flat):
        B = obs_flat.shape[0]

        # Split flat observation into structured tensors
        planet_dim = self.max_planets * self.planet_features
        fleet_dim = self.max_fleets * self.fleet_features

        planet_flat = obs_flat[:, :planet_dim]
        fleet_flat = obs_flat[:, planet_dim:planet_dim + fleet_dim]
        global_feat = obs_flat[:, planet_dim + fleet_dim:]

        Zp = planet_flat.reshape(B, self.max_planets, self.planet_features)
        Zf = fleet_flat.reshape(B, self.max_fleets, self.fleet_features)

        # Masks: distinguish real entities from zero-padding
        planet_mask = (Zp.abs().sum(dim=-1) > 1e-6).float()
        fleet_mask = (Zf.abs().sum(dim=-1) > 1e-6).float()

        # Ownership mask (is_me is feature index 1)
        ownership_mask = Zp[:, :, 1] > 0.5  # (B, max_planets)

        # -- 2. Planet GNN --
        Zp_t = self.Wa(Zp)
        Ag = torch.bmm(Zp_t, Zp_t.transpose(1, 2))

        # Mask and row-wise softmax normalization
        mask_2d = planet_mask.unsqueeze(1) * planet_mask.unsqueeze(2)
        Ag = Ag.masked_fill(mask_2d == 0, float('-inf'))
        Ag = torch.softmax(Ag, dim=-1).nan_to_num(0)

        # Graph convolution
        for gcn in self.gcn_layers:
            Zp = gcn(torch.bmm(Ag, Zp)).nan_to_num(0)

        # -- 3. Fleet self-attention (stacked layers) --
        for layer, d in zip(self.fleet_attn_layers, self.fleet_attn_dims):
            Qf = layer['q'](Zf)
            Kf = layer['k'](Zf)
            Vf = layer['v'](Zf)
            attn_f = torch.bmm(Qf, Kf.transpose(1, 2)) / (d ** 0.5)
            f_mask_2d = fleet_mask.unsqueeze(1) * fleet_mask.unsqueeze(2)
            attn_f = attn_f.masked_fill(f_mask_2d == 0, float('-inf'))
            attn_f = torch.softmax(attn_f, dim=-1).nan_to_num(0)
            Zf = torch.bmm(attn_f, Vf)

        # -- 4. Cross-attention: planets attend to fleets (stacked layers with residual) --
        for layer, d in zip(self.cross_attn_layers, self.cross_attn_dims):
            Q = layer['q'](Zp)
            K = layer['k'](Zf)
            V = layer['v'](Zf)
            attn_c = torch.bmm(Q, K.transpose(1, 2)) / (d ** 0.5)
            c_mask = planet_mask.unsqueeze(2) * fleet_mask.unsqueeze(1)
            attn_c = attn_c.masked_fill(c_mask == 0, float('-inf'))
            attn_c = torch.softmax(attn_c, dim=-1).nan_to_num(0)
            Za = torch.bmm(attn_c, V)
            Zp = layer['proj'](Za) + Zp  # residual back to hg

        # -- 5. Per-slot source selection via attention --
        # slot_query: (1, max_sources, hg) → expand to (B, max_sources, hg)
        Q_src = self.source_query.unsqueeze(0).expand(B, -1, -1)  # (B, max_sources, hg)
        K_src = self.source_key(Zp)  # (B, max_planets, hg)
        source_logits = torch.bmm(Q_src, K_src.transpose(1, 2)) / (self.hg ** 0.5)

        # Target scores — independent dot-product attention with separate query
        Q_tgt = self.target_query.unsqueeze(0).expand(B, -1, -1)  # (B, S, hg)
        target_scores = torch.bmm(Q_tgt, Zp.transpose(1, 2)) / (self.hg ** 0.5)
        # target_scores: (B, max_sources, max_planets)
 
        # -- 6. Slot embeddings + STOP head --
        src_mask = ownership_mask.unsqueeze(1).float()  # (B, 1, max_planets)
        src_attn = torch.softmax(
            source_logits.masked_fill(src_mask == 0, float('-inf')), dim=-1
        ).nan_to_num(0)  # (B, max_sources, max_planets)
        slot_embs = torch.bmm(src_attn, Zp)  # (B, max_sources, hg)
        stop_logits = self.stop_head(slot_embs)  # (B, max_sources, 1)

        # Fraction head is called per (source, chosen_target) later:
        #   frac_logits = fraction_head(cat([slot_embs[s], Zp[target_idx]]))

        # -- 7. Value head (mean-pool over planets) --
        Zp_pooled = (Zp * planet_mask.unsqueeze(-1)).sum(dim=1) / (
            planet_mask.sum(dim=1, keepdim=True) + 1e-8
        )  # (B, hg)
        Zf_pooled = (Zf * fleet_mask.unsqueeze(-1)).sum(dim=1) / (
            fleet_mask.sum(dim=1, keepdim=True) + 1e-8
        )  # (B, hf)
        value_input = torch.cat([Zp_pooled, Zf_pooled, global_feat], dim=-1)
        value = self.value_head(value_input)

        # -- 8. Pre-compute all (source, target) fraction logits in one pass --
        #     pair_inputs: (B, S, N, 2*hg)  →  fraction_head  →  (B, S, N, n_fractions)
        S = slot_embs.shape[1]; N = Zp.shape[1]; hg = slot_embs.shape[2]
        # Sanitise before building pair_inputs to prevent NaN in fraction logits
        slot_embs = slot_embs.nan_to_num(0.0)
        Zp = Zp.nan_to_num(0.0)
        pair_inputs = torch.cat([
            slot_embs.unsqueeze(2).expand(-1, -1, N, -1),
            Zp.unsqueeze(1).expand(-1, S, -1, -1),
        ], dim=-1)                                                           # (B, S, N, 2*hg)
        frac_logits_all = self.fraction_head(pair_inputs)                    # (B, S, N, n_frac)

        # NaN guard
        source_logits = source_logits.nan_to_num(0.0)
        target_scores = target_scores.nan_to_num(0.0)
        stop_logits = stop_logits.nan_to_num(0.0)
        frac_logits_all = frac_logits_all.nan_to_num(0.0)
        value = value.nan_to_num(0.0)
        return source_logits, target_scores, stop_logits, frac_logits_all, value, ownership_mask
