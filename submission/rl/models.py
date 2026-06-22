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
        n_offsets: int = 5,
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

        # 5. Per-slot source selection — state-conditioned planet scoring.
        #      Each planet gets an independent "quality" score from a small
        #      MLP, so ships / production / position auto-steer attention.
        #      Per-slot offsets add diversity across the 16 slots.
        self.source_scorer = nn.Sequential(
            nn.Linear(self.hg, self.hg),
            nn.LayerNorm(self.hg),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hg, 1),
        )
        self.source_slot_bias = nn.Parameter(
            torch.randn(max_sources, self.hg) * 1.0
        )
        self.source_key = nn.Linear(self.hg, self.hg, bias=False)

        # Explicit ships-weight for source selection.
        #   Simply biases source_logits toward planets with many ships,
        #   giving an immediate inductive bias that survives early PPO
        #   updates before the source_scorer MLP has converged.
        self.register_buffer('source_ships_weight', torch.tensor(2.0))

        # Sharper source attention — lower temperature makes slot_embs
        #   and src_pos concentrate on the primary source planet rather
        #   than a diluted average of all owned planets.
        self.source_attn_temp = 0.5

        # 5b. Per-slot target scoring — now source-conditioned:
        #      target query is computed from slot_embs, and Zp is projected
        #      through a separate target_key so source/target key spaces don't
        #      interfere.  No autograd conflict because slot_embs is a computed
        #      tensor, not a shared parameter.
        #      Cosine similarity (L2-normalised) prevents embedding explosion.
        self.target_query_proj = nn.Linear(self.hg, self.hg, bias=False)
        self.target_key = nn.Linear(self.hg, self.hg, bias=False)
        self.register_buffer('target_temperature', torch.tensor(3.0))

        # 5c. Fixed target bias — makes "any target" slightly positive
        #      by default, so the model explores different targets rather
        #      than collapsing to STOP when embedding scores drift negative.
        self.register_buffer('target_bias', torch.tensor(2.0))

        # 5d. Fixed distance bias for target scoring.
        #      Penalises far-away targets with a mild, constant weight.
        self.register_buffer('target_distance_scale', torch.tensor(0.05))

        # 6. STOP — learnable scalar threshold (not state-dependent).
        #      A single global bias competes against all target scores.
        #      The model can nudge it up/down through PPO, but because
        #      it lacks per-slot information it won't drown out targets.
        self.stop_bias = nn.Parameter(torch.tensor(-1.0))

        # 6b. Offset head — per-source scalar deciding how many extra ships
        #      to send on top of target_ships.  Input is the slot embedding (hg),
        #      output is logits over discrete offset bins [0, 3, 5, 10, 20].
        self.offset_head = nn.Sequential(
            nn.Linear(self.hg, self.hg),
            nn.LayerNorm(self.hg),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hg, n_offsets),
        )
        self.n_offsets = n_offsets

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

        # Save raw planet positions before GCN transforms them
        planet_pos = Zp[:, :, 4:6].clone()  # (B, N, 2) — x_norm, y_norm

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

        # -- 5. Per-slot source selection via state-conditioned scoring --
        #     planet_scores rates each planet independently via a small MLP
        #     so that ships / production / position steer the attention.
        planet_scores = self.source_scorer(Zp).squeeze(-1)          # (B, N)
        K_src = self.source_key(Zp)                                  # (B, N, hg)
        slot_offsets = (self.source_slot_bias.unsqueeze(0)           # (1, S, hg)
                         @ K_src.transpose(1, 2))                    # (B, S, N)
        slot_offsets = slot_offsets / (self.hg ** 0.5)
        source_logits = planet_scores.unsqueeze(1) + slot_offsets    # (B, S, N)

        # Explicit ships bias — planets with more ships score higher
        ships_feature = Zp[:, :, 7]                                    # (B, N) — ships_norm
        source_logits = (source_logits
                         + torch.abs(self.source_ships_weight)
                         * ships_feature.unsqueeze(1))

        # -- 6. Slot embeddings (source-conditioned) --
        #     Sharper attention (temperature 0.5) → slot_embs and src_pos
        #     concentrate on the primary source planet rather than a
        #     diluted mix of all owned planets.
        src_mask = ownership_mask.unsqueeze(1).float()  # (B, 1, max_planets)
        src_attn = torch.softmax(
            source_logits.masked_fill(src_mask == 0, float('-inf'))
            / self.source_attn_temp,
            dim=-1,
        ).nan_to_num(0)  # (B, max_sources, max_planets)
        slot_embs = torch.bmm(src_attn, Zp)  # (B, max_sources, hg)

        # -- 6b. Target scores via cosine similarity (bounded, safe) --
        Q_tgt = self.target_query_proj(slot_embs)          # (B, S, hg)
        K_tgt = self.target_key(Zp)                         # (B, N, hg)
        Q_tgt = Q_tgt / (Q_tgt.norm(dim=-1, keepdim=True) + 1e-8)  # unit
        K_tgt = K_tgt / (K_tgt.norm(dim=-1, keepdim=True) + 1e-8)  # unit
        target_scores = (torch.bmm(Q_tgt, K_tgt.transpose(1, 2))
                         * torch.abs(self.target_temperature))       # (B, S, N) in [-T, T]
        target_scores = target_scores + self.target_bias  # (B, S, N) — positive baseline

        # -- 6b-ii. Distance bias — penalise far-away targets --
        #     Soft source position per slot (weighted by src_attn),
        #     then Euclidean distance to every planet.
        src_pos = torch.bmm(src_attn, planet_pos)            # (B, S, 2)
        dist = torch.cdist(src_pos, planet_pos + 1e-8)       # (B, S, N) in [0, √2]
        target_scores = (target_scores
                         - torch.abs(self.target_distance_scale) * dist)

        # -- 6c. STOP logit — learnable scalar, no slot dependency --
        stop_logits = self.stop_bias.view(1, 1, 1).expand(B, self.max_sources, 1)

        # -- 7. Value head (mean-pool over planets) --
        Zp_pooled = (Zp * planet_mask.unsqueeze(-1)).sum(dim=1) / (
            planet_mask.sum(dim=1, keepdim=True) + 1e-8
        )  # (B, hg)
        Zf_pooled = (Zf * fleet_mask.unsqueeze(-1)).sum(dim=1) / (
            fleet_mask.sum(dim=1, keepdim=True) + 1e-8
        )  # (B, hf)
        value_input = torch.cat([Zp_pooled, Zf_pooled, global_feat], dim=-1)
        value = self.value_head(value_input)

        # -- 8. Per-source offset logits from slot embeddings --
        #     offset_head(slot_embs) → (B, S, n_offsets)
        slot_embs = slot_embs.nan_to_num(0.0)
        offset_logits = self.offset_head(slot_embs)                        # (B, S, n_offsets)

        # NaN guard
        source_logits = source_logits.nan_to_num(0.0)
        target_scores = target_scores.nan_to_num(0.0)
        stop_logits = stop_logits.nan_to_num(0.0)
        offset_logits = offset_logits.nan_to_num(0.0)
        value = value.nan_to_num(0.0)
        return source_logits, target_scores, stop_logits, offset_logits, value, ownership_mask
