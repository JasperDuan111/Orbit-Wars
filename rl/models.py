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
        return source_logits, slot_logits, value, ownership_mask


# GNN + Self-Attention + Cross-Attention model
class ActorCriticGNN(nn.Module):
    def __init__(
        self,
        obs_config: ObsConfig,
        actions_per_source: int,
        max_sources: int,
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

        self.hg = gnn_cfg.hg
        self.hf = gnn_cfg.hf
        self.ha = gnn_cfg.ha
        self.num_gcn_layers = gnn_cfg.num_gcn_layers
        self.dropout = config.dropout

        # 2.1 Learnable adjacency matrix weight
        self.Wa = nn.Linear(self.planet_features, self.planet_features, bias=False)

        # 2.2 Graph convolution layers
        self.gcn_layers = nn.ModuleList()
        in_dim = self.planet_features
        for _ in range(self.num_gcn_layers):
            self.gcn_layers.append(nn.Sequential(
                nn.Linear(in_dim, self.hg),
                nn.LayerNorm(self.hg),
                nn.GELU(),
                nn.Dropout(self.dropout),
            ))
            in_dim = self.hg

        # 3. Fleet self-attention QKV projections
        self.fleet_q = nn.Linear(self.fleet_features, self.hf)
        self.fleet_k = nn.Linear(self.fleet_features, self.hf)
        self.fleet_v = nn.Linear(self.fleet_features, self.hf)

        # 4. Cross-attention QKV projections
        self.cross_q = nn.Linear(self.hg, self.ha)
        self.cross_k = nn.Linear(self.hf, self.ha)
        self.cross_v = nn.Linear(self.hf, self.ha)

        # 4.3 Residual projection
        self.Wp = nn.Linear(self.ha, self.hg)

        # 5. Per-slot source selection via learned query vectors
        self.source_query = nn.Parameter(torch.randn(max_sources, self.hg) * 0.02)
        self.source_key = nn.Linear(self.hg, self.hg, bias=False)

        # 6. Per-slot target+fraction head (shared across slots)
        self.slot_policy_head = nn.Sequential(
            nn.Linear(self.hg, self.hg),
            nn.LayerNorm(self.hg),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hg, actions_per_source),
        )

        # 7. Value head (still mean-pooled)
        self.value_head = nn.Sequential(
            nn.Linear(self.hg, self.hg),
            nn.LayerNorm(self.hg),
            nn.GELU(),
            nn.Linear(self.hg, 1),
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
            Zp = gcn(torch.bmm(Ag, Zp))

        # -- 3. Fleet self-attention --
        Qf = self.fleet_q(Zf)
        Kf = self.fleet_k(Zf)
        Vf = self.fleet_v(Zf)

        attn_f = torch.bmm(Qf, Kf.transpose(1, 2)) / (self.hf ** 0.5)
        f_mask_2d = fleet_mask.unsqueeze(1) * fleet_mask.unsqueeze(2)
        attn_f = attn_f.masked_fill(f_mask_2d == 0, float('-inf'))
        attn_f = torch.softmax(attn_f, dim=-1).nan_to_num(0)
        Zf = torch.bmm(attn_f, Vf)

        # -- 4. Cross-attention: planets attend to fleets --
        Q = self.cross_q(Zp)
        K = self.cross_k(Zf)
        V = self.cross_v(Zf)

        attn_c = torch.bmm(Q, K.transpose(1, 2)) / (self.ha ** 0.5)
        c_mask = planet_mask.unsqueeze(2) * fleet_mask.unsqueeze(1)
        attn_c = attn_c.masked_fill(c_mask == 0, float('-inf'))
        attn_c = torch.softmax(attn_c, dim=-1).nan_to_num(0)
        Za = torch.bmm(attn_c, V)

        # 4.3 Residual connection
        Z = self.Wp(Za) + Zp  # (B, max_planets, hg)

        # -- 5. Per-slot source selection via attention --
        # slot_query: (1, max_sources, hg) → expand to (B, max_sources, hg)
        Q_src = self.source_query.unsqueeze(0).expand(B, -1, -1)  # (B, max_sources, hg)
        K_src = self.source_key(Z)  # (B, max_planets, hg)
        source_logits = torch.bmm(Q_src, K_src.transpose(1, 2)) / (self.hg ** 0.5)
        # source_logits: (B, max_sources, max_planets)

        # -- 6. Per-slot target+fraction logits --
        # Soft-attend over planets per slot to get slot embedding
        # Mask to owned planets for source attention
        src_mask = ownership_mask.unsqueeze(1).float()  # (B, 1, max_planets)
        src_attn = torch.softmax(
            source_logits.masked_fill(src_mask == 0, float('-inf')), dim=-1
        ).nan_to_num(0)  # (B, max_sources, max_planets)
        slot_embs = torch.bmm(src_attn, Z)  # (B, max_sources, hg)

        # Shared MLP over each slot embedding → (B, max_sources, actions_per_source)
        slot_logits = self.slot_policy_head(slot_embs)

        # -- 7. Value head (mean-pool over planets) --
        Z_pooled = (Z * planet_mask.unsqueeze(-1)).sum(dim=1) / (
            planet_mask.sum(dim=1, keepdim=True) + 1e-8
        )
        value = self.value_head(Z_pooled)

        return source_logits, slot_logits, value, ownership_mask
