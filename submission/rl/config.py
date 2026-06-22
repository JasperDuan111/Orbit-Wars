import dataclasses

import yaml
from dataclasses import dataclass, field
from typing import Optional, Tuple


def _from_dict(cls, data: dict):
    """Build a dataclass from a dict, using defaults for missing keys."""
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            val = data[f.name]
            origin = getattr(f.type, "__origin__", None)
            if origin is tuple:
                val = tuple(val)
            kwargs[f.name] = val
    return cls(**kwargs)


@dataclass(frozen=True)
class GameConfig:
    board_size: float = 100.0
    center_x: float = 50.0
    center_y: float = 50.0
    sun_radius: float = 10.0


@dataclass(frozen=True)
class ObsConfig:
    max_planets: int = 48
    max_fleets: int = 64
    planet_features: int = 14
    fleet_features: int = 12
    global_features: int = 9

    @property
    def obs_dim(self) -> int:
        return (
            self.max_planets * self.planet_features
            + self.max_fleets * self.fleet_features
            + self.global_features
        )


@dataclass(frozen=True)
class ActionSpaceConfig:
    max_sources: int = 16
    max_targets: int = 16
    offset_bins: Tuple[float, ...] = (0, 3, 5, 10, 20)
    max_launches_per_source: int = 3
    epsilon: float = 0.1                     # ε-greedy exploration rate (0 = pure argmax, 1 = pure random)

    @property
    def n_offsets(self) -> int:
        return len(self.offset_bins)

    @property
    def actions_per_source(self) -> int:
        return 1 + self.max_targets


@dataclass(frozen=True)
class GNNConfig:
    gcn_dims: Tuple[int, ...] = (128, 128)           # GCN output dims, one entry = one layer
    fleet_attn_dims: Tuple[int, ...] = (128,)         # Fleet self-attention projection dims
    cross_attn_dims: Tuple[int, ...] = (64,)          # Cross-attention projection dims


@dataclass(frozen=True)
class ModelConfig:
    model_type: str = "mlp"  # "mlp" or "gnn"
    hidden_sizes: Tuple[int, ...] = (512, 512, 256)
    dropout: float = 0.1
    gnn: GNNConfig = field(default_factory=GNNConfig)


@dataclass
class RewardConfig:
    # ── Economy mode: dense fleet advantage per step ──
    #     reward = (my_fleets - enemy_fleets) * scale
    fleet_advantage_scale: float = 1e-4

    # ── Economy mode: dense production advantage per step ──
    #     reward = (my_prod * log(my_prod) - enemy_prod * log(enemy_prod)) * scale
    production_advantage_scale: float = 0.01

    # ── Economy mode: terminal fleet efficiency ──
    #     reward = my_fleets / total_board_production * scale
    terminal_economy_scale: float = 0.1

    out_of_boundary_penalty_scale: float = 5e-4
    suicide_penalty_scale: float = 1e-3

    # ── Legacy dense: planet-count advantage per step ──
    #     reward = (my_planets − enemy_planets) × scale
    planet_count_scale: float = 0.03

    # ── Sparse events: planet capture / loss ──
    territory_scale: float = 3.0              # base reward per capture / loss
    production_weight: float = 0.5            # quality multiplier: 1 + prod × weight
    enemy_capture_bonus: float = 3.0          # extra multiplier for enemy→me captures (vs neutral baseline)
    territory_loss_penalty: float = 2.0        # extra multiplier for me→enemy losses (vs me→neutral baseline)

    # ── No-action penalty: applied when 0 launches in a step ──
    no_action_penalty_scale: float = 0.03   # penalty per idle step after grace
    no_action_grace_steps: int = 5          # consecutive idle steps before penalty

    # ── Launch bonus: immediate positive signal per launch ──
    launch_bonus_scale: float = 5.0e-4      # +5e-4 per ship launched (not per-launch)

    # ── Fleet combat events: bridge launch → capture gap ──
    defense_success_scale: float = 0.005    # +0.005 per enemy ship destroyed while defending own planet
    fleet_arrival_scale: float = 0.005      # +0.005 per ship that reached enemy/neutral planet

    # ── Terminal  (asymmetric) ──
    terminal_win_scale: float = 15.0
    terminal_lose_scale: float = -15.0

    # ── Invalid-action penalty (applied by env wrapper) ──
    invalid_action_penalty: float = 0.1

    only_economy: bool = False


@dataclass
class EnvConfig:
    num_players: int = 2
    episode_steps: int = 500
    act_timeout: int = 1
    seed: Optional[int] = None
    debug: bool = False


@dataclass
class TrainConfig:
    seed: int = 42
    num_envs: int = 10
    rollout_steps: int = 64
    total_updates: int = 2000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.3
    learning_rate: float = 2e-4
    ent_coef: float = 0.05
    vf_coef: float = 1.0
    value_reg: float = 0.01
    max_grad_norm: float = 0.5
    batch_size: int = 128
    epochs: int = 3
    save_every: int = 20
    opponent_refresh: int = 3
    opponent_pool_capacity: int = 10
    opponent_rule_prob: float = 0.3


@dataclass
class OrbitWarsConfig:
    game: GameConfig = field(default_factory=GameConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    action: ActionSpaceConfig = field(default_factory=ActionSpaceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def obs_dim(self) -> int:
        return self.obs.obs_dim

    @property
    def actions_per_source(self) -> int:
        return self.action.actions_per_source

    @classmethod
    def from_yaml(cls, path: str) -> "OrbitWarsConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        model_data = data.get("model", {})
        gnn_data = model_data.pop("gnn", {})
        model_dict = _from_dict(ModelConfig, model_data).__dict__
        model_dict.pop("gnn", None)  # remove default; replaced by parsed gnn_data
        return cls(
            game=_from_dict(GameConfig, data.get("game", {})),
            obs=_from_dict(ObsConfig, data.get("obs", {})),
            action=_from_dict(ActionSpaceConfig, data.get("action", {})),
            model=ModelConfig(
                **model_dict,
                gnn=_from_dict(GNNConfig, gnn_data),
            ),
            reward=_from_dict(RewardConfig, data.get("reward", {})),
            env=_from_dict(EnvConfig, data.get("env", {})),
            train=_from_dict(TrainConfig, data.get("train", {})),
        )


DEFAULT_CONFIG = OrbitWarsConfig()

BOARD_SIZE = DEFAULT_CONFIG.game.board_size
CENTER_X = DEFAULT_CONFIG.game.center_x
CENTER_Y = DEFAULT_CONFIG.game.center_y

MAX_PLANETS = DEFAULT_CONFIG.obs.max_planets
MAX_FLEETS = DEFAULT_CONFIG.obs.max_fleets
PLANET_FEATURES = DEFAULT_CONFIG.obs.planet_features
FLEET_FEATURES = DEFAULT_CONFIG.obs.fleet_features
GLOBAL_FEATURES = DEFAULT_CONFIG.obs.global_features

MAX_SOURCES = DEFAULT_CONFIG.action.max_sources
MAX_TARGETS = DEFAULT_CONFIG.action.max_targets
OFFSET_BINS = DEFAULT_CONFIG.action.offset_bins
MAX_LAUNCHES_PER_SOURCE = DEFAULT_CONFIG.action.max_launches_per_source

MODEL_HIDDEN_SIZES = DEFAULT_CONFIG.model.hidden_sizes

ACTIONS_PER_SOURCE = DEFAULT_CONFIG.action.actions_per_source
OBS_DIM = DEFAULT_CONFIG.obs.obs_dim
