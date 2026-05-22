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


@dataclass(frozen=True)
class ObsConfig:
    max_planets: int = 48
    max_fleets: int = 64
    planet_features: int = 11
    fleet_features: int = 9
    global_features: int = 6

    @property
    def obs_dim(self) -> int:
        return (
            self.max_planets * self.planet_features
            + self.max_fleets * self.fleet_features
            + self.global_features
        )


@dataclass(frozen=True)
class ActionSpaceConfig:
    max_sources: int = 8
    max_targets: int = 20
    ship_fractions: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    max_launches_per_source: int = 6

    @property
    def actions_per_source(self) -> int:
        return 1 + self.max_targets * len(self.ship_fractions)


@dataclass(frozen=True)
class GNNConfig:
    hg: int = 128
    hf: int = 128
    ha: int = 64
    num_gcn_layers: int = 2


@dataclass(frozen=True)
class ModelConfig:
    model_type: str = "mlp"  # "mlp" or "gnn"
    hidden_sizes: Tuple[int, ...] = (512, 512, 256)
    dropout: float = 0.1
    gnn: GNNConfig = field(default_factory=GNNConfig)


@dataclass
class RewardConfig:
    reward_scale: float = 0.01
    invalid_action_penalty: float = 0.05
    terminal_reward_scale: float = 0.01
    win_reward: float = 100.0
    lose_penalty: float = -100.0
    planet_control_scale: float = 10
    production_scale: float = 1
    survival_reward: float = 0.01


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
    total_updates: int = 2000
    rollout_steps: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    learning_rate: float = 3e-4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 64
    epochs: int = 4
    save_every: int = 50
    opponent_refresh: int = 10


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
        return cls(
            game=_from_dict(GameConfig, data.get("game", {})),
            obs=_from_dict(ObsConfig, data.get("obs", {})),
            action=_from_dict(ActionSpaceConfig, data.get("action", {})),
            model=ModelConfig(
                **_from_dict(ModelConfig, model_data).__dict__,
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
SHIP_FRACTIONS = DEFAULT_CONFIG.action.ship_fractions
MAX_LAUNCHES_PER_SOURCE = DEFAULT_CONFIG.action.max_launches_per_source

MODEL_HIDDEN_SIZES = DEFAULT_CONFIG.model.hidden_sizes

ACTIONS_PER_SOURCE = DEFAULT_CONFIG.action.actions_per_source
OBS_DIM = DEFAULT_CONFIG.obs.obs_dim
