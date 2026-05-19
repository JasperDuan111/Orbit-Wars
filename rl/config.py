from dataclasses import dataclass, field
from typing import Optional, Tuple


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
class ModelConfig:
    hidden_sizes: Tuple[int, ...] = (512, 512, 256)
    dropout: float = 0.1


@dataclass
class RewardConfig:
    reward_scale: float = 0.01
    invalid_action_penalty: float = 0.05
    terminal_reward_scale: float = 0.01
    planet_control_scale: float = 10
    production_scale: float = 1
    survival_reward: float = 100


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
