from dataclasses import dataclass

BOARD_SIZE = 100.0
CENTER_X = 50.0
CENTER_Y = 50.0

MAX_PLANETS = 48
MAX_FLEETS = 64

PLANET_FEATURES = 10
FLEET_FEATURES = 7
GLOBAL_FEATURES = 6

MAX_SOURCES = 8
MAX_TARGETS = 8
SHIP_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
MAX_LAUNCHES_PER_SOURCE = 6
NUM_ENVS = 2
MODEL_HIDDEN_SIZES = (512, 512, 256)

ACTIONS_PER_SOURCE = 1 + MAX_TARGETS * len(SHIP_FRACTIONS)
OBS_DIM = MAX_PLANETS * PLANET_FEATURES + MAX_FLEETS * FLEET_FEATURES + GLOBAL_FEATURES


@dataclass
class PPOConfig:
    seed: int = 42
    num_envs: int = NUM_ENVS
    max_launches_per_source: int = MAX_LAUNCHES_PER_SOURCE
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
    reward_scale: float = 0.01
    invalid_action_penalty: float = 0.05
    terminal_reward_scale: float = 0.01
