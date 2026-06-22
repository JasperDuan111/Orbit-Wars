import math
from typing import Optional

import numpy as np
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
from .config import DEFAULT_CONFIG, GameConfig, ObsConfig


def _get_field(obs, name, default=None):
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def _log_scale(value, max_value=1000.0):
    return math.log(value + 1.0) / math.log(max_value + 1.0)


def parse_entities(obs):
    raw_planets = _get_field(obs, "planets", [])
    raw_fleets = _get_field(obs, "fleets", [])
    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    return planets, fleets


def ship_totals(obs, player_id):
    planets, fleets = parse_entities(obs)
    my_total = 0.0
    enemy_total = 0.0
    for p in planets:
        if p.owner == player_id:
            my_total += p.ships
        elif p.owner != -1:
            enemy_total += p.ships
    for f in fleets:
        if f.owner == player_id:
            my_total += f.ships
        elif f.owner != -1:
            enemy_total += f.ships
    return my_total, enemy_total


def encode_observation(
    obs,
    max_planets: Optional[int] = None,
    max_fleets: Optional[int] = None,
    obs_config: Optional[ObsConfig] = None,
    game_config: Optional[GameConfig] = None,
    episode_steps: int = 500,
):
    obs_config = obs_config or DEFAULT_CONFIG.obs
    game_config = game_config or DEFAULT_CONFIG.game
    if max_planets is None:
        max_planets = obs_config.max_planets
    if max_fleets is None:
        max_fleets = obs_config.max_fleets
    player_id = _get_field(obs, "player", 0)
    planets, fleets = parse_entities(obs)
    comet_ids = set(_get_field(obs, "comet_planet_ids", []))
    board_size = game_config.board_size
    sun_x = game_config.center_x
    sun_y = game_config.center_y
    sun_radius = game_config.sun_radius
    sun_radius_norm = sun_radius / board_size

    planet_features = np.zeros((max_planets, obs_config.planet_features), dtype=np.float32)
    fleet_features = np.zeros((max_fleets, obs_config.fleet_features), dtype=np.float32)

    def _planet_sort_key(p):
        is_me = 1 if p.owner == player_id else 0
        is_enemy = 1 if (p.owner != player_id and p.owner != -1) else 0
        dist = math.hypot(p.x - sun_x, p.y - sun_y)
        return (-is_me, -is_enemy, dist)

    for idx, p in enumerate(sorted(planets, key=_planet_sort_key)[:max_planets]):
        is_me = 1.0 if p.owner == player_id else 0.0
        is_neutral = 1.0 if p.owner == -1 else 0.0
        is_enemy = 1.0 if (p.owner != player_id and p.owner != -1) else 0.0
        x_norm = p.x / board_size
        y_norm = p.y / board_size
        radius_norm = p.radius / board_size
        ships_norm = _log_scale(p.ships)
        production_norm = p.production / 5.0
        is_comet = 1.0 if p.id in comet_ids else 0.0
        dx_to_sun = (p.x - sun_x) / board_size
        dy_to_sun = (p.y - sun_y) / board_size
        dist = math.hypot(p.x - sun_x, p.y - sun_y)
        clearance_to_sun = (dist - sun_radius - p.radius) / board_size
        is_inner = 1.0 if dist + p.radius < 50.0 else 0.0
        planet_id_norm = p.id / max_planets
        planet_features[idx] = [
            planet_id_norm,
            is_me,
            is_enemy,
            is_neutral,
            x_norm,
            y_norm,
            radius_norm,
            ships_norm,
            production_norm,
            is_comet,
            is_inner,
            dx_to_sun,
            dy_to_sun,
            clearance_to_sun,
        ]

    for idx, f in enumerate(sorted(fleets, key=lambda item: (item.x, item.y))[:max_fleets]):
        is_me = 1.0 if f.owner == player_id else 0.0
        is_enemy = 1.0 if f.owner != player_id else 0.0
        x_norm = f.x / board_size
        y_norm = f.y / board_size
        cos_a = math.cos(f.angle)
        sin_a = math.sin(f.angle)
        ships_norm = _log_scale(f.ships)
        fleet_id_norm = f.id / max_fleets
        from_planet_norm = f.from_planet_id / max_planets
        dx_sun_ahead = (sun_x - f.x) / board_size
        dy_sun_ahead = (sun_y - f.y) / board_size
        sun_forward = dx_sun_ahead * cos_a + dy_sun_ahead * sin_a
        sun_lateral = abs(dx_sun_ahead * sin_a - dy_sun_ahead * cos_a)
        sun_clearance = sun_lateral - sun_radius_norm
        fleet_features[idx] = [
            fleet_id_norm,
            from_planet_norm,
            is_me,
            is_enemy,
            x_norm,
            y_norm,
            cos_a,
            sin_a,
            ships_norm,
            sun_forward,
            sun_lateral,
            sun_clearance,
        ]

    step = _get_field(obs, "step", 0)
    my_total, enemy_total = ship_totals(obs, player_id)
    num_my_planets = sum(1 for p in planets if p.owner == player_id)
    num_enemy_planets = sum(1 for p in planets if p.owner not in (-1, player_id))
    num_neutral = sum(1 for p in planets if p.owner == -1)

    global_features = np.array(
        [
            step / max(episode_steps, 1),
            num_my_planets / max_planets,
            num_enemy_planets / max_planets,
            num_neutral / max_planets,
            _log_scale(my_total),
            _log_scale(enemy_total),
            sun_x / board_size,
            sun_y / board_size,
            sun_radius_norm,
        ],
        dtype=np.float32,
    )

    obs_vector = np.concatenate(
        [planet_features.reshape(-1), fleet_features.reshape(-1), global_features],
        axis=0,
    )
    if obs_vector.shape[0] != (
        max_planets * obs_config.planet_features
        + max_fleets * obs_config.fleet_features
        + obs_config.global_features
    ):
        raise ValueError("Unexpected observation size.")
    return obs_vector
