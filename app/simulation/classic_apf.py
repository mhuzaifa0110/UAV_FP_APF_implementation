from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from .geometry import Obstacle, Vector3, add, mul, norm, normalize, sub
from .planner import is_segment_safe, limit_vertical_step


Bounds3D = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


@dataclass
class ClassicAPFConfig:
    k_att: float = 1.0
    k_rep: float = 70.0
    rho_d: float = 12.0
    step_size: float = 2.0
    goal_threshold: float = 1.0
    max_iterations: int = 900
    local_minimum_threshold: float = 1e-5
    collision_margin: float = 0.5
    drone_radius: float = 2.2
    max_vertical_step: float = 10.0
    segment_samples: int = 24


def clamp_to_bounds(point: Vector3, bounds: Bounds3D | None) -> Vector3:
    if bounds is None:
        return point
    return (
        min(max(point[0], bounds[0][0]), bounds[0][1]),
        min(max(point[1], bounds[1][0]), bounds[1][1]),
        min(max(point[2], bounds[2][0]), bounds[2][1]),
    )


def attractive_force(point: Vector3, goal: Vector3, cfg: ClassicAPFConfig) -> Vector3:
    return mul(sub(goal, point), cfg.k_att)


def attractive_potential(point: Vector3, goal: Vector3, cfg: ClassicAPFConfig) -> float:
    distance = norm(sub(point, goal))
    return 0.5 * cfg.k_att * distance**2


def repulsive_potential(
    point: Vector3,
    obstacle: Obstacle,
    cfg: ClassicAPFConfig,
) -> float:
    clearance = obstacle.clearance(point) - cfg.drone_radius
    if clearance >= cfg.rho_d:
        return 0.0
    safe_clearance = max(clearance, 1e-6)
    return 0.5 * cfg.k_rep * ((1.0 / safe_clearance) - (1.0 / cfg.rho_d)) ** 2


def repulsive_force(
    point: Vector3,
    obstacle: Obstacle,
    cfg: ClassicAPFConfig,
) -> Vector3:
    clearance = obstacle.clearance(point) - cfg.drone_radius
    if clearance >= cfg.rho_d:
        return (0.0, 0.0, 0.0)

    safe_clearance = max(clearance, 1e-6)
    direction = obstacle.direction_to_point(point)
    magnitude = cfg.k_rep * ((1.0 / safe_clearance) - (1.0 / cfg.rho_d)) / (safe_clearance**2)
    return mul(direction, magnitude)


def total_force(
    point: Vector3,
    goal: Vector3,
    obstacles: Sequence[Obstacle],
    cfg: ClassicAPFConfig,
) -> Vector3:
    force = attractive_force(point, goal, cfg)
    for obstacle in obstacles:
        force = add(force, repulsive_force(point, obstacle, cfg))
    return force


def total_potential(
    point: Vector3,
    goal: Vector3,
    obstacles: Sequence[Obstacle],
    cfg: ClassicAPFConfig,
) -> float:
    potential = attractive_potential(point, goal, cfg)
    for obstacle in obstacles:
        potential += repulsive_potential(point, obstacle, cfg)
    return potential


def _candidate_steps(point: Vector3, direction: Vector3, cfg: ClassicAPFConfig) -> tuple[Vector3, ...]:
    step = max(cfg.step_size, 1e-6)
    lateral = normalize((-direction[1], direction[0], 0.0))
    if norm(lateral) < 1e-9:
        lateral = (1.0, 0.0, 0.0)

    forward = add(point, mul(direction, step))
    return (
        forward,
        add(point, mul(direction, step * 0.5)),
        add(add(point, mul(direction, step * 0.75)), mul(lateral, step * 0.45)),
        add(add(point, mul(direction, step * 0.75)), mul(lateral, -step * 0.45)),
        add(add(point, mul(direction, step * 0.45)), (0.0, 0.0, step * 0.6)),
        add(add(point, mul(direction, step * 0.45)), (0.0, 0.0, -step * 0.35)),
    )


def _best_safe_candidate(
    current: Vector3,
    goal: Vector3,
    direction: Vector3,
    obstacles: Sequence[Obstacle],
    cfg: ClassicAPFConfig,
    bounds: Bounds3D | None,
) -> Vector3:
    best = current
    best_score = total_potential(current, goal, obstacles, cfg)
    for candidate in _candidate_steps(current, direction, cfg):
        candidate = clamp_to_bounds(candidate, bounds)
        candidate = limit_vertical_step(current, candidate, cfg)
        if not is_segment_safe(current, candidate, obstacles, cfg):
            continue
        clearance_bonus = min((obstacle.clearance(candidate) - cfg.drone_radius for obstacle in obstacles), default=0.0)
        score = total_potential(candidate, goal, obstacles, cfg) - 0.1 * clearance_bonus
        if score < best_score:
            best = candidate
            best_score = score
    return best


def classic_apf_plan(
    start: Vector3,
    goal: Vector3,
    obstacles: Sequence[Obstacle],
    cfg: ClassicAPFConfig,
    bounds: Bounds3D | None = None,
    seed: int | None = None,
) -> list[Vector3]:
    if seed is not None:
        random.seed(seed)

    current = clamp_to_bounds(start, bounds)
    goal = clamp_to_bounds(goal, bounds)
    path = [current]
    stalled_steps = 0

    for _ in range(cfg.max_iterations):
        distance_to_goal = norm(sub(current, goal))
        if distance_to_goal <= cfg.goal_threshold:
            if current != goal and is_segment_safe(current, goal, obstacles, cfg):
                path.append(goal)
            break

        force = total_force(current, goal, obstacles, cfg)
        force_norm = norm(force)
        if force_norm < cfg.local_minimum_threshold:
            goal_direction = normalize(sub(goal, current))
            random_direction = normalize(
                (
                    goal_direction[0] + random.uniform(-0.4, 0.4),
                    goal_direction[1] + random.uniform(-0.4, 0.4),
                    goal_direction[2] + random.uniform(-0.2, 0.5),
                )
            )
            direction = random_direction if norm(random_direction) > 0.0 else goal_direction
        else:
            direction = normalize(force)

        next_point = _best_safe_candidate(current, goal, direction, obstacles, cfg, bounds)
        if norm(sub(next_point, current)) < 1e-6:
            stalled_steps += 1
            direction = normalize(
                (
                    direction[0] + random.uniform(-0.8, 0.8),
                    direction[1] + random.uniform(-0.8, 0.8),
                    direction[2] + random.uniform(0.0, 0.8),
                )
            )
            next_point = _best_safe_candidate(current, goal, direction, obstacles, cfg, bounds)
            if norm(sub(next_point, current)) < 1e-6 or stalled_steps >= 12:
                break
        else:
            stalled_steps = 0

        current = next_point
        path.append(current)

    return path
