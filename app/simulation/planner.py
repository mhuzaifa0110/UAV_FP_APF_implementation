from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence

from .geometry import CylinderObstacle, Vector3, add, mul, norm, normalize, sub


@dataclass
class FPAPFConfig:
    k_att: float = 0.4
    k_rep: float = 10.0
    rho_d: float = 10.0
    gaussian_sigma: float = 4.0
    step_scale: float = 0.5
    goal_threshold: float = 1.0
    max_iterations: int = 400
    alpha: float = 0.35
    swarm_size: int = 40
    pso_iterations: int = 35
    inertia: float = 0.7
    c1: float = 1.5
    c2: float = 1.5
    init_sigma: float = 2.0
    velocity_sigma: float = 0.5
    obstacle_penalty_gain: float = 1000.0
    collision_margin: float = 15.5
    drone_radius: float = 2.2
    max_vertical_step: float = 3.0
    max_climb_ratio: float = 1.0
    segment_samples: int = 24
    repair_attempts: int = 8
    repair_shrink_factor: float = 0.72
    vertical_bias: float = 5.0


@dataclass
class Particle:
    position: Vector3
    velocity: Vector3
    best_position: Vector3
    best_fitness: float


def lerp(a: Vector3, b: Vector3, t: float) -> Vector3:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def random_gaussian_vector(sigma: float) -> Vector3:
    return (
        random.gauss(0.0, sigma),
        random.gauss(0.0, sigma),
        random.gauss(0.0, sigma),
    )


def attractive_force(q: Vector3, q_goal: Vector3, k_att: float) -> Vector3:
    return mul(sub(q_goal, q), k_att)


def gaussian_repulsive_force(
    point: Vector3,
    obstacle: CylinderObstacle,
    k_rep: float,
    sigma: float,
    rho_d: float,
    drone_radius: float,
) -> Vector3:
    rho_q = max(obstacle.clearance(point) - drone_radius, 1e-6)
    n_d = obstacle.direction_to_point(point)
    gaussian_term = math.exp(-(rho_q ** 2) / (2.0 * sigma ** 2))
    shaping_term = max((1.0 / rho_q) - (1.0 / max(rho_d, 1e-6)), 0.0)
    return mul(n_d, k_rep * gaussian_term * shaping_term)


def total_repulsive_force(
    point: Vector3,
    obstacles: Sequence[CylinderObstacle],
    cfg: FPAPFConfig,
) -> Vector3:
    force = (0.0, 0.0, 0.0)
    for obstacle in obstacles:
        force = add(
            force,
            gaussian_repulsive_force(
                point=point,
                obstacle=obstacle,
                k_rep=cfg.k_rep,
                sigma=cfg.gaussian_sigma,
                rho_d=cfg.rho_d,
                drone_radius=cfg.drone_radius,
            ),
        )
    return force


def obstacle_penalty(
    point: Vector3,
    obstacles: Sequence[CylinderObstacle],
    gain: float,
    margin: float,
    drone_radius: float,
) -> float:
    penalty = 0.0
    for obstacle in obstacles:
        clearance = obstacle.clearance(point) - drone_radius
        if clearance <= 0.0:
            penalty += gain * 1000.0
        elif clearance < margin:
            penalty += gain * ((margin - clearance) / margin) ** 2
    return penalty


def fitness(
    point: Vector3,
    goal: Vector3,
    obstacles: Sequence[CylinderObstacle],
    cfg: FPAPFConfig,
) -> float:
    return norm(sub(point, goal)) + obstacle_penalty(
        point=point,
        obstacles=obstacles,
        gain=cfg.obstacle_penalty_gain,
        margin=cfg.collision_margin,
        drone_radius=cfg.drone_radius,
    )


def point_clearance(point: Vector3, obstacles: Sequence[CylinderObstacle], cfg: FPAPFConfig) -> float:
    if not obstacles:
        return float("inf")
    return min(obstacle.clearance(point) - cfg.drone_radius for obstacle in obstacles)


def segment_min_clearance(
    start: Vector3,
    end: Vector3,
    obstacles: Sequence[CylinderObstacle],
    cfg: FPAPFConfig,
) -> float:
    min_clearance = float("inf")
    sample_count = max(cfg.segment_samples, 2)
    for idx in range(sample_count + 1):
        point = lerp(start, end, idx / sample_count)
        min_clearance = min(min_clearance, point_clearance(point, obstacles, cfg))
    return min_clearance


def is_segment_safe(
    start: Vector3,
    end: Vector3,
    obstacles: Sequence[CylinderObstacle],
    cfg: FPAPFConfig,
) -> bool:
    return segment_min_clearance(start, end, obstacles, cfg) >= cfg.collision_margin


def limit_vertical_step(current: Vector3, candidate: Vector3, cfg: FPAPFConfig) -> Vector3:
    max_step = max(cfg.max_vertical_step, 1e-6)
    max_climb_ratio = max(cfg.max_climb_ratio, 1e-6)
    horizontal_step = math.hypot(candidate[0] - current[0], candidate[1] - current[1])
    dz = candidate[2] - current[2]
    max_slope_step = horizontal_step * max_climb_ratio
    allowed_dz = min(max_step, max_slope_step)
    if abs(dz) <= allowed_dz:
        return candidate
    limited_z = current[2] + math.copysign(allowed_dz, dz)
    return (candidate[0], candidate[1], limited_z)


def lateral_unit(direction: Vector3) -> Vector3:
    lateral = (-direction[1], direction[0], 0.0)
    if norm(lateral) < 1e-9:
        lateral = (1.0, 0.0, 0.0)
    return normalize(lateral)


def choose_safer_candidate(
    current: Vector3,
    proposed: Vector3,
    goal: Vector3,
    obstacles: Sequence[CylinderObstacle],
    cfg: FPAPFConfig,
) -> Vector3:
    proposed = limit_vertical_step(current, proposed, cfg)
    if is_segment_safe(current, proposed, obstacles, cfg):
        return proposed

    motion = sub(proposed, current)
    motion_norm = norm(motion)
    if motion_norm < 1e-9:
        return current

    direction = normalize(motion)
    lateral = lateral_unit(direction)
    vertical = (0.0, 0.0, 1.0)

    best_candidate: Vector3 | None = None
    best_score = float("inf")

    shrink = 1.0
    for attempt in range(cfg.repair_attempts):
        shrink *= cfg.repair_shrink_factor
        base = add(current, mul(direction, motion_norm * shrink))
        variants = [
            base,
            add(add(base, mul(vertical, cfg.vertical_bias * 0.6 * shrink)), mul(lateral, motion_norm * 0.35 * shrink)),
            add(add(base, mul(vertical, cfg.vertical_bias * 0.6 * shrink)), mul(lateral, -motion_norm * 0.35 * shrink)),
            add(base, mul(lateral, motion_norm * 0.22 * shrink)),
            add(base, mul(lateral, -motion_norm * 0.22 * shrink)),
        ]

        for variant in variants:
            candidate = limit_vertical_step(current, variant, cfg)
            if not is_segment_safe(current, candidate, obstacles, cfg):
                continue
            clearance_bonus = segment_min_clearance(current, candidate, obstacles, cfg)
            score = fitness(candidate, goal, obstacles, cfg) - 0.15 * clearance_bonus
            if score < best_score:
                best_score = score
                best_candidate = candidate

    if best_candidate is not None:
        return best_candidate

    safe_progress = current
    for attempt in range(1, cfg.segment_samples + 1):
        fraction = attempt / cfg.segment_samples
        candidate = limit_vertical_step(current, lerp(current, proposed, fraction), cfg)
        if is_segment_safe(current, candidate, obstacles, cfg):
            safe_progress = candidate
        else:
            break
    return safe_progress


def initialize_particles(
    x_apf: Vector3,
    goal: Vector3,
    obstacles: Sequence[CylinderObstacle],
    cfg: FPAPFConfig,
) -> List[Particle]:
    particles: List[Particle] = []
    for _ in range(cfg.swarm_size):
        position = add(x_apf, random_gaussian_vector(cfg.init_sigma))
        velocity = random_gaussian_vector(cfg.velocity_sigma)
        fit = fitness(position, goal, obstacles, cfg)
        particles.append(
            Particle(
                position=position,
                velocity=velocity,
                best_position=position,
                best_fitness=fit,
            )
        )
    return particles


def pso_refine(
    x_apf: Vector3,
    goal: Vector3,
    obstacles: Sequence[CylinderObstacle],
    cfg: FPAPFConfig,
) -> Vector3:
    particles = initialize_particles(x_apf, goal, obstacles, cfg)
    g_best_particle = min(particles, key=lambda particle: particle.best_fitness)
    g_best = g_best_particle.best_position
    g_best_fitness = g_best_particle.best_fitness

    for _ in range(cfg.pso_iterations):
        for particle in particles:
            r1 = random.random()
            r2 = random.random()

            inertia_term = mul(particle.velocity, cfg.inertia)
            cognitive_term = mul(sub(particle.best_position, particle.position), cfg.c1 * r1)
            social_term = mul(sub(g_best, particle.position), cfg.c2 * r2)

            particle.velocity = add(add(inertia_term, cognitive_term), social_term)
            particle.position = add(particle.position, particle.velocity)

            current_fitness = fitness(particle.position, goal, obstacles, cfg)
            if current_fitness < particle.best_fitness:
                particle.best_position = particle.position
                particle.best_fitness = current_fitness

            if particle.best_fitness < g_best_fitness:
                g_best = particle.best_position
                g_best_fitness = particle.best_fitness

    return g_best


def fp_apf_plan(
    start: Vector3,
    goal: Vector3,
    obstacles: Sequence[CylinderObstacle],
    cfg: FPAPFConfig,
    seed: int | None = None,
) -> List[Vector3]:
    if seed is not None:
        random.seed(seed)

    current = start
    path = [current]

    for _ in range(cfg.max_iterations):
        if norm(sub(current, goal)) <= cfg.goal_threshold:
            break

        f_att = attractive_force(current, goal, cfg.k_att)
        f_rep = total_repulsive_force(current, obstacles, cfg)
        f_total = add(f_att, f_rep)

        direction = normalize(f_total)
        if norm(direction) < 1e-12:
            direction = normalize(sub(goal, current))

        x_apf = add(current, mul(direction, cfg.step_scale))
        g_best = pso_refine(x_apf, goal, obstacles, cfg)
        next_point = add(mul(x_apf, cfg.alpha), mul(g_best, 1.0 - cfg.alpha))
        next_point = limit_vertical_step(current, next_point, cfg)
        next_point = choose_safer_candidate(current, next_point, goal, obstacles, cfg)

        if norm(sub(next_point, current)) < 1e-6:
            escape_direction = normalize(sub(goal, current))
            escape = limit_vertical_step(current, add(current, mul(escape_direction, max(cfg.step_scale, 1.0))), cfg)
            if is_segment_safe(current, escape, obstacles, cfg):
                next_point = escape
            else:
                break

        current = next_point
        path.append(current)

        if norm(sub(current, goal)) <= cfg.goal_threshold and is_segment_safe(current, goal, obstacles, cfg):
            path.append(goal)
            break

    return path
