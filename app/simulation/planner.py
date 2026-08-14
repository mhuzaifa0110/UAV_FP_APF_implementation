from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence

from .geometry import Obstacle, Vector3, add, mul, norm, normalize, sub


@dataclass
class FPAPFConfig:
    k_att: float = 3.4
    k_rep: float = 6.0
    rho_d: float = 6.0
    gaussian_sigma: float = 4.0
    step_scale: float = 0.5
    goal_threshold: float = 1.0
    max_iterations: int = 400
    alpha: float = 0.35
    swarm_size: int = 40
    pso_iterations: int = 35
    inertia: float = 0.5
    c1: float = 1.5
    c2: float = 1.5
    init_sigma: float = 2.0
    velocity_sigma: float = 0.5
    obstacle_penalty_gain: float = 300.0
    collision_margin: float = 0.5
    drone_radius: float = 2.2
    max_vertical_step: float = 15.0     # vertical change allowed in one step
    heading_smoothness_gain: float = 0.8 #penalty for sharp turns
    smoothing_passes: int = 2
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


def climb_guidance_goal(current: Vector3, goal: Vector3, cfg: FPAPFConfig) -> Vector3:
    return goal


def gaussian_repulsive_force(
    point: Vector3,
    obstacle: Obstacle,
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
    obstacles: Sequence[Obstacle],
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
    obstacles: Sequence[Obstacle],
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
    obstacles: Sequence[Obstacle],
    cfg: FPAPFConfig,
    previous: Vector3 | None = None,
    current: Vector3 | None = None,
) -> float:
    return norm(sub(point, goal)) + obstacle_penalty(
        point=point,
        obstacles=obstacles,
        gain=cfg.obstacle_penalty_gain,
        margin=cfg.collision_margin,
        drone_radius=cfg.drone_radius,
    ) + heading_smoothness_penalty(previous, current, point, cfg)


def heading_smoothness_penalty(
    previous: Vector3 | None,
    current: Vector3 | None,
    candidate: Vector3,
    cfg: FPAPFConfig,
) -> float:
    if previous is None or current is None:
        return 0.0

    incoming = sub(current, previous)
    outgoing = sub(candidate, current)
    incoming_norm = norm(incoming)
    outgoing_norm = norm(outgoing)
    if incoming_norm < 1e-9 or outgoing_norm < 1e-9:
        return 0.0

    dot = sum(a * b for a, b in zip(incoming, outgoing)) / (incoming_norm * outgoing_norm)
    dot = max(-1.0, min(1.0, dot))
    return cfg.heading_smoothness_gain * (1.0 - dot) * outgoing_norm


def point_clearance(point: Vector3, obstacles: Sequence[Obstacle], cfg: FPAPFConfig) -> float:
    if not obstacles:
        return float("inf")
    return min(obstacle.clearance(point) - cfg.drone_radius for obstacle in obstacles)


def segment_min_clearance(
    start: Vector3,
    end: Vector3,
    obstacles: Sequence[Obstacle],
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
    obstacles: Sequence[Obstacle],
    cfg: FPAPFConfig,
) -> bool:
    return segment_min_clearance(start, end, obstacles, cfg) >= cfg.collision_margin


def is_motion_feasible(start: Vector3, end: Vector3, cfg: FPAPFConfig) -> bool:
    dz = abs(end[2] - start[2])
    return dz <= cfg.max_vertical_step + 1e-9


def is_path_safe(path: Sequence[Vector3], obstacles: Sequence[Obstacle], cfg: FPAPFConfig) -> bool:
    return all(
        is_motion_feasible(start, end, cfg) and is_segment_safe(start, end, obstacles, cfg)
        for start, end in zip(path[:-1], path[1:])
    )


def limit_vertical_step(current: Vector3, candidate: Vector3, cfg: FPAPFConfig) -> Vector3:
    max_step = max(cfg.max_vertical_step, 1e-6)
    dz = candidate[2] - current[2]
    if abs(dz) <= max_step:
        return candidate
    limited_z = current[2] + math.copysign(max_step, dz)
    return (candidate[0], candidate[1], limited_z)


def lateral_unit(direction: Vector3) -> Vector3:
    lateral = (-direction[1], direction[0], 0.0)
    if norm(lateral) < 1e-9:
        lateral = (1.0, 0.0, 0.0)
    return normalize(lateral)


def choose_safer_candidate(
    previous: Vector3 | None,
    current: Vector3,
    proposed: Vector3,
    goal: Vector3,
    obstacles: Sequence[Obstacle],
    cfg: FPAPFConfig,
) -> Vector3:
    proposed = limit_vertical_step(current, proposed, cfg)
    if is_motion_feasible(current, proposed, cfg) and is_segment_safe(current, proposed, obstacles, cfg):
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
            if not is_motion_feasible(current, candidate, cfg) or not is_segment_safe(current, candidate, obstacles, cfg):
                continue
            clearance_bonus = segment_min_clearance(current, candidate, obstacles, cfg)
            score = fitness(candidate, goal, obstacles, cfg, previous, current) - 0.15 * clearance_bonus
            if score < best_score:
                best_score = score
                best_candidate = candidate

    if best_candidate is not None:
        return best_candidate

    safe_progress = current
    for attempt in range(1, cfg.segment_samples + 1):
        fraction = attempt / cfg.segment_samples
        candidate = limit_vertical_step(current, lerp(current, proposed, fraction), cfg)
        if is_motion_feasible(current, candidate, cfg) and is_segment_safe(current, candidate, obstacles, cfg):
            safe_progress = candidate
        else:
            break
    return safe_progress


def initialize_particles(
    x_apf: Vector3,
    goal: Vector3,
    obstacles: Sequence[Obstacle],
    cfg: FPAPFConfig,
    previous: Vector3 | None = None,
    current: Vector3 | None = None,
) -> List[Particle]:
    particles: List[Particle] = []
    for _ in range(cfg.swarm_size):
        position = add(x_apf, random_gaussian_vector(cfg.init_sigma))
        velocity = random_gaussian_vector(cfg.velocity_sigma)
        fit = fitness(position, goal, obstacles, cfg, previous, current)
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
    obstacles: Sequence[Obstacle],
    cfg: FPAPFConfig,
    previous: Vector3 | None = None,
    current: Vector3 | None = None,
) -> Vector3:
    particles = initialize_particles(x_apf, goal, obstacles, cfg, previous, current)
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

            current_fitness = fitness(particle.position, goal, obstacles, cfg, previous, current)
            if current_fitness < particle.best_fitness:
                particle.best_position = particle.position
                particle.best_fitness = current_fitness

            if particle.best_fitness < g_best_fitness:
                g_best = particle.best_position
                g_best_fitness = particle.best_fitness

    return g_best


def smooth_path_once(path: Sequence[Vector3]) -> List[Vector3]:
    if len(path) < 3:
        return list(path)

    smoothed = [path[0]]
    for start, end in zip(path[:-1], path[1:]):
        smoothed.append(lerp(start, end, 0.25))
        smoothed.append(lerp(start, end, 0.75))
    smoothed.append(path[-1])
    return smoothed


def smooth_path_safely(
    path: Sequence[Vector3],
    obstacles: Sequence[Obstacle],
    cfg: FPAPFConfig,
) -> List[Vector3]:
    best_path = list(path)
    for _ in range(max(cfg.smoothing_passes, 0)):
        candidate = smooth_path_once(best_path)
        if is_path_safe(candidate, obstacles, cfg):
            best_path = candidate
        else:
            break
    return best_path


def fp_apf_plan(
    start: Vector3,
    goal: Vector3,
    obstacles: Sequence[Obstacle],
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

        active_goal = climb_guidance_goal(current, goal, cfg)
        f_att = attractive_force(current, active_goal, cfg.k_att)
        f_rep = total_repulsive_force(current, obstacles, cfg)
        f_total = add(f_att, f_rep)

        direction = normalize(f_total)
        if norm(direction) < 1e-12:
            direction = normalize(sub(goal, current))

        previous = path[-2] if len(path) >= 2 else None
        x_apf = add(current, mul(direction, cfg.step_scale))
        g_best = pso_refine(x_apf, active_goal, obstacles, cfg, previous, current)
        next_point = add(mul(x_apf, cfg.alpha), mul(g_best, 1.0 - cfg.alpha))
        next_point = limit_vertical_step(current, next_point, cfg)
        next_point = choose_safer_candidate(previous, current, next_point, active_goal, obstacles, cfg)

        if norm(sub(next_point, current)) < 1e-6:
            escape_direction = normalize(sub(goal, current))
            escape = limit_vertical_step(current, add(current, mul(escape_direction, max(cfg.step_scale, 1.0))), cfg)
            if is_motion_feasible(current, escape, cfg) and is_segment_safe(current, escape, obstacles, cfg):
                next_point = escape
            else:
                break

        current = next_point
        path.append(current)

        if (
            norm(sub(current, goal)) <= cfg.goal_threshold
            and is_motion_feasible(current, goal, cfg)
            and is_segment_safe(current, goal, obstacles, cfg)
        ):
            path.append(goal)
            break

    return smooth_path_safely(path, obstacles, cfg)
