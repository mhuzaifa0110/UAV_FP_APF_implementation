from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


Vector3 = Tuple[float, float, float]


@dataclass
class CylinderObstacle:
    center_xy: Tuple[float, float]
    radius: float
    height: float

    def clearance(self, point: Vector3) -> float:
        px, py, pz = point
        cx, cy = self.center_xy
        radial = math.hypot(px - cx, py - cy) - self.radius

        if 0.0 <= pz <= self.height:
            return radial

        if pz < 0.0:
            vertical = -pz
        else:
            vertical = pz - self.height

        if radial <= 0.0:
            return vertical

        return math.hypot(radial, vertical)

    def direction_to_point(self, point: Vector3) -> Vector3:
        px, py, pz = point
        cx, cy = self.center_xy

        closest_z = min(max(pz, 0.0), self.height)
        dx = px - cx
        dy = py - cy
        radial_norm = math.hypot(dx, dy)

        if radial_norm > 1e-9:
            closest_x = cx + dx / radial_norm * self.radius
            closest_y = cy + dy / radial_norm * self.radius
        else:
            closest_x = cx + self.radius
            closest_y = cy

        nx = px - closest_x
        ny = py - closest_y
        nz = pz - closest_z
        return normalize((nx, ny, nz))


@dataclass
class FPAPFConfig:
    k_att: float = 0.4
    k_rep: float = 10.0
    rho_0: float = 10.0
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
    collision_margin: float = 0.5


@dataclass
class Particle:
    position: Vector3
    velocity: Vector3
    best_position: Vector3
    best_fitness: float


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a: Vector3, scalar: float) -> Vector3:
    return (a[0] * scalar, a[1] * scalar, a[2] * scalar)


def norm(a: Vector3) -> float:
    return math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)


def normalize(a: Vector3) -> Vector3:
    mag = norm(a)
    if mag < 1e-12:
        return (0.0, 0.0, 0.0)
    return (a[0] / mag, a[1] / mag, a[2] / mag)


def random_gaussian_vector(sigma: float) -> Vector3:
    return (
        random.gauss(0.0, sigma),
        random.gauss(0.0, sigma),
        random.gauss(0.0, sigma),
    )


def attractive_potential(q: Vector3, q_goal: Vector3, k_att: float) -> float:
    return 0.5 * k_att * norm(sub(q, q_goal)) ** 2


def attractive_force(q: Vector3, q_goal: Vector3, k_att: float) -> Vector3:
    return mul(sub(q_goal, q), k_att)


def classical_repulsive_potential(rho_q: float, rho_0: float, k_rep: float) -> float:
    if rho_q <= 0.0:
        return float("inf")
    if rho_q <= rho_0:
        return 0.5 * k_rep * ((1.0 / rho_q) - (1.0 / rho_0)) ** 2
    return 0.0


def gaussian_repulsive_force(
    point: Vector3,
    obstacle: CylinderObstacle,
    k_rep: float,
    sigma: float,
    rho_d: float,
) -> Vector3:
    rho_q = max(obstacle.clearance(point), 1e-6)
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
                point,
                obstacle,
                cfg.k_rep,
                cfg.gaussian_sigma,
                cfg.rho_d,
            ),
        )
    return force


def obstacle_penalty(
    point: Vector3,
    obstacles: Sequence[CylinderObstacle],
    gain: float,
    margin: float,
) -> float:
    penalty = 0.0
    for obstacle in obstacles:
        clearance = obstacle.clearance(point)
        if clearance <= 0.0:
            penalty += gain * 1_000.0
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
        point,
        obstacles,
        cfg.obstacle_penalty_gain,
        cfg.collision_margin,
    )


def initialize_particles(x_apf: Vector3, goal: Vector3, obstacles: Sequence[CylinderObstacle], cfg: FPAPFConfig) -> List[Particle]:
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


def argmin_particle(particles: Sequence[Particle]) -> Particle:
    return min(particles, key=lambda p: p.best_fitness)


def pso_refine(
    x_apf: Vector3,
    goal: Vector3,
    obstacles: Sequence[CylinderObstacle],
    cfg: FPAPFConfig,
) -> Vector3:
    particles = initialize_particles(x_apf, goal, obstacles, cfg)
    g_best = argmin_particle(particles).best_position
    g_best_fitness = fitness(g_best, goal, obstacles, cfg)

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
) -> List[Vector3]:
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
        current = next_point
        path.append(current)

        if norm(sub(current, goal)) <= cfg.goal_threshold:
            path.append(goal)
            break

    return path


def path_length(path: Sequence[Vector3]) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        total += norm(sub(b, a))
    return total


def clearance_stats(path: Sequence[Vector3], obstacles: Sequence[CylinderObstacle]) -> Tuple[float, float]:
    if not path:
        return 0.0, 0.0

    clearances = []
    for point in path:
        if obstacles:
            clearances.append(min(obstacle.clearance(point) for obstacle in obstacles))
        else:
            clearances.append(float("inf"))

    if any(math.isinf(x) for x in clearances):
        return float("inf"), float("inf")

    avg_clearance = sum(clearances) / len(clearances)
    min_clearance = min(clearances)
    return avg_clearance, min_clearance


def weighted_score(
    length_value: float,
    avg_clearance: float,
    min_clearance: float,
    length_max: float,
    avg_clearance_max: float,
    min_clearance_max: float,
    t1: float = 0.7,
    t2: float = 0.2,
    t3: float = 0.1,
) -> float:
    if length_max <= 0.0 or avg_clearance_max <= 0.0 or min_clearance_max <= 0.0:
        raise ValueError("Normalization maxima must be positive.")

    return (
        t1 * (1.0 - length_value / length_max)
        + t2 * (avg_clearance / avg_clearance_max)
        + t3 * (min_clearance / min_clearance_max)
    )


def make_demo_environment() -> Tuple[Vector3, Vector3, List[CylinderObstacle]]:
    start = (10.0, 10.0, 0.0)
    goal = (100.0, 150.0, 50.0)
    obstacles = [
        CylinderObstacle(center_xy=(40.0, 45.0), radius=8.0, height=35.0),
        CylinderObstacle(center_xy=(60.0, 80.0), radius=10.0, height=45.0),
        CylinderObstacle(center_xy=(78.0, 110.0), radius=12.0, height=55.0),
        CylinderObstacle(center_xy=(95.0, 95.0), radius=8.0, height=30.0),
    ]
    return start, goal, obstacles


def to_numpy_points(path: Sequence[Vector3]) -> np.ndarray:
    if not path:
        return np.empty((0, 3))
    return np.array(path, dtype=float)


def compute_plot_bounds(
    start: Vector3,
    goal: Vector3,
    obstacles: Sequence[CylinderObstacle],
    path: Sequence[Vector3],
    padding: float = 12.0,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    xs = [start[0], goal[0], *(p[0] for p in path)]
    ys = [start[1], goal[1], *(p[1] for p in path)]
    zs = [start[2], goal[2], *(p[2] for p in path)]

    for obstacle in obstacles:
        cx, cy = obstacle.center_xy
        xs.extend([cx - obstacle.radius, cx + obstacle.radius])
        ys.extend([cy - obstacle.radius, cy + obstacle.radius])
        zs.extend([0.0, obstacle.height])

    return (
        (min(xs) - padding, max(xs) + padding),
        (min(ys) - padding, max(ys) + padding),
        (min(zs) - padding * 0.3, max(zs) + padding * 0.3),
    )


def plot_cylinder(ax: plt.Axes, obstacle: CylinderObstacle, color: str = "#7f8c8d", alpha: float = 0.28) -> None:
    cx, cy = obstacle.center_xy
    theta = np.linspace(0.0, 2.0 * math.pi, 40)
    z = np.linspace(0.0, obstacle.height, 24)
    theta_grid, z_grid = np.meshgrid(theta, z)

    x_grid = cx + obstacle.radius * np.cos(theta_grid)
    y_grid = cy + obstacle.radius * np.sin(theta_grid)

    ax.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        color=color,
        alpha=alpha,
        linewidth=0.0,
        shade=True,
    )


def setup_3d_axes(
    start: Vector3,
    goal: Vector3,
    obstacles: Sequence[CylinderObstacle],
    path: Sequence[Vector3],
    title: str,
) -> Tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    for obstacle in obstacles:
        plot_cylinder(ax, obstacle)

    x_lim, y_lim, z_lim = compute_plot_bounds(start, goal, obstacles, path)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.view_init(elev=28, azim=-58)
    ax.grid(True, alpha=0.3)
    return fig, ax


def save_static_plot(
    start: Vector3,
    goal: Vector3,
    obstacles: Sequence[CylinderObstacle],
    path: Sequence[Vector3],
    output_path: Path,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    points = to_numpy_points(path)

    fig, ax = setup_3d_axes(start, goal, obstacles, path, "FP-APF 3D Path Planning")
    if len(points) > 0:
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#2980b9", linewidth=2.6, label="FP-APF Path")
    ax.scatter(*start, color="#27ae60", s=90, label="Start")
    ax.scatter(*goal, color="#c0392b", s=90, label="Goal")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_animation(
    start: Vector3,
    goal: Vector3,
    obstacles: Sequence[CylinderObstacle],
    path: Sequence[Vector3],
    output_path: Path,
    fps: int = 2,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    points = to_numpy_points(path)
    if len(points) == 0:
        raise ValueError("Path is empty. Cannot animate an empty path.")

    fig, ax = setup_3d_axes(start, goal, obstacles, path, "FP-APF UAV Path Animation")
    ax.scatter(*start, color="#27ae60", s=90, label="Start")
    ax.scatter(*goal, color="#c0392b", s=90, label="Goal")
    trail_line, = ax.plot([], [], [], color="#2980b9", linewidth=2.6, label="UAV Trail")
    uav_marker, = ax.plot([], [], [], marker="o", markersize=8, color="#f39c12", linestyle="", label="UAV")
    ax.legend(loc="upper left")

    def init() -> Tuple[plt.Artist, plt.Artist]:
        trail_line.set_data([], [])
        trail_line.set_3d_properties([])
        uav_marker.set_data([], [])
        uav_marker.set_3d_properties([])
        return trail_line, uav_marker

    def update(frame: int) -> Tuple[plt.Artist, plt.Artist]:
        segment = points[: frame + 1]
        trail_line.set_data(segment[:, 0], segment[:, 1])
        trail_line.set_3d_properties(segment[:, 2])
        uav_marker.set_data([segment[-1, 0]], [segment[-1, 1]])
        uav_marker.set_3d_properties([segment[-1, 2]])
        return trail_line, uav_marker

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(points),
        interval=max(100, int(1000 / max(fps, 1))),
        blit=False,
        repeat=False,
    )
    ani.save(output_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return output_path


def run_demo(mode: str) -> None:
    random.seed(7)

    start, goal, obstacles = make_demo_environment()
    cfg = FPAPFConfig()

    path = fp_apf_plan(start, goal, obstacles, cfg)
    length_value = path_length(path)
    avg_clearance, min_clearance = clearance_stats(path, obstacles)

    print("FP-APF reproduction scaffold")
    print(f"Path points: {len(path)}")
    print(f"Path length: {length_value:.3f}")
    print(f"Average clearance: {avg_clearance:.3f}")
    print(f"Minimum clearance: {min_clearance:.3f}")
    print("Last point:", path[-1] if path else None)

    static_path = OUTPUT_DIR / "fp_apf_static_plot.png"
    animation_path = OUTPUT_DIR / "fp_apf_animation.gif"

    if mode in {"plot", "both"}:
        saved_plot = save_static_plot(start, goal, obstacles, path, static_path)
        print("Saved static plot:", saved_plot)

    if mode in {"animate", "both"}:
        saved_animation = save_animation(start, goal, obstacles, path, animation_path)
        print("Saved animation:", saved_animation)


def main() -> None:
    parser = argparse.ArgumentParser(description="FP-APF path planning reproduction with visualization.")
    parser.add_argument(
        "--mode",
        choices=("plot", "animate", "both"),
        default="both",
        help="Visualization mode to generate.",
    )
    args = parser.parse_args()
    run_demo(args.mode)


if __name__ == "__main__":
    main()

