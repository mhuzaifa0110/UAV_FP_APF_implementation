from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .geometry import Obstacle, Vector3, norm, sub


def path_length(path: Sequence[Vector3]) -> float:
    if len(path) < 2:
        return 0.0
    return sum(norm(sub(b, a)) for a, b in zip(path[:-1], path[1:]))


def clearance_stats(
    path: Sequence[Vector3],
    obstacles: Sequence[Obstacle],
    drone_radius: float = 0.0,
) -> tuple[float, float]:
    if not path:
        return 0.0, 0.0
    if not obstacles:
        return float("inf"), float("inf")

    clearances = [min(obstacle.clearance(point) - drone_radius for obstacle in obstacles) for point in path]
    avg_clearance = sum(clearances) / len(clearances)
    min_clearance = min(clearances)
    return avg_clearance, min_clearance


@dataclass
class PathMetrics:
    point_count: int
    path_length: float
    average_clearance: float
    minimum_clearance: float


def summarize_path(
    path: Sequence[Vector3],
    obstacles: Sequence[Obstacle],
    drone_radius: float = 0.0,
) -> PathMetrics:
    avg_clearance, min_clearance = clearance_stats(path, obstacles, drone_radius)
    return PathMetrics(
        point_count=len(path),
        path_length=path_length(path),
        average_clearance=avg_clearance,
        minimum_clearance=min_clearance,
    )
