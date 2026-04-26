from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

Vector3 = Tuple[float, float, float]


def add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a: Vector3, scalar: float) -> Vector3:
    return (a[0] * scalar, a[1] * scalar, a[2] * scalar)


def norm(a: Vector3) -> float:
    return math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)


def normalize(a: Vector3) -> Vector3:
    magnitude = norm(a)
    if magnitude < 1e-12:
        return (0.0, 0.0, 0.0)
    return (a[0] / magnitude, a[1] / magnitude, a[2] / magnitude)


@dataclass(frozen=True)
class CylinderObstacle:
    center_xy: Tuple[float, float]
    radius: float
    height: float
    color_rgba: Tuple[float, float, float, float] = (0.42, 0.58, 0.63, 0.35)

    def clearance(self, point: Vector3) -> float:
        px, py, pz = point
        cx, cy = self.center_xy
        radial = math.hypot(px - cx, py - cy) - self.radius

        if 0.0 <= pz <= self.height:
            return radial

        vertical = -pz if pz < 0.0 else pz - self.height
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

        return normalize((px - closest_x, py - closest_y, pz - closest_z))


@dataclass(frozen=True)
class EnvironmentSpec:
    key: str
    name: str
    description: str
    start: Vector3
    goal: Vector3
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    obstacles: Tuple[CylinderObstacle, ...] = field(default_factory=tuple)
    accent_rgb: Tuple[float, float, float] = (0.07, 0.55, 0.62)
