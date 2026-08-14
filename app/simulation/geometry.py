from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple, TypeAlias

Vector3 = Tuple[float, float, float]
Vector2 = Tuple[float, float]
ColorRGBA = Tuple[float, float, float, float]


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
    center_xy: Vector2
    radius: float
    height: float
    color_rgba: ColorRGBA = (0.42, 0.58, 0.63, 0.35)
    base_z: float = 0.0
    safety_margin: float = 0.0

    def clearance(self, point: Vector3) -> float:
        px, py, pz = point
        cx, cy = self.center_xy
        radial = math.hypot(px - cx, py - cy) - self.radius
        top_z = self.base_z + self.height

        if self.base_z <= pz <= top_z:
            return radial - self.safety_margin

        vertical = self.base_z - pz if pz < self.base_z else pz - top_z
        if radial <= 0.0:
            return vertical - self.safety_margin
        return math.hypot(radial, vertical) - self.safety_margin

    def direction_to_point(self, point: Vector3) -> Vector3:
        px, py, pz = point
        cx, cy = self.center_xy
        closest_z = min(max(pz, self.base_z), self.base_z + self.height)

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


def _point_segment_closest(point: Vector2, start: Vector2, end: Vector2) -> Vector2:
    vx = end[0] - start[0]
    vy = end[1] - start[1]
    length_sq = vx * vx + vy * vy
    if length_sq < 1e-12:
        return start
    t = ((point[0] - start[0]) * vx + (point[1] - start[1]) * vy) / length_sq
    t = min(max(t, 0.0), 1.0)
    return (start[0] + t * vx, start[1] + t * vy)


def _point_in_polygon(point: Vector2, vertices: Tuple[Vector2, ...]) -> bool:
    inside = False
    px, py = point
    previous = vertices[-1]
    for current in vertices:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > py) != (y2 > py):
            x_intersection = (x2 - x1) * (py - y1) / (y2 - y1) + x1
            if px < x_intersection:
                inside = not inside
        previous = current
    return inside


def _orientation(a: Vector2, b: Vector2, c: Vector2) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a: Vector2, b: Vector2, c: Vector2, d: Vector2) -> bool:
    return (
        _orientation(a, b, c) * _orientation(a, b, d) < -1e-12
        and _orientation(c, d, a) * _orientation(c, d, b) < -1e-12
    )


@dataclass(frozen=True)
class PolygonPrismObstacle:
    center_xy: Vector2
    vertices_xy: Tuple[Vector2, ...]
    height: float
    color_rgba: ColorRGBA = (0.55, 0.66, 0.46, 0.38)
    base_z: float = 0.0
    rotation_deg: float = 0.0
    safety_margin: float = 0.0

    def __post_init__(self) -> None:
        if len(self.vertices_xy) < 3:
            raise ValueError("A polygon obstacle requires at least three vertices.")
        if self.height <= 0.0:
            raise ValueError("A polygon obstacle height must be positive.")
        area_twice = sum(
            start[0] * end[1] - end[0] * start[1]
            for start, end in zip(self.vertices_xy, self.vertices_xy[1:] + self.vertices_xy[:1])
        )
        if abs(area_twice) < 1e-9:
            raise ValueError("Polygon vertices must enclose a non-zero area.")
        turn_signs = []
        for index in range(len(self.vertices_xy)):
            turn = _orientation(
                self.vertices_xy[index - 1],
                self.vertices_xy[index],
                self.vertices_xy[(index + 1) % len(self.vertices_xy)],
            )
            if abs(turn) > 1e-9:
                turn_signs.append(turn > 0.0)
        if turn_signs and not all(sign == turn_signs[0] for sign in turn_signs):
            raise ValueError("Custom polygon vertices must form a convex polygon.")
        edge_count = len(self.vertices_xy)
        for first in range(edge_count):
            a = self.vertices_xy[first]
            b = self.vertices_xy[(first + 1) % edge_count]
            for second in range(first + 1, edge_count):
                if second in (first, (first + 1) % edge_count) or (second + 1) % edge_count == first:
                    continue
                c = self.vertices_xy[second]
                d = self.vertices_xy[(second + 1) % edge_count]
                if _segments_intersect(a, b, c, d):
                    raise ValueError("Polygon edges must not self-intersect.")

    @property
    def world_vertices_xy(self) -> Tuple[Vector2, ...]:
        angle = math.radians(self.rotation_deg)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        cx, cy = self.center_xy
        return tuple(
            (
                cx + vertex[0] * cos_angle - vertex[1] * sin_angle,
                cy + vertex[0] * sin_angle + vertex[1] * cos_angle,
            )
            for vertex in self.vertices_xy
        )

    def _horizontal_surface(self, point_xy: Vector2) -> tuple[float, Vector2, bool]:
        vertices = self.world_vertices_xy
        closest = vertices[0]
        closest_distance = math.inf
        for start, end in zip(vertices, vertices[1:] + vertices[:1]):
            candidate = _point_segment_closest(point_xy, start, end)
            distance = math.hypot(point_xy[0] - candidate[0], point_xy[1] - candidate[1])
            if distance < closest_distance:
                closest_distance = distance
                closest = candidate
        return closest_distance, closest, _point_in_polygon(point_xy, vertices)

    def clearance(self, point: Vector3) -> float:
        px, py, pz = point
        horizontal_distance, _, inside_xy = self._horizontal_surface((px, py))
        top_z = self.base_z + self.height
        inside_z = self.base_z <= pz <= top_z

        if inside_xy and inside_z:
            boundary_distance = min(horizontal_distance, pz - self.base_z, top_z - pz)
            return -boundary_distance - self.safety_margin

        vertical_distance = 0.0 if inside_z else min(abs(pz - self.base_z), abs(pz - top_z))
        planar_distance = 0.0 if inside_xy else horizontal_distance
        return math.hypot(planar_distance, vertical_distance) - self.safety_margin

    def direction_to_point(self, point: Vector3) -> Vector3:
        px, py, pz = point
        horizontal_distance, closest_xy, inside_xy = self._horizontal_surface((px, py))
        top_z = self.base_z + self.height
        inside_z = self.base_z <= pz <= top_z

        if inside_xy and inside_z:
            distances = (
                (horizontal_distance, (closest_xy[0] - px, closest_xy[1] - py, 0.0)),
                (pz - self.base_z, (0.0, 0.0, -1.0)),
                (top_z - pz, (0.0, 0.0, 1.0)),
            )
            return normalize(min(distances, key=lambda item: item[0])[1])

        closest_z = min(max(pz, self.base_z), top_z)
        closest_x, closest_y = (px, py) if inside_xy else closest_xy
        return normalize((px - closest_x, py - closest_y, pz - closest_z))

    @classmethod
    def rectangle(
        cls,
        center_xy: Vector2,
        width: float,
        length: float,
        height: float,
        **kwargs: object,
    ) -> "PolygonPrismObstacle":
        half_width = width / 2.0
        half_length = length / 2.0
        vertices = (
            (-half_width, -half_length),
            (half_width, -half_length),
            (half_width, half_length),
            (-half_width, half_length),
        )
        return cls(center_xy=center_xy, vertices_xy=vertices, height=height, **kwargs)


Obstacle: TypeAlias = CylinderObstacle | PolygonPrismObstacle


@dataclass(frozen=True)
class EnvironmentSpec:
    key: str
    name: str
    description: str
    start: Vector3
    goal: Vector3
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    obstacles: Tuple[Obstacle, ...] = field(default_factory=tuple)
    accent_rgb: Tuple[float, float, float] = (0.07, 0.55, 0.62)
