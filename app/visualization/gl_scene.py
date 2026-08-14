from __future__ import annotations

from typing import Sequence

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui

from ..simulation.geometry import CylinderObstacle, EnvironmentSpec, Obstacle, PolygonPrismObstacle, Vector3


class UAVScene(gl.GLViewWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setBackgroundColor((244, 239, 229))
        self.opts["distance"] = 230
        self.opts["elevation"] = 24
        self.opts["azimuth"] = 35

        self._obstacle_items: list[gl.GLMeshItem] = []
        self._obstacle_colors: list[tuple[float, float, float, float]] = []
        self._path_item: gl.GLLinePlotItem | None = None
        self._trail_item: gl.GLLinePlotItem | None = None
        self._start_item: gl.GLScatterPlotItem | None = None
        self._goal_item: gl.GLScatterPlotItem | None = None
        self._preview_item: gl.GLMeshItem | None = None
        self._axis_items: list[object] = []
        self._current_path: np.ndarray = np.empty((0, 3))

        grid = gl.GLGridItem()
        grid.setSize(180, 180)
        grid.setSpacing(10, 10)
        grid.translate(75, 75, 0)
        self.addItem(grid)

        sphere = gl.MeshData.sphere(rows=12, cols=20, radius=2.2)
        self._uav_item = gl.GLMeshItem(
            meshdata=sphere,
            color=(0.93, 0.54, 0.15, 1.0),
            smooth=True,
            shader="shaded",
            drawFaces=True,
        )
        self.addItem(self._uav_item)

    def set_exact_top_view(self) -> None:
        self.opts["elevation"] = 90
        self.opts["azimuth"] = -90
        self.opts["distance"] = 230
        self.update()

    def clear_environment(self) -> None:
        for item in self._obstacle_items:
            self.removeItem(item)
        self._obstacle_items.clear()
        self._obstacle_colors.clear()

        for item in (self._path_item, self._trail_item, self._start_item, self._goal_item):
            if item is not None:
                self.removeItem(item)
        if self._preview_item is not None:
            self.removeItem(self._preview_item)
        for item in self._axis_items:
            self.removeItem(item)
        self._axis_items.clear()

        self._path_item = None
        self._trail_item = None
        self._start_item = None
        self._goal_item = None
        self._preview_item = None
        self._current_path = np.empty((0, 3))
        self.update_uav_position((0.0, 0.0, 0.0))

    def load_environment(self, environment: EnvironmentSpec, show_axes: bool = False) -> None:
        self.clear_environment()
        self._draw_obstacles(environment.obstacles)
        self._draw_markers(environment.start, environment.goal)
        if show_axes:
            self._draw_axes(environment)
        self.opts["center"] = QtGui.QVector3D(  # type: ignore[attr-defined]
            (environment.bounds[0][0] + environment.bounds[0][1]) / 2.0,
            (environment.bounds[1][0] + environment.bounds[1][1]) / 2.0,
            environment.bounds[2][1] / 3.0,
        )
        self.update()

    def _draw_axes(self, environment: EnvironmentSpec) -> None:
        x_max = environment.bounds[0][1]
        y_max = environment.bounds[1][1]
        z_max = environment.bounds[2][1]

        axes = (
            ([(0.0, 0.0, 0.0), (x_max, 0.0, 0.0)], (1.0, 0.0, 0.0, 1.0)),
            ([(0.0, 0.0, 0.0), (0.0, y_max, 0.0)], (0.10, 0.56, 0.25, 1.0)),
            ([(0.0, 0.0, 0.0), (0.0, 0.0, z_max)], (0.12, 0.32, 0.82, 1.0)),
        )
        for points, color in axes:
            axis = gl.GLLinePlotItem(
                pos=np.array(points, dtype=float),
                color=color,
                width=3.0,
                antialias=True,
                mode="line_strip",
            )
            self.addItem(axis)
            self._axis_items.append(axis)

        labels = (
            ("X", (x_max + 5.0, 0.0, 0.0), (1.0, 0.0, 0.0, 1.0)),
            ("Y", (0.0, y_max + 5.0, 0.0), (0.10, 0.56, 0.25, 1.0)),
            ("Z", (0.0, 0.0, z_max + 5.0), (0.12, 0.32, 0.82, 1.0)),
        )
        for text, position, color in labels:
            label = gl.GLTextItem(pos=position, text=text, color=color)
            self.addItem(label)
            self._axis_items.append(label)

    def _draw_obstacles(self, obstacles: Sequence[Obstacle]) -> None:
        for obstacle in obstacles:
            item = self._build_obstacle_item(obstacle)
            self.addItem(item)
            self._obstacle_items.append(item)
            self._obstacle_colors.append(obstacle.color_rgba)

    def _build_obstacle_item(
        self,
        obstacle: Obstacle,
        color: tuple[float, float, float, float] | None = None,
        draw_edges: bool = False,
    ) -> gl.GLMeshItem:
        if isinstance(obstacle, PolygonPrismObstacle):
            mesh = self._polygon_prism_mesh(obstacle)
        else:
            mesh = gl.MeshData.cylinder(
                rows=2,
                cols=32,
                radius=[obstacle.radius, obstacle.radius],
                length=obstacle.height,
            )
        item = gl.GLMeshItem(
            meshdata=mesh,
            color=color or obstacle.color_rgba,
            smooth=isinstance(obstacle, CylinderObstacle),
            shader="shaded",
            drawEdges=draw_edges,
            drawFaces=True,
        )
        if isinstance(obstacle, CylinderObstacle):
            cx, cy = obstacle.center_xy
            item.translate(cx, cy, obstacle.base_z)
        return item

    def _polygon_prism_mesh(self, obstacle: PolygonPrismObstacle) -> gl.MeshData:
        footprint = obstacle.world_vertices_xy
        count = len(footprint)
        base_z = obstacle.base_z
        top_z = obstacle.base_z + obstacle.height
        vertices = [(x, y, base_z) for x, y in footprint] + [(x, y, top_z) for x, y in footprint]

        faces: list[tuple[int, int, int]] = []
        for index in range(1, count - 1):
            faces.append((0, index + 1, index))
            faces.append((count, count + index, count + index + 1))
        for index in range(count):
            next_index = (index + 1) % count
            faces.append((index, next_index, count + next_index))
            faces.append((index, count + next_index, count + index))

        return gl.MeshData(
            vertexes=np.array(vertices, dtype=float),
            faces=np.array(faces, dtype=int),
        )

    def _draw_markers(self, start: Vector3, goal: Vector3) -> None:
        self._start_item = gl.GLScatterPlotItem(pos=np.array([start], dtype=float), color=(0.12, 0.36, 0.78, 1.0), size=10)
        self._goal_item = gl.GLScatterPlotItem(pos=np.array([goal], dtype=float), color=(0.10, 0.68, 0.30, 1.0), size=11)
        self.addItem(self._start_item)
        self.addItem(self._goal_item)
        self.update_uav_position(start)

    def highlight_obstacle(self, selected_index: int | None) -> None:
        self._apply_obstacle_highlight(selected_index)
        self.update()

    def _apply_obstacle_highlight(self, selected_index: int | None, dim_selected: bool = False) -> None:
        for index, item in enumerate(self._obstacle_items):
            if index == selected_index:
                if dim_selected and index < len(self._obstacle_colors):
                    base_color = self._obstacle_colors[index]
                    item.setColor((base_color[0], base_color[1], base_color[2], 0.14))
                else:
                    item.setColor((1.0, 0.18, 0.08, 0.72))
            elif index < len(self._obstacle_colors):
                item.setColor(self._obstacle_colors[index])

    def set_preview_obstacle(self, obstacle: Obstacle | None, selected_index: int | None = None) -> None:
        if self._preview_item is not None:
            self.removeItem(self._preview_item)
            self._preview_item = None
        if obstacle is None:
            self._apply_obstacle_highlight(selected_index)
            self.update()
            return
        self._apply_obstacle_highlight(selected_index, dim_selected=selected_index is not None)
        preview_color = (1.0, 0.18, 0.08, 0.46) if selected_index is not None else (1.0, 0.84, 0.18, 0.28)
        self._preview_item = self._build_obstacle_item(
            obstacle,
            color=preview_color,
            draw_edges=True,
        )
        self.addItem(self._preview_item)
        self.update()

    def set_path(self, path: Sequence[Vector3], accent_rgb: tuple[float, float, float]) -> None:
        path_np = np.array(path, dtype=float)
        self._current_path = path_np

        if self._path_item is not None:
            self.removeItem(self._path_item)
        if self._trail_item is not None:
            self.removeItem(self._trail_item)

        if len(path_np) == 0:
            self._path_item = None
            self._trail_item = None
            return

        self._path_item = gl.GLLinePlotItem(
            pos=path_np,
            color=(accent_rgb[0], accent_rgb[1], accent_rgb[2], 0.45),
            width=2.0,
            antialias=True,
            mode="line_strip",
        )
        self._trail_item = gl.GLLinePlotItem(
            pos=path_np[:1],
            color=(0.95, 0.56, 0.13, 0.95),
            width=4.0,
            antialias=True,
            mode="line_strip",
        )
        self.addItem(self._path_item)
        self.addItem(self._trail_item)
        self.update_uav_position(tuple(path_np[0]))

    def update_progress(self, path_index: int) -> None:
        if len(self._current_path) == 0:
            return
        clamped = max(0, min(path_index, len(self._current_path) - 1))
        visible = self._current_path[: clamped + 1]
        if self._trail_item is not None:
            self._trail_item.setData(pos=visible)
        self.update_uav_position(tuple(visible[-1]))

    def update_uav_position(self, position: Vector3) -> None:
        self._uav_item.resetTransform()
        self._uav_item.translate(position[0], position[1], position[2])
