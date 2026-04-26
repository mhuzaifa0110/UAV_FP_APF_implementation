from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Dict

from PySide6 import QtCore, QtWidgets

from ..simulation.environments import build_environment_library
from ..simulation.geometry import CylinderObstacle, EnvironmentSpec
from ..simulation.metrics import PathMetrics, summarize_path
from ..simulation.planner import FPAPFConfig, fp_apf_plan
from ..visualization.gl_scene import UAVScene
from .styles import APP_STYLESHEET

SAVED_ENVIRONMENTS_PATH = Path(__file__).resolve().parents[1] / "saved_custom_environments.json"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FP-APF UAV Implementation")
        self.setFixedSize(1280, 648)
        self.setStyleSheet(APP_STYLESHEET)

        self.environments: Dict[str, EnvironmentSpec] = build_environment_library()
        self.environments.update(self._load_saved_custom_environments())
        self.current_environment: EnvironmentSpec = next(iter(self.environments.values()))
        self.current_path: list[tuple[float, float, float]] = []
        self.current_metrics: PathMetrics | None = None
        self.play_index = 0

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._advance_animation)

        self.scene = UAVScene()
        self._build_ui()
        self._populate_environments()
        self.load_environment(self.current_environment.key)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(14)

        root_layout.addWidget(self._build_header())

        self.sidebar_scroll = QtWidgets.QScrollArea()
        self.sidebar_scroll.setObjectName("SidebarScroll")
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.sidebar_scroll.setFixedWidth(390)
        self.sidebar_scroll.setWidget(self._build_sidebar())

        content = QtWidgets.QHBoxLayout()
        content.setSpacing(14)
        content.addWidget(self.sidebar_scroll, 0)
        content.addWidget(self.scene, 1)
        root_layout.addLayout(content, 1)

        self.setCentralWidget(central)

    def _build_header(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame(objectName="HeaderCard")
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(18, 14, 18, 14)

        title_col = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("FP-APF UAV Implementation", objectName="TitleLabel")
        subtitle = QtWidgets.QLabel(
            "Done by Muhammad Huzaifa and Shahab Shinwari",
            objectName="SubtitleLabel",
        )
        subtitle.setWordWrap(True)
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        layout.addLayout(title_col, 1)

        return frame

    def _build_sidebar(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame(objectName="SidebarCard")
        frame.setMinimumWidth(300)
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        layout.addWidget(self._section_label("Environment"))
        self.environment_combo = QtWidgets.QComboBox()
        self.environment_combo.currentIndexChanged.connect(self._on_environment_changed)
        layout.addWidget(self.environment_combo)

        self.custom_editor = self._build_custom_editor()
        layout.addWidget(self.custom_editor)

        layout.addWidget(self._section_label("Planner Controls"))
        controls_row = QtWidgets.QHBoxLayout()
        controls_row.setSpacing(8)

        self.plan_button = QtWidgets.QPushButton("Plan")
        self.plan_button.setObjectName("CompactButton")
        self.plan_button.clicked.connect(self.plan_current_environment)
        controls_row.addWidget(self.plan_button)

        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.setObjectName("CompactButton")
        self.play_button.clicked.connect(self.toggle_animation)
        controls_row.addWidget(self.play_button)

        self.reset_button = QtWidgets.QPushButton("Reset", objectName="CompactSecondaryButton")
        self.reset_button.clicked.connect(self.reset_animation)
        controls_row.addWidget(self.reset_button)
        layout.addLayout(controls_row)

        self.next_env_button = QtWidgets.QPushButton("Next Environment", objectName="SecondaryButton")
        self.next_env_button.clicked.connect(self.load_next_environment)
        layout.addWidget(self.next_env_button)

        self.top_view_button = QtWidgets.QPushButton("Exact Top View", objectName="SecondaryButton")
        self.top_view_button.clicked.connect(self.scene.set_exact_top_view)
        layout.addWidget(self.top_view_button)

        layout.addWidget(self._section_label("Animation Speed"))
        speed_row = QtWidgets.QHBoxLayout()
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(12)
        self.speed_slider.setValue(5)
        self.speed_slider.valueChanged.connect(self._update_timer_speed)
        speed_row.addWidget(self.speed_slider, 1)
        self.speed_label = QtWidgets.QLabel("5 fps")
        speed_row.addWidget(self.speed_label)
        layout.addLayout(speed_row)

        layout.addWidget(self._section_label("Planner Parameters"))
        params_grid = QtWidgets.QGridLayout()
        params_grid.setHorizontalSpacing(10)
        params_grid.setVerticalSpacing(10)

        self.step_scale_spin = self._double_spin(0.1, 3.0, 0.5, 0.1)
        self.alpha_spin = self._double_spin(0.05, 0.95, 0.35, 0.05)
        self.swarm_spin = self._int_spin(10, 100, 40, 5)
        self.iter_spin = self._int_spin(5, 100, 35, 5)

        params_grid.addWidget(QtWidgets.QLabel("Step scale"), 0, 0)
        params_grid.addWidget(self.step_scale_spin, 0, 1)
        params_grid.addWidget(QtWidgets.QLabel("Fusion alpha"), 1, 0)
        params_grid.addWidget(self.alpha_spin, 1, 1)
        params_grid.addWidget(QtWidgets.QLabel("Swarm size"), 2, 0)
        params_grid.addWidget(self.swarm_spin, 2, 1)
        params_grid.addWidget(QtWidgets.QLabel("PSO iterations"), 3, 0)
        params_grid.addWidget(self.iter_spin, 3, 1)
        layout.addLayout(params_grid)

        stats_frame = QtWidgets.QFrame(objectName="StatsCard")
        stats_layout = QtWidgets.QVBoxLayout(stats_frame)
        stats_layout.setContentsMargins(12, 12, 12, 12)
        stats_layout.addWidget(self._section_label("Run Metrics"))

        self.points_value = QtWidgets.QLabel("-")
        self.length_value = QtWidgets.QLabel("-")
        self.avg_clearance_value = QtWidgets.QLabel("-")
        self.min_clearance_value = QtWidgets.QLabel("-")

        for label, value_widget in (
            ("Path points", self.points_value),
            ("Path length", self.length_value),
            ("Avg clearance", self.avg_clearance_value),
            ("Min clearance", self.min_clearance_value),
        ):
            row = QtWidgets.QHBoxLayout()
            key = QtWidgets.QLabel(label)
            key.setStyleSheet("font-weight:700;color:#244650;")
            row.addWidget(key)
            row.addStretch(1)
            row.addWidget(value_widget)
            stats_layout.addLayout(row)

        layout.addWidget(stats_frame)

        layout.addStretch(1)
        return frame

    def _build_custom_editor(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame(objectName="CustomEditor")
        frame.setVisible(False)
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        layout.addLayout(self._section_header_row("Start / Goal", "Scroll to change values"))
        points_grid = QtWidgets.QGridLayout()
        points_grid.setHorizontalSpacing(8)
        points_grid.setVerticalSpacing(8)

        self.start_x_spin = self._double_spin(0.0, 155.0, 10.0, 1.0)
        self.start_y_spin = self._double_spin(0.0, 155.0, 10.0, 1.0)
        self.start_z_spin = self._double_spin(0.0, 80.0, 0.0, 1.0)
        self.goal_x_spin = self._double_spin(0.0, 155.0, 140.0, 1.0)
        self.goal_y_spin = self._double_spin(0.0, 155.0, 140.0, 1.0)
        self.goal_z_spin = self._double_spin(0.0, 80.0, 48.0, 1.0)

        for column, label in enumerate(("X", "Y", "Z"), start=1):
            points_grid.addWidget(QtWidgets.QLabel(label), 0, column)
        points_grid.addWidget(QtWidgets.QLabel("Start"), 1, 0)
        points_grid.addWidget(self.start_x_spin, 1, 1)
        points_grid.addWidget(self.start_y_spin, 1, 2)
        points_grid.addWidget(self.start_z_spin, 1, 3)
        points_grid.addWidget(QtWidgets.QLabel("Goal"), 2, 0)
        points_grid.addWidget(self.goal_x_spin, 2, 1)
        points_grid.addWidget(self.goal_y_spin, 2, 2)
        points_grid.addWidget(self.goal_z_spin, 2, 3)
        layout.addLayout(points_grid)

        self.apply_points_button = QtWidgets.QPushButton("Apply Start / Goal")
        self.apply_points_button.clicked.connect(self.apply_custom_start_goal)
        layout.addWidget(self.apply_points_button)

        save_row = QtWidgets.QHBoxLayout()
        save_row.setSpacing(8)
        self.custom_name_edit = QtWidgets.QLineEdit()
        self.custom_name_edit.setPlaceholderText("Custom environment name")
        save_row.addWidget(self.custom_name_edit, 1)
        self.save_custom_button = QtWidgets.QPushButton("Save")
        self.save_custom_button.setObjectName("CompactButton")
        self.save_custom_button.clicked.connect(self.save_custom_environment)
        save_row.addWidget(self.save_custom_button, 0)
        layout.addLayout(save_row)

        layout.addLayout(self._section_header_row("Custom Obstacles", "Scroll to change values"))

        controls_row = QtWidgets.QHBoxLayout()
        controls_row.setSpacing(6)

        self.custom_x_spin = self._double_spin(0.0, 155.0, 60.0, 1.0)
        self.custom_y_spin = self._double_spin(0.0, 155.0, 60.0, 1.0)
        self.custom_radius_spin = self._double_spin(1.0, 30.0, 8.0, 0.5)
        self.custom_height_spin = self._double_spin(5.0, 80.0, 40.0, 1.0)
        for spin in (
            self.custom_x_spin,
            self.custom_y_spin,
            self.custom_radius_spin,
            self.custom_height_spin,
        ):
            spin.setFixedWidth(54)
            spin.valueChanged.connect(self._update_custom_preview)

        for label, spin in (
            ("X", self.custom_x_spin),
            ("Y", self.custom_y_spin),
            ("R", self.custom_radius_spin),
            ("H", self.custom_height_spin),
        ):
            controls_row.addWidget(QtWidgets.QLabel(label), 0)
            controls_row.addWidget(spin, 1)
        layout.addLayout(controls_row)

        buttons_row = QtWidgets.QHBoxLayout()
        buttons_row.setSpacing(6)
        self.add_obstacle_button = QtWidgets.QPushButton("Add")
        self.add_obstacle_button.setObjectName("CompactButton")
        self.add_obstacle_button.clicked.connect(self.add_custom_obstacle)
        buttons_row.addWidget(self.add_obstacle_button)

        self.update_obstacle_button = QtWidgets.QPushButton("Update", objectName="CompactSecondaryButton")
        self.update_obstacle_button.clicked.connect(self.update_selected_obstacle)
        buttons_row.addWidget(self.update_obstacle_button)

        self.remove_obstacle_button = QtWidgets.QPushButton("Remove", objectName="CompactSecondaryButton")
        self.remove_obstacle_button.clicked.connect(self.remove_selected_obstacle)
        buttons_row.addWidget(self.remove_obstacle_button)

        self.clear_obstacles_button = QtWidgets.QPushButton("Clear", objectName="ClearButton")
        self.clear_obstacles_button.clicked.connect(self.clear_custom_obstacles)
        buttons_row.addWidget(self.clear_obstacles_button)
        layout.addLayout(buttons_row)

        self.obstacle_table = QtWidgets.QTableWidget(0, 4)
        self.obstacle_table.setHorizontalHeaderLabels(("X", "Y", "Radius", "Height"))
        self.obstacle_table.verticalHeader().setVisible(False)
        self.obstacle_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.obstacle_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.obstacle_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.obstacle_table.setFixedHeight(104)
        self.obstacle_table.horizontalHeader().setStretchLastSection(True)
        self.obstacle_table.itemSelectionChanged.connect(self._load_selected_obstacle_into_controls)
        self.obstacle_table.itemSelectionChanged.connect(self._update_selected_obstacle_highlight)
        layout.addWidget(self.obstacle_table)

        return frame

    def _section_label(self, text: str) -> QtWidgets.QLabel:
        return QtWidgets.QLabel(text, objectName="SectionLabel")

    def _section_header_row(self, title: str, hint: str) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._section_label(title))
        row.addStretch(1)
        hint_label = QtWidgets.QLabel(hint, objectName="HintLabel")
        row.addWidget(hint_label)
        return row

    def _double_spin(self, minimum: float, maximum: float, value: float, step: float) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.setDecimals(2)
        spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        return spin

    def _int_spin(self, minimum: int, maximum: int, value: int, step: int) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        return spin

    def _populate_environments(self) -> None:
        for environment in self.environments.values():
            self.environment_combo.addItem(environment.name, environment.key)

    def _on_environment_changed(self) -> None:
        env_key = self.environment_combo.currentData()
        if env_key:
            self.load_environment(env_key)

    def _update_timer_speed(self) -> None:
        fps = self.speed_slider.value()
        self.speed_label.setText(f"{fps} fps")
        self.timer.setInterval(max(40, int(1000 / fps)))

    def _build_config(self) -> FPAPFConfig:
        return FPAPFConfig(
            step_scale=self.step_scale_spin.value(),
            alpha=self.alpha_spin.value(),
            swarm_size=self.swarm_spin.value(),
            pso_iterations=self.iter_spin.value(),
        )

    def _is_custom_environment(self, env_key: str) -> bool:
        return env_key == "custom_sandbox" or env_key.startswith("saved_custom_")

    def _environment_to_dict(self, environment: EnvironmentSpec) -> dict[str, object]:
        return {
            "key": environment.key,
            "name": environment.name,
            "description": environment.description,
            "start": list(environment.start),
            "goal": list(environment.goal),
            "bounds": [list(axis_bounds) for axis_bounds in environment.bounds],
            "accent_rgb": list(environment.accent_rgb),
            "obstacles": [
                {
                    "center_xy": list(obstacle.center_xy),
                    "radius": obstacle.radius,
                    "height": obstacle.height,
                    "color_rgba": list(obstacle.color_rgba),
                }
                for obstacle in environment.obstacles
            ],
        }

    def _environment_from_dict(self, data: dict[str, object]) -> EnvironmentSpec:
        obstacles_data = data.get("obstacles", [])
        obstacles = tuple(
            CylinderObstacle(
                center_xy=tuple(obstacle["center_xy"]),  # type: ignore[index,arg-type]
                radius=float(obstacle["radius"]),  # type: ignore[index]
                height=float(obstacle["height"]),  # type: ignore[index]
                color_rgba=tuple(obstacle.get("color_rgba", (0.44, 0.63, 0.76, 0.35))),  # type: ignore[attr-defined,arg-type]
            )
            for obstacle in obstacles_data  # type: ignore[assignment]
        )
        return EnvironmentSpec(
            key=str(data["key"]),
            name=str(data["name"]),
            description=str(data.get("description", "Saved custom environment.")),
            start=tuple(data["start"]),  # type: ignore[arg-type]
            goal=tuple(data["goal"]),  # type: ignore[arg-type]
            bounds=tuple(tuple(axis_bounds) for axis_bounds in data["bounds"]),  # type: ignore[arg-type]
            obstacles=obstacles,
            accent_rgb=tuple(data.get("accent_rgb", (0.22, 0.58, 0.62))),  # type: ignore[arg-type]
        )

    def _load_saved_custom_environments(self) -> Dict[str, EnvironmentSpec]:
        if not SAVED_ENVIRONMENTS_PATH.exists():
            return {}
        try:
            raw_data = json.loads(SAVED_ENVIRONMENTS_PATH.read_text(encoding="utf-8"))
            environments = [self._environment_from_dict(item) for item in raw_data.get("environments", [])]
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return {}
        return {environment.key: environment for environment in environments}

    def _write_saved_custom_environments(self) -> None:
        saved = [
            self._environment_to_dict(environment)
            for environment in self.environments.values()
            if environment.key.startswith("saved_custom_")
        ]
        SAVED_ENVIRONMENTS_PATH.write_text(
            json.dumps({"environments": saved}, indent=2),
            encoding="utf-8",
        )

    def _slugify_environment_name(self, name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return slug or "environment"

    def save_custom_environment(self) -> None:
        name = self.custom_name_edit.text().strip() or "Saved Custom Environment"
        key = f"saved_custom_{self._slugify_environment_name(name)}"
        custom_environment = replace(
            self.current_environment,
            key=key,
            name=name,
            description="Saved custom environment.",
        )
        self.environments[key] = custom_environment
        self._write_saved_custom_environments()

        combo_index = self.environment_combo.findData(key)
        if combo_index < 0:
            self.environment_combo.addItem(custom_environment.name, custom_environment.key)
        else:
            self.environment_combo.setItemText(combo_index, custom_environment.name)
        self.load_environment(key)

    def _custom_obstacle_color(self, index: int) -> tuple[float, float, float, float]:
        colors = (
            (0.44, 0.63, 0.76, 0.35),
            (0.89, 0.66, 0.31, 0.35),
            (0.53, 0.70, 0.59, 0.35),
            (0.78, 0.42, 0.45, 0.35),
            (0.68, 0.58, 0.82, 0.35),
            (0.86, 0.70, 0.40, 0.35),
        )
        return colors[index % len(colors)]

    def _refresh_custom_table(self) -> None:
        if not hasattr(self, "obstacle_table"):
            return
        self._refresh_custom_point_controls()
        if hasattr(self, "custom_name_edit"):
            self.custom_name_edit.setText(
                "" if self.current_environment.key == "custom_sandbox" else self.current_environment.name
            )
        self.obstacle_table.blockSignals(True)
        self.obstacle_table.setRowCount(len(self.current_environment.obstacles))
        for row, obstacle in enumerate(self.current_environment.obstacles):
            values = (
                obstacle.center_xy[0],
                obstacle.center_xy[1],
                obstacle.radius,
                obstacle.height,
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(f"{value:.1f}")
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.obstacle_table.setItem(row, column, item)
        self.obstacle_table.blockSignals(False)

    def _refresh_custom_point_controls(self) -> None:
        if not hasattr(self, "start_x_spin"):
            return
        point_widgets = (
            self.start_x_spin,
            self.start_y_spin,
            self.start_z_spin,
            self.goal_x_spin,
            self.goal_y_spin,
            self.goal_z_spin,
        )
        for widget in point_widgets:
            widget.blockSignals(True)
        self.start_x_spin.setValue(self.current_environment.start[0])
        self.start_y_spin.setValue(self.current_environment.start[1])
        self.start_z_spin.setValue(self.current_environment.start[2])
        self.goal_x_spin.setValue(self.current_environment.goal[0])
        self.goal_y_spin.setValue(self.current_environment.goal[1])
        self.goal_z_spin.setValue(self.current_environment.goal[2])
        for widget in point_widgets:
            widget.blockSignals(False)

    def _load_selected_obstacle_into_controls(self) -> None:
        selected_rows = self.obstacle_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        obstacle = self.current_environment.obstacles[selected_rows[0].row()]
        self.custom_x_spin.setValue(obstacle.center_xy[0])
        self.custom_y_spin.setValue(obstacle.center_xy[1])
        self.custom_radius_spin.setValue(obstacle.radius)
        self.custom_height_spin.setValue(obstacle.height)

    def _selected_obstacle_row(self) -> int | None:
        selected_rows = self.obstacle_table.selectionModel().selectedRows()
        if not selected_rows:
            return None
        return selected_rows[0].row()

    def _update_selected_obstacle_highlight(self) -> None:
        if self.current_environment.key != "custom_sandbox":
            return
        self.scene.highlight_obstacle(self._selected_obstacle_row())

    def _update_custom_preview(self) -> None:
        if self.current_environment.key != "custom_sandbox":
            return
        self.scene.set_preview_obstacle(self._custom_obstacle_from_controls(-1))

    def _custom_obstacle_from_controls(self, index: int) -> CylinderObstacle:
        return CylinderObstacle(
            center_xy=(self.custom_x_spin.value(), self.custom_y_spin.value()),
            radius=self.custom_radius_spin.value(),
            height=self.custom_height_spin.value(),
            color_rgba=self._custom_obstacle_color(index),
        )

    def _replace_custom_environment(self, **changes: object) -> None:
        env_key = self.current_environment.key
        if not self._is_custom_environment(env_key):
            env_key = "custom_sandbox"
        custom_environment = replace(self.environments[env_key], **changes)
        self.environments[env_key] = custom_environment
        self.load_environment(env_key)

    def _replace_custom_obstacles(self, obstacles: tuple[CylinderObstacle, ...]) -> None:
        self._replace_custom_environment(obstacles=obstacles)

    def apply_custom_start_goal(self) -> None:
        self._replace_custom_environment(
            start=(self.start_x_spin.value(), self.start_y_spin.value(), self.start_z_spin.value()),
            goal=(self.goal_x_spin.value(), self.goal_y_spin.value(), self.goal_z_spin.value()),
        )

    def add_custom_obstacle(self) -> None:
        obstacles = self.current_environment.obstacles
        self._replace_custom_obstacles(obstacles + (self._custom_obstacle_from_controls(len(obstacles)),))
        self.obstacle_table.selectRow(len(obstacles))

    def update_selected_obstacle(self) -> None:
        selected_rows = self.obstacle_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        obstacles = list(self.current_environment.obstacles)
        obstacles[row] = self._custom_obstacle_from_controls(row)
        self._replace_custom_obstacles(tuple(obstacles))
        self.obstacle_table.selectRow(row)

    def remove_selected_obstacle(self) -> None:
        selected_rows = self.obstacle_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        obstacles = tuple(
            obstacle for index, obstacle in enumerate(self.current_environment.obstacles) if index != row
        )
        self._replace_custom_obstacles(obstacles)

    def clear_custom_obstacles(self) -> None:
        self._replace_custom_obstacles(())

    def load_environment(self, env_key: str) -> None:
        self.timer.stop()
        self.play_button.setText("Play")
        self.current_environment = self.environments[env_key]
        self.environment_combo.blockSignals(True)
        combo_index = self.environment_combo.findData(env_key)
        if combo_index >= 0:
            self.environment_combo.setCurrentIndex(combo_index)
        self.environment_combo.blockSignals(False)

        is_custom = self._is_custom_environment(env_key)
        self.custom_editor.setVisible(is_custom)
        if is_custom:
            self._refresh_custom_table()
        self.scene.load_environment(self.current_environment, show_axes=True)
        if is_custom:
            self._update_custom_preview()
        self.current_path = []
        self.play_index = 0
        self._clear_metrics()
        self.plan_current_environment()

    def load_next_environment(self) -> None:
        keys = list(self.environments.keys())
        current_idx = keys.index(self.current_environment.key)
        self.load_environment(keys[(current_idx + 1) % len(keys)])

    def plan_current_environment(self) -> None:
        self.timer.stop()
        self.play_button.setText("Play")
        config = self._build_config()
        env = self.current_environment
        self.current_path = fp_apf_plan(
            start=env.start,
            goal=env.goal,
            obstacles=env.obstacles,
            cfg=config,
            seed=7,
        )
        self.current_metrics = summarize_path(self.current_path, env.obstacles, config.drone_radius)
        self.scene.set_path(self.current_path, env.accent_rgb)
        self.play_index = 0
        self.scene.update_progress(self.play_index)
        self._update_metrics(self.current_metrics)
        self._update_timer_speed()

    def toggle_animation(self) -> None:
        if not self.current_path:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.timer.start()
            self.play_button.setText("Pause")

    def reset_animation(self) -> None:
        self.timer.stop()
        self.play_button.setText("Play")
        self.play_index = 0
        self.scene.update_progress(self.play_index)

    def _advance_animation(self) -> None:
        if not self.current_path:
            self.timer.stop()
            return
        self.play_index += 1
        if self.play_index >= len(self.current_path):
            self.play_index = len(self.current_path) - 1
            self.timer.stop()
            self.play_button.setText("Play")
        self.scene.update_progress(self.play_index)

    def _clear_metrics(self) -> None:
        for widget in (self.points_value, self.length_value, self.avg_clearance_value, self.min_clearance_value):
            widget.setText("-")

    def _update_metrics(self, metrics: PathMetrics) -> None:
        self.points_value.setText(str(metrics.point_count))
        self.length_value.setText(f"{metrics.path_length:.2f}")
        self.avg_clearance_value.setText(f"{metrics.average_clearance:.2f}")
        self.min_clearance_value.setText(f"{metrics.minimum_clearance:.2f}")
