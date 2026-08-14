from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Dict

from PySide6 import QtCore, QtWidgets

from ..simulation.environments import build_environment_library
from ..simulation.geometry import CylinderObstacle, EnvironmentSpec, Obstacle, PolygonPrismObstacle
from ..simulation.metrics import PathMetrics, summarize_path
from ..simulation.classic_apf import ClassicAPFConfig, classic_apf_plan
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

        planner_col = QtWidgets.QVBoxLayout()
        planner_col.setSpacing(4)
        planner_col.addWidget(QtWidgets.QLabel("Planner", objectName="HeaderFieldLabel"))
        self.planner_combo = QtWidgets.QComboBox()
        self.planner_combo.setObjectName("HeaderCombo")
        self.planner_combo.addItem("FP-APF", "fp_apf")
        self.planner_combo.addItem("Classic APF", "classic_apf")
        self.planner_combo.currentIndexChanged.connect(self._on_planner_changed)
        planner_col.addWidget(self.planner_combo)
        layout.addLayout(planner_col, 0)

        return frame

    def _build_sidebar(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame(objectName="SidebarCard")
        frame.setMinimumWidth(300)
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        layout.addWidget(self._section_label("Environment"))
        environment_row = QtWidgets.QHBoxLayout()
        environment_row.setSpacing(8)
        self.environment_combo = QtWidgets.QComboBox()
        self.environment_combo.currentIndexChanged.connect(self._on_environment_changed)
        environment_row.addWidget(self.environment_combo, 1)
        self.next_env_button = QtWidgets.QPushButton("Next", objectName="CompactSecondaryButton")
        self.next_env_button.clicked.connect(self.load_next_environment)
        environment_row.addWidget(self.next_env_button, 0)
        self.delete_environment_button = QtWidgets.QPushButton("Delete", objectName="ClearButton")
        self.delete_environment_button.clicked.connect(self.delete_current_environment)
        environment_row.addWidget(self.delete_environment_button, 0)
        layout.addLayout(environment_row)

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

        self.top_view_button = QtWidgets.QPushButton("Top View", objectName="CompactSecondaryButton")
        self.top_view_button.clicked.connect(self.scene.set_exact_top_view)
        controls_row.addWidget(self.top_view_button)
        layout.addLayout(controls_row)

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

        self.start_goal_editor = self._build_start_goal_editor()
        layout.addWidget(self.start_goal_editor)

        self.fp_apf_params_frame = QtWidgets.QFrame()
        fp_apf_params_layout = QtWidgets.QVBoxLayout(self.fp_apf_params_frame)
        fp_apf_params_layout.setContentsMargins(0, 0, 0, 0)
        fp_apf_params_layout.setSpacing(10)
        fp_apf_params_layout.addWidget(self._section_label("Planner Parameters"))

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
        fp_apf_params_layout.addLayout(params_grid)
        layout.addWidget(self.fp_apf_params_frame)

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

        layout.addLayout(self._section_header_row("Custom Obstacles", "3D cylinders and polygon prisms"))

        shape_row = QtWidgets.QHBoxLayout()
        shape_row.setSpacing(6)
        self.custom_shape_combo = QtWidgets.QComboBox()
        for label, key in (
            ("Circle", "circle"),
            ("Square", "square"),
            ("Rectangle", "rectangle"),
            ("Triangle", "triangle"),
            ("Custom polygon", "custom"),
        ):
            self.custom_shape_combo.addItem(label, key)
        self.custom_shape_combo.currentIndexChanged.connect(self._update_custom_control_visibility)
        shape_row.addWidget(QtWidgets.QLabel("Shape"))
        shape_row.addWidget(self.custom_shape_combo, 1)
        layout.addLayout(shape_row)

        dimensions_grid = QtWidgets.QGridLayout()
        dimensions_grid.setHorizontalSpacing(6)
        dimensions_grid.setVerticalSpacing(6)

        self.custom_x_spin = self._double_spin(0.0, 155.0, 60.0, 1.0)
        self.custom_y_spin = self._double_spin(0.0, 155.0, 60.0, 1.0)
        self.custom_radius_spin = self._double_spin(1.0, 30.0, 8.0, 0.5)
        self.custom_width_spin = self._double_spin(2.0, 60.0, 16.0, 0.5)
        self.custom_length_spin = self._double_spin(2.0, 60.0, 24.0, 0.5)
        self.custom_height_spin = self._double_spin(5.0, 80.0, 40.0, 1.0)
        self.custom_base_z_spin = self._double_spin(0.0, 80.0, 0.0, 1.0)
        self.custom_rotation_spin = self._double_spin(-180.0, 180.0, 0.0, 5.0)
        self.custom_margin_spin = self._double_spin(0.0, 10.0, 0.0, 0.25)
        for spin in (
            self.custom_x_spin,
            self.custom_y_spin,
            self.custom_radius_spin,
            self.custom_width_spin,
            self.custom_length_spin,
            self.custom_height_spin,
            self.custom_base_z_spin,
            self.custom_rotation_spin,
            self.custom_margin_spin,
        ):
            spin.valueChanged.connect(self._update_custom_preview)

        self.custom_radius_label = QtWidgets.QLabel("Radius")
        self.custom_width_label = QtWidgets.QLabel("Width")
        self.custom_length_label = QtWidgets.QLabel("Length")
        for row, (label, spin) in enumerate((
            ("X", self.custom_x_spin),
            ("Y", self.custom_y_spin),
            ("Height", self.custom_height_spin),
            ("Base Z", self.custom_base_z_spin),
            ("Rotation", self.custom_rotation_spin),
            ("Margin", self.custom_margin_spin),
        )):
            column = 0 if row < 3 else 2
            grid_row = row if row < 3 else row - 3
            dimensions_grid.addWidget(QtWidgets.QLabel(label), grid_row, column)
            dimensions_grid.addWidget(spin, grid_row, column + 1)
        dimensions_grid.addWidget(self.custom_radius_label, 3, 0)
        dimensions_grid.addWidget(self.custom_radius_spin, 3, 1)
        dimensions_grid.addWidget(self.custom_width_label, 4, 0)
        dimensions_grid.addWidget(self.custom_width_spin, 4, 1)
        dimensions_grid.addWidget(self.custom_length_label, 4, 2)
        dimensions_grid.addWidget(self.custom_length_spin, 4, 3)
        layout.addLayout(dimensions_grid)

        self.custom_vertices_edit = QtWidgets.QLineEdit("-8,-8; 8,-8; 10,2; 0,10; -10,2")
        self.custom_vertices_edit.setPlaceholderText("Local vertices: x,y; x,y; x,y")
        self.custom_vertices_edit.textChanged.connect(self._update_custom_preview)
        self.custom_vertices_label = QtWidgets.QLabel("Vertices")
        vertices_row = QtWidgets.QHBoxLayout()
        vertices_row.addWidget(self.custom_vertices_label)
        vertices_row.addWidget(self.custom_vertices_edit, 1)
        layout.addLayout(vertices_row)

        for label, widget in (
            (self.custom_radius_label, self.custom_radius_spin),
            (self.custom_width_label, self.custom_width_spin),
            (self.custom_length_label, self.custom_length_spin),
            (self.custom_vertices_label, self.custom_vertices_edit),
        ):
            widget.setProperty("pairedLabel", label)

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

        self.obstacle_table = QtWidgets.QTableWidget(0, 6)
        self.obstacle_table.setHorizontalHeaderLabels(("Shape", "X", "Y", "Size", "Height", "Base Z"))
        self.obstacle_table.verticalHeader().setVisible(False)
        self.obstacle_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.obstacle_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.obstacle_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.obstacle_table.setFixedHeight(104)
        self.obstacle_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.obstacle_table.itemSelectionChanged.connect(self._load_selected_obstacle_into_controls)
        self.obstacle_table.itemSelectionChanged.connect(self._update_selected_obstacle_highlight)
        layout.addWidget(self.obstacle_table)

        self._update_custom_control_visibility()
        return frame

    def _build_start_goal_editor(self) -> QtWidgets.QFrame:
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

    def _build_classic_apf_config(self) -> ClassicAPFConfig:
        return ClassicAPFConfig()

    def _on_planner_changed(self) -> None:
        planner_key = self.planner_combo.currentData()
        if hasattr(self, "fp_apf_params_frame"):
            self.fp_apf_params_frame.setVisible(planner_key == "fp_apf")
        if hasattr(self, "current_environment"):
            self.plan_current_environment()

    def _is_custom_environment(self, env_key: str) -> bool:
        return env_key == "custom_sandbox"

    def _environment_to_dict(self, environment: EnvironmentSpec) -> dict[str, object]:
        def obstacle_to_dict(obstacle: Obstacle) -> dict[str, object]:
            common = {
                "center_xy": list(obstacle.center_xy),
                "height": obstacle.height,
                "base_z": obstacle.base_z,
                "safety_margin": obstacle.safety_margin,
                "color_rgba": list(obstacle.color_rgba),
            }
            if isinstance(obstacle, PolygonPrismObstacle):
                return {
                    **common,
                    "type": "polygon_prism",
                    "vertices_xy": [list(vertex) for vertex in obstacle.vertices_xy],
                    "rotation_deg": obstacle.rotation_deg,
                }
            return {**common, "type": "cylinder", "radius": obstacle.radius}

        return {
            "key": environment.key,
            "name": environment.name,
            "description": environment.description,
            "start": list(environment.start),
            "goal": list(environment.goal),
            "bounds": [list(axis_bounds) for axis_bounds in environment.bounds],
            "accent_rgb": list(environment.accent_rgb),
            "obstacles": [obstacle_to_dict(obstacle) for obstacle in environment.obstacles],
        }

    def _environment_from_dict(self, data: dict[str, object]) -> EnvironmentSpec:
        obstacles_data = data.get("obstacles", [])

        def obstacle_from_dict(obstacle: dict[str, object]) -> Obstacle:
            common = {
                "center_xy": tuple(obstacle["center_xy"]),
                "height": float(obstacle["height"]),
                "color_rgba": tuple(obstacle.get("color_rgba", (0.44, 0.63, 0.76, 0.35))),
                "base_z": float(obstacle.get("base_z", 0.0)),
                "safety_margin": float(obstacle.get("safety_margin", 0.0)),
            }
            if obstacle.get("type") == "polygon_prism":
                return PolygonPrismObstacle(
                    vertices_xy=tuple(tuple(vertex) for vertex in obstacle["vertices_xy"]),  # type: ignore[arg-type]
                    rotation_deg=float(obstacle.get("rotation_deg", 0.0)),
                    **common,  # type: ignore[arg-type]
                )
            return CylinderObstacle(
                radius=float(obstacle["radius"]),
                **common,  # type: ignore[arg-type]
            )

        obstacles = tuple(
            obstacle_from_dict(obstacle)
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

    def delete_current_environment(self) -> None:
        env_key = self.environment_combo.currentData()
        if not env_key or not str(env_key).startswith("saved_custom_"):
            QtWidgets.QMessageBox.information(
                self,
                "Environment Delete",
                "Only saved custom environments can be deleted.",
            )
            return

        environment = self.environments[env_key]
        confirmed = QtWidgets.QMessageBox.question(
            self,
            "Delete Environment",
            f"Delete '{environment.name}'?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if confirmed != QtWidgets.QMessageBox.Yes:
            return

        del self.environments[env_key]
        self._write_saved_custom_environments()

        combo_index = self.environment_combo.findData(env_key)
        if combo_index >= 0:
            self.environment_combo.removeItem(combo_index)
        self.load_environment("custom_sandbox")

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

    def _update_custom_control_visibility(self) -> None:
        shape = self.custom_shape_combo.currentData()
        is_circle = shape == "circle"
        is_custom = shape == "custom"
        uses_length = shape == "rectangle"

        self.custom_radius_label.setVisible(is_circle)
        self.custom_radius_spin.setVisible(is_circle)
        self.custom_width_label.setVisible(not is_circle and not is_custom)
        self.custom_width_spin.setVisible(not is_circle and not is_custom)
        self.custom_length_label.setVisible(uses_length)
        self.custom_length_spin.setVisible(uses_length)
        self.custom_vertices_label.setVisible(is_custom)
        self.custom_vertices_edit.setVisible(is_custom)
        self._update_custom_preview()

    def _parse_custom_vertices(self) -> tuple[tuple[float, float], ...]:
        vertices = []
        for pair in self.custom_vertices_edit.text().split(";"):
            coordinates = [value.strip() for value in pair.split(",")]
            if len(coordinates) != 2:
                raise ValueError("Each vertex must use the format x,y.")
            vertices.append((float(coordinates[0]), float(coordinates[1])))
        return tuple(vertices)

    def _polygon_vertices_from_controls(self) -> tuple[tuple[float, float], ...]:
        shape = self.custom_shape_combo.currentData()
        width = self.custom_width_spin.value()
        half_width = width / 2.0
        if shape == "square":
            return ((-half_width, -half_width), (half_width, -half_width), (half_width, half_width), (-half_width, half_width))
        if shape == "rectangle":
            half_length = self.custom_length_spin.value() / 2.0
            return ((-half_width, -half_length), (half_width, -half_length), (half_width, half_length), (-half_width, half_length))
        if shape == "triangle":
            return ((0.0, half_width), (half_width, -half_width), (-half_width, -half_width))
        return self._parse_custom_vertices()

    def _obstacle_shape_name(self, obstacle: Obstacle) -> str:
        if isinstance(obstacle, CylinderObstacle):
            return "Circle"
        vertex_count = len(obstacle.vertices_xy)
        if vertex_count == 3:
            return "Triangle"
        if vertex_count == 4:
            side_lengths = [
                ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
                for start, end in zip(obstacle.vertices_xy, obstacle.vertices_xy[1:] + obstacle.vertices_xy[:1])
            ]
            if max(side_lengths) - min(side_lengths) < 1e-6:
                return "Square"
            return "Rectangle"
        return "Polygon"

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
            if isinstance(obstacle, CylinderObstacle):
                size = f"R {obstacle.radius:.1f}"
            else:
                xs = [vertex[0] for vertex in obstacle.vertices_xy]
                ys = [vertex[1] for vertex in obstacle.vertices_xy]
                size = f"{max(xs) - min(xs):.1f}x{max(ys) - min(ys):.1f}"
            values = (
                self._obstacle_shape_name(obstacle),
                obstacle.center_xy[0],
                obstacle.center_xy[1],
                size,
                obstacle.height,
                obstacle.base_z,
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(f"{value:.1f}" if isinstance(value, float) else str(value))
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
        self.custom_height_spin.setValue(obstacle.height)
        self.custom_base_z_spin.setValue(obstacle.base_z)
        self.custom_margin_spin.setValue(obstacle.safety_margin)
        if isinstance(obstacle, CylinderObstacle):
            self.custom_shape_combo.setCurrentIndex(self.custom_shape_combo.findData("circle"))
            self.custom_radius_spin.setValue(obstacle.radius)
            self.custom_rotation_spin.setValue(0.0)
            return

        shape_name = self._obstacle_shape_name(obstacle).lower()
        shape = shape_name if shape_name in {"square", "rectangle", "triangle"} else "custom"
        self.custom_shape_combo.setCurrentIndex(self.custom_shape_combo.findData(shape))
        self.custom_rotation_spin.setValue(obstacle.rotation_deg)
        xs = [vertex[0] for vertex in obstacle.vertices_xy]
        ys = [vertex[1] for vertex in obstacle.vertices_xy]
        self.custom_width_spin.setValue(max(xs) - min(xs))
        self.custom_length_spin.setValue(max(ys) - min(ys))
        self.custom_vertices_edit.setText(
            "; ".join(f"{x:g},{y:g}" for x, y in obstacle.vertices_xy)
        )

    def _selected_obstacle_row(self) -> int | None:
        selected_rows = self.obstacle_table.selectionModel().selectedRows()
        if not selected_rows:
            return None
        return selected_rows[0].row()

    def _update_selected_obstacle_highlight(self) -> None:
        if not self._is_custom_environment(self.current_environment.key):
            return
        self.scene.highlight_obstacle(self._selected_obstacle_row())
        self._update_custom_preview()

    def _update_custom_preview(self) -> None:
        if not self._is_custom_environment(self.current_environment.key):
            return
        try:
            obstacle = self._custom_obstacle_from_controls(-1)
        except ValueError:
            self.scene.set_preview_obstacle(None, selected_index=self._selected_obstacle_row())
            return
        self.scene.set_preview_obstacle(obstacle, selected_index=self._selected_obstacle_row())

    def _custom_obstacle_from_controls(self, index: int) -> Obstacle:
        common = {
            "center_xy": (self.custom_x_spin.value(), self.custom_y_spin.value()),
            "height": self.custom_height_spin.value(),
            "color_rgba": self._custom_obstacle_color(index),
            "base_z": self.custom_base_z_spin.value(),
            "safety_margin": self.custom_margin_spin.value(),
        }
        if self.custom_shape_combo.currentData() == "circle":
            return CylinderObstacle(radius=self.custom_radius_spin.value(), **common)
        return PolygonPrismObstacle(
            vertices_xy=self._polygon_vertices_from_controls(),
            rotation_deg=self.custom_rotation_spin.value(),
            **common,
        )

    def _replace_custom_environment(self, **changes: object) -> None:
        env_key = self.current_environment.key
        if env_key.startswith("saved_custom_"):
            custom_environment = replace(
                self.current_environment,
                key="custom_sandbox",
                name="Custom Sandbox",
                description="Editable environment for manually placing 3D cylinder and polygon-prism obstacles.",
                **changes,
            )
            self.environments["custom_sandbox"] = custom_environment
            self.load_environment("custom_sandbox")
            return
        if not self._is_custom_environment(env_key):
            env_key = "custom_sandbox"
        custom_environment = replace(self.environments[env_key], **changes)
        self.environments[env_key] = custom_environment
        self.load_environment(env_key)

    def _replace_custom_obstacles(self, obstacles: tuple[Obstacle, ...]) -> None:
        self._replace_custom_environment(obstacles=obstacles)

    def apply_custom_start_goal(self) -> None:
        self._replace_custom_environment(
            start=(self.start_x_spin.value(), self.start_y_spin.value(), self.start_z_spin.value()),
            goal=(self.goal_x_spin.value(), self.goal_y_spin.value(), self.goal_z_spin.value()),
        )

    def add_custom_obstacle(self) -> None:
        obstacles = self.current_environment.obstacles
        try:
            obstacle = self._custom_obstacle_from_controls(len(obstacles))
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "Invalid Polygon", str(error))
            return
        self._replace_custom_obstacles(obstacles + (obstacle,))
        self.obstacle_table.selectRow(len(obstacles))

    def update_selected_obstacle(self) -> None:
        selected_rows = self.obstacle_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        obstacles = list(self.current_environment.obstacles)
        try:
            obstacles[row] = self._custom_obstacle_from_controls(row)
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "Invalid Polygon", str(error))
            return
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
        self.start_goal_editor.setVisible(is_custom)
        if is_custom:
            self._refresh_custom_table()
        if hasattr(self, "delete_environment_button"):
            self.delete_environment_button.setEnabled(env_key.startswith("saved_custom_"))
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
        env = self.current_environment
        planner_key = self.planner_combo.currentData()

        if planner_key == "classic_apf":
            classic_config = self._build_classic_apf_config()
            self.current_path = classic_apf_plan(
                start=env.start,
                goal=env.goal,
                obstacles=env.obstacles,
                cfg=classic_config,
                bounds=env.bounds,
                seed=7,
            )
            drone_radius = classic_config.drone_radius
        else:
            config = self._build_config()
            self.current_path = fp_apf_plan(
                start=env.start,
                goal=env.goal,
                obstacles=env.obstacles,
                cfg=config,
                seed=7,
            )
            drone_radius = config.drone_radius

        self.current_metrics = summarize_path(self.current_path, env.obstacles, drone_radius)
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
