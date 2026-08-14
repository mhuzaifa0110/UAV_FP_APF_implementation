from __future__ import annotations

from typing import Dict, List

from .geometry import CylinderObstacle, EnvironmentSpec, PolygonPrismObstacle


def build_environment_library() -> Dict[str, EnvironmentSpec]:
    return {
        "canyon_gate": EnvironmentSpec(
            key="canyon_gate",
            name="Canyon Gate",
            description="A balanced corridor with tall pillars that force an early climb and then a smooth descent to the goal.",
            start=(10.0, 10.0, 0.0),
            goal=(100.0, 150.0, 50.0),
            bounds=((0.0, 130.0), (0.0, 170.0), (0.0, 70.0)),
            obstacles=(
                CylinderObstacle((40.0, 45.0), 8.0, 35.0, (0.79, 0.56, 0.34, 0.40)),
                CylinderObstacle((60.0, 80.0), 10.0, 45.0, (0.88, 0.67, 0.34, 0.38)),
                CylinderObstacle((78.0, 110.0), 12.0, 55.0, (0.70, 0.54, 0.30, 0.38)),
                CylinderObstacle((95.0, 95.0), 8.0, 30.0, (0.53, 0.67, 0.62, 0.38)),
            ),
            accent_rgb=(0.12, 0.54, 0.58),
        ),
        "dense_urban": EnvironmentSpec(
            key="dense_urban",
            name="Dense Urban",
            description="A tighter obstacle field that rewards careful local refinement and stronger clearance management.",
            start=(8.0, 8.0, 0.0),
            goal=(118.0, 138.0, 42.0),
            bounds=((0.0, 140.0), (0.0, 155.0), (0.0, 70.0)),
            obstacles=(
                CylinderObstacle((28.0, 28.0), 7.0, 35.0, (0.82, 0.49, 0.34, 0.35)),
                CylinderObstacle((42.0, 52.0), 8.0, 42.0, (0.89, 0.63, 0.29, 0.35)),
                CylinderObstacle((58.0, 70.0), 9.0, 50.0, (0.65, 0.76, 0.67, 0.35)),
                CylinderObstacle((72.0, 88.0), 7.5, 38.0, (0.65, 0.53, 0.78, 0.35)),
                CylinderObstacle((88.0, 96.0), 8.5, 55.0, (0.37, 0.66, 0.75, 0.35)),
                CylinderObstacle((102.0, 116.0), 9.0, 52.0, (0.78, 0.42, 0.45, 0.35)),
                CylinderObstacle((54.0, 102.0), 10.0, 46.0, (0.55, 0.64, 0.32, 0.35)),
                CylinderObstacle((90.0, 66.0), 7.0, 35.0, (0.90, 0.74, 0.44, 0.35)),
            ),
            accent_rgb=(0.80, 0.42, 0.24),
        ),
        "spire_garden": EnvironmentSpec(
            key="spire_garden",
            name="Spire Garden",
            description="A higher-altitude environment with staggered tower heights that encourages more vertical planning variation.",
            start=(12.0, 15.0, 0.0),
            goal=(132.0, 126.0, 58.0),
            bounds=((0.0, 150.0), (0.0, 145.0), (0.0, 90.0)),
            obstacles=(
                CylinderObstacle((30.0, 35.0), 6.0, 45.0, (0.72, 0.63, 0.31, 0.36)),
                CylinderObstacle((48.0, 62.0), 8.0, 68.0, (0.42, 0.63, 0.72, 0.36)),
                CylinderObstacle((72.0, 58.0), 7.0, 54.0, (0.61, 0.69, 0.49, 0.36)),
                CylinderObstacle((88.0, 84.0), 10.0, 74.0, (0.81, 0.57, 0.43, 0.36)),
                CylinderObstacle((110.0, 92.0), 9.0, 63.0, (0.84, 0.71, 0.39, 0.36)),
                CylinderObstacle((98.0, 118.0), 8.0, 52.0, (0.48, 0.66, 0.58, 0.36)),
            ),
            accent_rgb=(0.34, 0.57, 0.72),
        ),
        "harbor_weave": EnvironmentSpec(
            key="harbor_weave",
            name="Harbor Weave",
            description="A broad mixed environment with cross-pattern obstacles, useful for comparing detours and path compactness.",
            start=(6.0, 18.0, 0.0),
            goal=(142.0, 140.0, 48.0),
            bounds=((0.0, 160.0), (0.0, 160.0), (0.0, 75.0)),
            obstacles=(
                CylinderObstacle((40.0, 40.0), 9.0, 40.0, (0.44, 0.63, 0.76, 0.35)),
                CylinderObstacle((60.0, 70.0), 11.0, 44.0, (0.89, 0.66, 0.31, 0.35)),
                CylinderObstacle((84.0, 42.0), 8.0, 37.0, (0.76, 0.52, 0.42, 0.35)),
                CylinderObstacle((84.0, 100.0), 8.0, 60.0, (0.53, 0.70, 0.59, 0.35)),
                CylinderObstacle((110.0, 74.0), 10.0, 52.0, (0.68, 0.58, 0.82, 0.35)),
                CylinderObstacle((126.0, 112.0), 9.0, 47.0, (0.85, 0.71, 0.42, 0.35)),
                CylinderObstacle((54.0, 118.0), 10.0, 50.0, (0.40, 0.62, 0.67, 0.35)),
            ),
            accent_rgb=(0.18, 0.60, 0.46),
        ),
        "narrow_passage": EnvironmentSpec(
            key="narrow_passage",
            name="Narrow Passage",
            description="Two offset obstacle walls create a slim central channel for stress-testing local-minima behavior.",
            start=(10.0, 20.0, 0.0),
            goal=(148.0, 118.0, 46.0),
            bounds=((0.0, 165.0), (0.0, 135.0), (0.0, 78.0)),
            obstacles=(
                CylinderObstacle((38.0, 36.0), 8.5, 45.0, (0.50, 0.66, 0.74, 0.36)),
                CylinderObstacle((55.0, 51.0), 9.0, 52.0, (0.79, 0.56, 0.34, 0.36)),
                CylinderObstacle((73.0, 68.0), 8.5, 58.0, (0.53, 0.70, 0.59, 0.36)),
                CylinderObstacle((96.0, 56.0), 9.5, 50.0, (0.68, 0.58, 0.82, 0.36)),
                CylinderObstacle((115.0, 72.0), 8.0, 47.0, (0.86, 0.68, 0.39, 0.36)),
                CylinderObstacle((130.0, 88.0), 9.0, 54.0, (0.78, 0.42, 0.45, 0.36)),
            ),
            accent_rgb=(0.55, 0.40, 0.72),
        ),
        "high_low_slalom": EnvironmentSpec(
            key="high_low_slalom",
            name="High-Low Slalom",
            description="Alternating obstacle heights encourage vertical corrections while the route bends through a slalom.",
            start=(14.0, 12.0, 0.0),
            goal=(136.0, 152.0, 54.0),
            bounds=((0.0, 150.0), (0.0, 170.0), (0.0, 88.0)),
            obstacles=(
                CylinderObstacle((35.0, 34.0), 9.0, 68.0, (0.43, 0.62, 0.74, 0.36)),
                CylinderObstacle((62.0, 48.0), 7.0, 32.0, (0.87, 0.66, 0.33, 0.36)),
                CylinderObstacle((48.0, 78.0), 8.0, 72.0, (0.68, 0.72, 0.48, 0.36)),
                CylinderObstacle((82.0, 92.0), 10.0, 42.0, (0.78, 0.51, 0.44, 0.36)),
                CylinderObstacle((68.0, 124.0), 9.5, 76.0, (0.54, 0.67, 0.83, 0.36)),
                CylinderObstacle((108.0, 128.0), 8.0, 50.0, (0.70, 0.57, 0.80, 0.36)),
                CylinderObstacle((122.0, 102.0), 7.5, 62.0, (0.48, 0.69, 0.58, 0.36)),
            ),
            accent_rgb=(0.32, 0.54, 0.78),
        ),
        "open_field_cluster": EnvironmentSpec(
            key="open_field_cluster",
            name="Open Field Cluster",
            description="A sparse field with one dense central cluster for testing whether the method detours or climbs.",
            start=(8.0, 142.0, 0.0),
            goal=(150.0, 18.0, 44.0),
            bounds=((0.0, 165.0), (0.0, 155.0), (0.0, 74.0)),
            obstacles=(
                CylinderObstacle((64.0, 92.0), 10.0, 48.0, (0.81, 0.59, 0.34, 0.35)),
                CylinderObstacle((80.0, 78.0), 9.0, 56.0, (0.48, 0.66, 0.74, 0.35)),
                CylinderObstacle((96.0, 94.0), 11.0, 44.0, (0.65, 0.73, 0.50, 0.35)),
                CylinderObstacle((82.0, 111.0), 8.5, 62.0, (0.71, 0.55, 0.78, 0.35)),
                CylinderObstacle((122.0, 46.0), 7.5, 36.0, (0.78, 0.44, 0.43, 0.35)),
                CylinderObstacle((36.0, 118.0), 8.0, 40.0, (0.40, 0.65, 0.58, 0.35)),
            ),
            accent_rgb=(0.64, 0.47, 0.22),
        ),
        "custom_sandbox": EnvironmentSpec(
            key="custom_sandbox",
            name="Custom Sandbox",
            description="Editable environment for manually placing 3D cylinder and polygon-prism obstacles.",
            start=(10.0, 10.0, 0.0),
            goal=(140.0, 140.0, 48.0),
            bounds=((0.0, 155.0), (0.0, 155.0), (0.0, 80.0)),
            obstacles=(
                CylinderObstacle((45.0, 55.0), 10.0, 45.0, (0.44, 0.63, 0.76, 0.35)),
                PolygonPrismObstacle.rectangle(
                    center_xy=(82.0, 82.0),
                    width=20.0,
                    length=30.0,
                    height=58.0,
                    rotation_deg=20.0,
                    color_rgba=(0.89, 0.66, 0.31, 0.35),
                ),
                PolygonPrismObstacle.rectangle(
                    center_xy=(112.0, 105.0),
                    width=18.0,
                    length=18.0,
                    height=42.0,
                    rotation_deg=-15.0,
                    color_rgba=(0.53, 0.70, 0.59, 0.35),
                ),
            ),
            accent_rgb=(0.22, 0.58, 0.62),
        ),
    }


def list_environments() -> List[EnvironmentSpec]:
    return list(build_environment_library().values())
