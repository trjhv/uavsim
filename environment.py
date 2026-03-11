"""
environment.py — 3D Environment with Obstacles & No-Fly Zones
=============================================================
Features:
  - Cylinder obstacles (buildings / towers)
  - Box obstacles (warehouses / structures)
  - No-Fly Zones (restricted airspace cylinders)
  - Per-point collision queries with configurable safety margin
  - Path-segment collision check via parametric sampling
  - Nearest obstacle distance query for LiDAR / alerts
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Obstacle primitives
# ---------------------------------------------------------------------------

@dataclass
class Obstacle:
    """Generic 3-D obstacle."""
    center: np.ndarray          # (x, y, 0) — base center on ground
    radius: float               # half-width for box; radius for cylinder
    height: float               # full height in metres
    obstacle_type: str          # 'cylinder' | 'box'
    color: str = "#E74C3C"
    label: str = ""

    # ------------------------------------------------------------------ #

    def check_collision(self, point: np.ndarray, safety_margin: float = 1.5) -> bool:
        """Return True if `point` is inside this obstacle + safety margin."""
        if self.obstacle_type == "cylinder":
            dx = point[0] - self.center[0]
            dy = point[1] - self.center[1]
            horiz = np.sqrt(dx * dx + dy * dy)
            in_height = 0.0 <= point[2] <= (self.height + safety_margin)
            return horiz < (self.radius + safety_margin) and in_height

        else:  # box
            hs = self.radius + safety_margin
            in_x = abs(point[0] - self.center[0]) < hs
            in_y = abs(point[1] - self.center[1]) < hs
            in_z = 0.0 <= point[2] <= (self.height + safety_margin)
            return in_x and in_y and in_z

    def surface_distance(self, point: np.ndarray) -> float:
        """Approximate distance from point to nearest surface of this obstacle."""
        if self.obstacle_type == "cylinder":
            dx = point[0] - self.center[0]
            dy = point[1] - self.center[1]
            horiz = max(0.0, np.sqrt(dx * dx + dy * dy) - self.radius)
            vert = max(0.0, point[2] - self.height) if point[2] > self.height else 0.0
            below = max(0.0, -point[2]) if point[2] < 0 else 0.0
            return float(np.sqrt(horiz**2 + vert**2 + below**2))

        else:  # box
            dx = max(0.0, abs(point[0] - self.center[0]) - self.radius)
            dy = max(0.0, abs(point[1] - self.center[1]) - self.radius)
            dz = max(0.0, point[2] - self.height) if point[2] > self.height else 0.0
            return float(np.sqrt(dx**2 + dy**2 + dz**2))


@dataclass
class NoFlyZone:
    """Restricted airspace — cylindrical column, altitude band."""
    center: np.ndarray
    radius: float
    z_min: float = 0.0
    z_max: float = 50.0
    color: str = "#E74C3C"
    label: str = "NFZ"

    def contains(self, point: np.ndarray) -> bool:
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        horiz = np.sqrt(dx * dx + dy * dy)
        return horiz < self.radius and self.z_min <= point[2] <= self.z_max


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class Environment:
    """
    3-D environment container.

    Attributes:
        bounds  : (x_max, y_max, z_max) in metres
        obstacles : list of Obstacle
        no_fly_zones : list of NoFlyZone
    """

    def __init__(self, bounds: Tuple[float, float, float] = (60.0, 60.0, 35.0)):
        self.bounds = bounds
        self.obstacles: List[Obstacle] = []
        self.no_fly_zones: List[NoFlyZone] = []

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def generate_urban_environment(
        self,
        n_cylinders: int = 6,
        n_boxes: int = 4,
        n_nfz: int = 2,
        seed: int = 42,
    ):
        """Generate a realistic urban obstacle field."""
        rng = np.random.RandomState(seed)
        self.obstacles = []
        self.no_fly_zones = []

        bx, by, bz = self.bounds

        # Keep a 10-m clear zone around start (0,0) and goal (bx,by)
        def too_close_to_terminals(cx, cy, r):
            d_start = np.sqrt(cx**2 + cy**2)
            d_goal = np.sqrt((cx - bx)**2 + (cy - by)**2)
            return d_start < 10.0 + r or d_goal < 10.0 + r

        # --- Cylinder buildings ---
        cylinder_colors = ["#C0392B", "#E74C3C", "#922B21", "#D35400", "#E67E22"]
        for i in range(n_cylinders):
            for _ in range(200):
                cx = rng.uniform(8, bx - 8)
                cy = rng.uniform(8, by - 8)
                r = rng.uniform(2.0, 5.0)
                h = rng.uniform(8.0, min(bz * 0.85, 28.0))
                if not too_close_to_terminals(cx, cy, r) and not self._overlaps(cx, cy, r):
                    self.obstacles.append(Obstacle(
                        center=np.array([cx, cy, 0.0]),
                        radius=r, height=h,
                        obstacle_type="cylinder",
                        color=cylinder_colors[i % len(cylinder_colors)],
                        label=f"Tower-{i+1}",
                    ))
                    break

        # --- Box buildings ---
        box_colors = ["#784212", "#935116", "#A04000", "#7D6608"]
        for i in range(n_boxes):
            for _ in range(200):
                cx = rng.uniform(8, bx - 8)
                cy = rng.uniform(8, by - 8)
                r = rng.uniform(2.5, 6.0)
                h = rng.uniform(10.0, min(bz * 0.75, 25.0))
                if not too_close_to_terminals(cx, cy, r) and not self._overlaps(cx, cy, r):
                    self.obstacles.append(Obstacle(
                        center=np.array([cx, cy, 0.0]),
                        radius=r, height=h,
                        obstacle_type="box",
                        color=box_colors[i % len(box_colors)],
                        label=f"Block-{i+1}",
                    ))
                    break

        # --- No-Fly Zones ---
        for i in range(n_nfz):
            for _ in range(100):
                cx = rng.uniform(10, bx - 10)
                cy = rng.uniform(10, by - 10)
                r = rng.uniform(5.0, 10.0)
                z_min = rng.uniform(0, 8)
                z_max = rng.uniform(12, bz)
                if not too_close_to_terminals(cx, cy, r + 5):
                    self.no_fly_zones.append(NoFlyZone(
                        center=np.array([cx, cy, 0.0]),
                        radius=r, z_min=z_min, z_max=z_max,
                        label=f"NFZ-{i+1}",
                    ))
                    break

    def _overlaps(self, cx, cy, r, min_gap=2.0) -> bool:
        """Check if a proposed obstacle overlaps existing ones."""
        for obs in self.obstacles:
            dx = cx - obs.center[0]
            dy = cy - obs.center[1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < (r + obs.radius + min_gap):
                return True
        return False

    # ------------------------------------------------------------------ #
    # Collision queries
    # ------------------------------------------------------------------ #

    def is_point_in_collision(self, point: np.ndarray, safety_margin: float = 1.5) -> bool:
        """Check point against all obstacles + bounds."""
        # World bounds
        if not (0 <= point[0] <= self.bounds[0] and
                0 <= point[1] <= self.bounds[1] and
                0 <= point[2] <= self.bounds[2]):
            return True
        for obs in self.obstacles:
            if obs.check_collision(point, safety_margin):
                return True
        return False

    def check_segment_collision(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        n_samples: int = 20,
        safety_margin: float = 1.5,
    ) -> bool:
        """Parametric collision check along a line segment."""
        for t in np.linspace(0, 1, n_samples):
            pt = p1 + t * (p2 - p1)
            if self.is_point_in_collision(pt, safety_margin):
                return True
        return False

    def nearest_obstacle_info(self, point: np.ndarray) -> Tuple[float, Optional[Obstacle]]:
        """Return (distance, obstacle) for the closest obstacle."""
        min_dist = float("inf")
        nearest: Optional[Obstacle] = None
        for obs in self.obstacles:
            d = obs.surface_distance(point)
            if d < min_dist:
                min_dist = d
                nearest = obs
        return min_dist, nearest

    def in_no_fly_zone(self, point: np.ndarray) -> bool:
        for nfz in self.no_fly_zones:
            if nfz.contains(point):
                return True
        return False
