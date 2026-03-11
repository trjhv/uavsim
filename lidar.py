"""
lidar.py — Rotating LiDAR Sensor Simulation
============================================
Simulates a spinning LiDAR sensor (Velodyne VLP-16 style):
  - Configurable horizontal/vertical angular resolution
  - Ray-marching collision detection against environment obstacles
  - Gaussian range noise model
  - Returns hit-point cloud + per-point intensity (1/distance)
  - Optionally returns rays for visualising sensor beams
"""

import numpy as np
from typing import Optional, Tuple

from simulation.environment import Environment


class LiDARSensor:
    """
    Simulated rotating 3-D LiDAR.

    Args:
        range_max     : Maximum detection range (metres).
        n_horizontal  : Number of horizontal rays (angular resolution = 360°/n).
        n_vertical    : Number of vertical scan lines (elevation channels).
        fov_vertical  : Vertical field of view in degrees (±half above/below).
        noise_std     : Gaussian range noise standard deviation (metres).
        step_size     : Ray-march step size (metres). Smaller = more accurate, slower.
    """

    def __init__(
        self,
        range_max: float = 20.0,
        n_horizontal: int = 72,
        n_vertical: int = 16,
        fov_vertical: float = 30.0,
        noise_std: float = 0.05,
        step_size: float = 0.4,
    ):
        self.range_max = range_max
        self.n_h = n_horizontal
        self.n_v = n_vertical
        self.fov_v_rad = np.deg2rad(fov_vertical / 2)
        self.noise_std = noise_std
        self.step_size = step_size

        # Pre-compute direction arrays
        self._h_angles = np.linspace(0, 2 * np.pi, n_horizontal, endpoint=False)
        self._v_angles = np.linspace(-self.fov_v_rad, self.fov_v_rad, n_vertical)

    # ------------------------------------------------------------------ #

    def scan(
        self,
        position: np.ndarray,
        yaw: float,
        environment: Environment,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one full scan from `position` at heading `yaw`.

        Returns:
            points      : (N, 3) array of hit positions in world frame.
            intensities : (N,) array of values in [0, 1] (brighter = closer).
        """
        if rng is None:
            rng = np.random.RandomState()

        hit_pts = []
        hit_int = []

        for v_ang in self._v_angles:
            cos_v = np.cos(v_ang)
            sin_v = np.sin(v_ang)

            for h_ang in self._h_angles:
                # Ray direction in world frame (rotated by yaw)
                total_h = h_ang + yaw
                dx = cos_v * np.cos(total_h)
                dy = cos_v * np.sin(total_h)
                dz = sin_v
                direction = np.array([dx, dy, dz])

                dist = self._march(position, direction, environment)

                if dist < self.range_max * 0.97:
                    # Add range noise
                    dist_n = float(np.clip(
                        dist + rng.normal(0, self.noise_std),
                        0, self.range_max,
                    ))
                    hit_pts.append(position + dist_n * direction)
                    hit_int.append(1.0 - dist_n / self.range_max)

        if hit_pts:
            return np.array(hit_pts, dtype=np.float32), np.array(hit_int, dtype=np.float32)
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.float32)

    # ------------------------------------------------------------------ #

    def _march(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        environment: Environment,
    ) -> float:
        """Ray-march along direction until obstacle hit or max range."""
        t = self.step_size
        while t <= self.range_max:
            point = origin + t * direction
            # Ground plane
            if point[2] < 0:
                return t
            for obs in environment.obstacles:
                if obs.check_collision(point, safety_margin=0.0):
                    return t
            t += self.step_size
        return self.range_max
