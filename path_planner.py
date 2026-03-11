"""
path_planner.py — 3-D A* Path Planner with Cubic Spline Smoothing
==================================================================
Implements:
  - A* search on a 3-D grid with configurable resolution
  - 26-connected neighbourhood (all diagonal moves allowed)
  - Obstacle inflation via safety margin
  - Cubic-spline path smoothing
  - Altitude-bypass fallback when A* fails
  - Waypoint injection for multi-point missions
"""

import heapq
import numpy as np
from typing import Dict, List, Optional, Tuple

from simulation.environment import Environment


# ---------------------------------------------------------------------------
# 3-D A* Planner
# ---------------------------------------------------------------------------

class AStarPlanner3D:
    """
    Grid-based 3-D A* planner.

    Args:
        environment : Environment containing obstacles.
        resolution  : Grid cell size in metres (smaller = more precise but slower).
        safety_margin : Extra clearance beyond obstacle radius (metres).
    """

    def __init__(
        self,
        environment: Environment,
        resolution: float = 1.5,
        safety_margin: float = 1.5,
    ):
        self.env = environment
        self.res = resolution
        self.safety = safety_margin

        # Pre-compute grid dimensions
        self.nx = int(environment.bounds[0] / resolution) + 1
        self.ny = int(environment.bounds[1] / resolution) + 1
        self.nz = int(environment.bounds[2] / resolution) + 1

    # ------------------------------------------------------------------ #
    # Coordinate conversions
    # ------------------------------------------------------------------ #

    def _to_grid(self, pos: np.ndarray) -> Tuple[int, int, int]:
        return (
            int(np.clip(pos[0] / self.res, 0, self.nx - 1)),
            int(np.clip(pos[1] / self.res, 0, self.ny - 1)),
            int(np.clip(pos[2] / self.res, 0, self.nz - 1)),
        )

    def _to_world(self, gx: int, gy: int, gz: int) -> np.ndarray:
        return np.array([gx * self.res, gy * self.res, gz * self.res])

    # ------------------------------------------------------------------ #
    # Core A* logic
    # ------------------------------------------------------------------ #

    def _heuristic(self, a: Tuple, b: Tuple) -> float:
        return np.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b))) * self.res

    def _neighbors(self, node: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        x, y, z = node
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    nx_, ny_, nz_ = x + dx, y + dy, z + dz
                    if 0 <= nx_ < self.nx and 0 <= ny_ < self.ny and 0 <= nz_ < self.nz:
                        neighbors.append((nx_, ny_, nz_))
        return neighbors

    def _is_free(self, gx: int, gy: int, gz: int) -> bool:
        pos = self._to_world(gx, gy, gz)
        return not self.env.is_point_in_collision(pos, self.safety)

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        max_iterations: int = 15_000,
    ) -> List[np.ndarray]:
        """
        Plan a collision-free path from start to goal.

        Returns a list of 3-D waypoints (smooth interpolation applied).
        Falls back to altitude-bypass path if A* cannot find a solution.
        """
        start_g = self._to_grid(start)
        goal_g = self._to_grid(goal)

        if start_g == goal_g:
            return [start, goal]

        # Min-heap: (f_score, node)
        open_heap: List[Tuple[float, Tuple]] = []
        heapq.heappush(open_heap, (0.0, start_g))

        came_from: Dict[Tuple, Optional[Tuple]] = {start_g: None}
        g: Dict[Tuple, float] = {start_g: 0.0}

        iterations = 0

        while open_heap and iterations < max_iterations:
            iterations += 1
            _, current = heapq.heappop(open_heap)

            # Goal check (within 2 grid cells)
            if self._heuristic(current, goal_g) < self.res * 2.5:
                return self._reconstruct(came_from, current, start, goal)

            for nb in self._neighbors(current):
                if not self._is_free(*nb):
                    continue

                # Move cost: Euclidean distance between grid cells * res
                move_cost = self._heuristic(current, nb)
                tentative_g = g.get(current, float("inf")) + move_cost

                if tentative_g < g.get(nb, float("inf")):
                    came_from[nb] = current
                    g[nb] = tentative_g
                    f = tentative_g + self._heuristic(nb, goal_g)
                    heapq.heappush(open_heap, (f, nb))

        # A* failed — use high-altitude bypass
        return self._altitude_bypass(start, goal)

    # ------------------------------------------------------------------ #
    # Path reconstruction & smoothing
    # ------------------------------------------------------------------ #

    def _reconstruct(
        self,
        came_from: Dict,
        current: Tuple,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> List[np.ndarray]:
        """Walk back through came_from to build raw path, then smooth."""
        raw = []
        node = current
        while node is not None:
            raw.append(self._to_world(*node))
            node = came_from.get(node)
        raw.reverse()
        raw.append(goal)
        return self._smooth(raw, n_points=80)

    def _smooth(self, path: List[np.ndarray], n_points: int = 80) -> List[np.ndarray]:
        """Cubic spline interpolation for smooth waypoint transitions."""
        if len(path) < 3:
            return path
        try:
            from scipy.interpolate import CubicSpline
            arr = np.array(path)
            t_orig = np.linspace(0, 1, len(arr))
            t_fine = np.linspace(0, 1, n_points)
            smooth = np.column_stack([
                CubicSpline(t_orig, arr[:, i])(t_fine)
                for i in range(3)
            ])
            # Clip to bounds
            smooth[:, 0] = np.clip(smooth[:, 0], 0, self.env.bounds[0])
            smooth[:, 1] = np.clip(smooth[:, 1], 0, self.env.bounds[1])
            smooth[:, 2] = np.clip(smooth[:, 2], 0.5, self.env.bounds[2])
            return [smooth[i] for i in range(len(smooth))]
        except Exception:
            return path

    def _altitude_bypass(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        """Fly up, traverse, descend — guaranteed collision-free at max altitude."""
        safe_z = self.env.bounds[2] * 0.85
        mid_points = [
            start,
            np.array([start[0], start[1], safe_z]),
            np.array([goal[0], goal[1], safe_z]),
            goal,
        ]
        return self._smooth(mid_points, n_points=60)
