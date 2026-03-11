"""
drone.py — UAV Physics Engine
================================
Implements a simplified 6-DOF quadrotor model with:
  - PID-like proportional velocity control toward waypoints
  - Aerodynamic drag
  - Gravity compensation
  - Battery consumption model
  - Roll/Pitch/Yaw attitude estimation from velocity
  - Wind disturbance injection
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class DroneState:
    """Snapshot of the drone's full state at a single timestep."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    roll: float = 0.0          # rad
    pitch: float = 0.0         # rad
    yaw: float = 0.0           # rad
    battery: float = 100.0     # %
    timestamp: float = 0.0     # s
    thrust: float = 0.0        # N (total)
    motor_rpm: float = 0.0     # approximate RPM

    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    def altitude(self) -> float:
        return float(self.position[2])

    def copy(self) -> "DroneState":
        return DroneState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            acceleration=self.acceleration.copy(),
            roll=self.roll,
            pitch=self.pitch,
            yaw=self.yaw,
            battery=self.battery,
            timestamp=self.timestamp,
            thrust=self.thrust,
            motor_rpm=self.motor_rpm,
        )


# ---------------------------------------------------------------------------
# Physics Engine
# ---------------------------------------------------------------------------

class DronePhysics:
    """
    Simplified quadrotor dynamics.

    Control law:
        desired_velocity  = K_p * (target - position)  [clipped to max_speed]
        thrust_accel      = K_v * (desired_vel - current_vel)
        drag_accel        = -C_d * v
        net_accel         = thrust_accel + drag_accel
    """

    G = 9.81  # m/s²

    def __init__(
        self,
        mass: float = 1.5,           # kg
        max_speed: float = 8.0,      # m/s
        max_thrust: float = 35.0,    # N
        drag_coeff: float = 0.12,
        motor_time_const: float = 0.1,
        battery_capacity: float = 5000.0,  # mAh (conceptual)
    ):
        self.mass = mass
        self.max_speed = max_speed
        self.max_thrust = max_thrust
        self.drag_coeff = drag_coeff
        self.motor_tc = motor_time_const

        self.state = DroneState()
        self.trajectory: List[np.ndarray] = [self.state.position.copy()]

        # Battery model: full charge → max_capacity mAh
        self._battery_mah = 5000.0
        self._battery_max_mah = 5000.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, start_position: Optional[np.ndarray] = None):
        self.state = DroneState()
        if start_position is not None:
            self.state.position = start_position.copy()
        self.trajectory = [self.state.position.copy()]
        self._battery_mah = self._battery_max_mah

    def step(
        self,
        target: np.ndarray,
        dt: float = 0.05,
        wind: Optional[np.ndarray] = None,
    ) -> DroneState:
        """Advance simulation by one timestep toward `target`."""

        # --- Proportional position controller ---
        error = target - self.state.position
        dist = np.linalg.norm(error)

        if dist < 0.02:
            # At waypoint: coast to a stop
            self.state.velocity *= max(0.0, 1.0 - 8.0 * dt)
            self.state.acceleration = np.zeros(3)
        else:
            direction = error / dist
            desired_speed = min(dist * 2.5, self.max_speed)
            desired_vel = direction * desired_speed

            # --- Velocity controller ---
            vel_err = desired_vel - self.state.velocity
            thrust_accel = vel_err * 4.0                       # Kv = 4
            drag_accel = -self.drag_coeff * self.state.velocity

            # Wind disturbance
            wind_accel = (wind / self.mass) if wind is not None else np.zeros(3)

            net_accel = thrust_accel + drag_accel + wind_accel

            # Clip to physical limits
            max_accel = self.max_thrust / self.mass
            accel_mag = np.linalg.norm(net_accel)
            if accel_mag > max_accel:
                net_accel = net_accel / accel_mag * max_accel

            self.state.acceleration = net_accel
            self.state.velocity += net_accel * dt

            # Clip velocity
            v_mag = np.linalg.norm(self.state.velocity)
            if v_mag > self.max_speed:
                self.state.velocity = self.state.velocity / v_mag * self.max_speed

        # --- Integrate position ---
        self.state.position = self.state.position + self.state.velocity * dt
        self.state.timestamp += dt

        # --- Attitude estimation ---
        self._update_attitude()

        # --- Battery model ---
        speed = self.state.speed()
        # Hover cost + propulsion cost
        idle_drain = 0.8   # mAh/s at hover
        prop_drain = 0.5 * speed  # additional mAh/s
        self._battery_mah -= (idle_drain + prop_drain) * dt
        self._battery_mah = max(0.0, self._battery_mah)
        self.state.battery = (self._battery_mah / self._battery_max_mah) * 100.0

        # --- Motor RPM (approximate) ---
        thrust = self.mass * self.G + self.mass * np.linalg.norm(self.state.acceleration)
        self.state.thrust = float(np.clip(thrust, 0, self.max_thrust))
        self.state.motor_rpm = float(5000 + (self.state.thrust / self.max_thrust) * 7000)

        self.trajectory.append(self.state.position.copy())
        return self.state.copy()

    def simulate_mission(
        self,
        waypoints: List[np.ndarray],
        dt: float = 0.05,
        wind_fn=None,
        arrival_radius: float = 0.8,
        max_steps_per_wp: int = 600,
    ) -> List[DroneState]:
        """
        Run the full mission through all waypoints.

        Args:
            waypoints: List of 3-D target positions.
            dt: Simulation timestep in seconds.
            wind_fn: Optional callable(t) → np.ndarray[3] returning wind force.
            arrival_radius: Distance threshold to declare waypoint reached.
            max_steps_per_wp: Safety limit to prevent infinite loops.

        Returns:
            List of DroneState snapshots (one per simulation step).
        """
        states: List[DroneState] = []

        for wp in waypoints:
            for _ in range(max_steps_per_wp):
                wind = wind_fn(self.state.timestamp) if wind_fn else None
                s = self.step(wp, dt=dt, wind=wind)
                states.append(s)
                if np.linalg.norm(wp - self.state.position) < arrival_radius:
                    break

        return states

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _update_attitude(self):
        """Derive roll/pitch/yaw from velocity vector."""
        v = self.state.velocity
        speed_xy = np.linalg.norm(v[:2])

        if speed_xy > 0.05:
            self.state.yaw = float(np.arctan2(v[1], v[0]))
            self.state.pitch = float(-np.arctan2(v[2], speed_xy))
        # Roll: simplified bank for turns (zero for now)
        self.state.roll = 0.0
