"""
visualization.py — Plotly 3-D Scene Builder
===========================================
Builds the complete 3-D Plotly figure for the UAV simulation, including:
  - Ground plane grid
  - Cylinder and box obstacle meshes
  - No-fly zone indicators
  - Planned A* path (dashed line)
  - Drone flight trail (coloured by speed)
  - Drone body (diamond marker + motor arm lines)
  - LiDAR point cloud (coloured by intensity)
  - Start / Goal markers
"""

import numpy as np
import plotly.graph_objects as go
from typing import List, Optional, Tuple

from simulation.environment import Environment, Obstacle
from simulation.drone import DroneState


# ---------------------------------------------------------------------------
# Colour constants
# ---------------------------------------------------------------------------

DARK_BG = "#060612"
GRID_COLOR = "#0e0e2c"
PATH_COLOR = "#00D4FF"
TRAIL_COLOR_LOW = "#F39C12"
TRAIL_COLOR_HIGH = "#E74C3C"
DRONE_COLOR = "#A855F7"
START_COLOR = "#2ECC71"
GOAL_COLOR = "#E74C3C"


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def _cylinder_mesh(center, radius, height, color, n=20) -> go.Mesh3d:
    theta = np.linspace(0, 2 * np.pi, n + 1)
    # Bottom ring
    xb = center[0] + radius * np.cos(theta)
    yb = center[1] + radius * np.sin(theta)
    zb = np.zeros(n + 1)
    # Top ring
    xt = center[0] + radius * np.cos(theta)
    yt = center[1] + radius * np.sin(theta)
    zt = np.full(n + 1, height)

    x = np.concatenate([xb, xt])
    y = np.concatenate([yb, yt])
    z = np.concatenate([zb, zt])

    # Side triangles
    i_idx, j_idx, k_idx = [], [], []
    for s in range(n):
        # Two triangles per quad
        i_idx += [s, s]
        j_idx += [s + 1, n + 1 + s]
        k_idx += [n + 1 + s, n + 1 + s + 1]

    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i_idx, j=j_idx, k=k_idx,
        color=color, opacity=0.70,
        showlegend=False,
        flatshading=True,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.6),
        lightposition=dict(x=100, y=200, z=300),
    )


def _box_mesh(center, half_size, height, color) -> go.Mesh3d:
    cx, cy = center[0], center[1]
    hs = half_size
    # 8 vertices
    x = [cx-hs, cx+hs, cx+hs, cx-hs, cx-hs, cx+hs, cx+hs, cx-hs]
    y = [cy-hs, cy-hs, cy+hs, cy+hs, cy-hs, cy-hs, cy+hs, cy+hs]
    z = [0, 0, 0, 0, height, height, height, height]

    # 12 triangles (2 per face × 6 faces)
    i_ = [0, 0, 0, 0, 4, 4, 1, 1, 2, 2, 3, 3]
    j_ = [1, 2, 4, 5, 5, 6, 2, 5, 3, 6, 0, 7]
    k_ = [2, 3, 5, 6, 6, 7, 5, 6, 6, 7, 7, 4]

    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i_, j=j_, k=k_,
        color=color, opacity=0.72,
        showlegend=False,
        flatshading=True,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2, roughness=0.7),
        lightposition=dict(x=100, y=200, z=300),
    )


def _nfz_cylinder(center, radius, z_min, z_max, color="#FF0040") -> go.Mesh3d:
    """Transparent No-Fly Zone cylinder."""
    n = 24
    theta = np.linspace(0, 2 * np.pi, n + 1)
    xb = center[0] + radius * np.cos(theta)
    yb = center[1] + radius * np.sin(theta)
    zb = np.full(n + 1, z_min)
    xt = center[0] + radius * np.cos(theta)
    yt = center[1] + radius * np.sin(theta)
    zt = np.full(n + 1, z_max)

    x = np.concatenate([xb, xt])
    y = np.concatenate([yb, yt])
    z = np.concatenate([zb, zt])

    i_idx, j_idx, k_idx = [], [], []
    for s in range(n):
        i_idx += [s, s]
        j_idx += [s + 1, n + 1 + s]
        k_idx += [n + 1 + s, n + 1 + s + 1]

    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i_idx, j=j_idx, k=k_idx,
        color=color, opacity=0.15,
        showlegend=False,
        flatshading=False,
    )


# ---------------------------------------------------------------------------
# Main scene builder
# ---------------------------------------------------------------------------

def build_3d_scene(
    environment: Environment,
    path: Optional[List[np.ndarray]] = None,
    drone_position: Optional[np.ndarray] = None,
    drone_yaw: float = 0.0,
    lidar_points: Optional[np.ndarray] = None,
    lidar_intensities: Optional[np.ndarray] = None,
    trajectory_history: Optional[List[np.ndarray]] = None,
    trajectory_speeds: Optional[List[float]] = None,
    show_lidar: bool = True,
    show_nfz: bool = True,
    arm_length: float = 1.8,
    camera_eye: Optional[dict] = None,
) -> go.Figure:
    """
    Assemble the complete 3-D Plotly scene.

    Args:
        environment       : Environment instance with obstacles / NFZs.
        path              : List of planned waypoint arrays.
        drone_position    : Current drone position (3-vector).
        drone_yaw         : Current yaw in radians.
        lidar_points      : (N,3) world-frame LiDAR hit points.
        lidar_intensities : (N,) intensity values [0,1].
        trajectory_history: Past drone positions for trail.
        trajectory_speeds : Speed at each trail point (for colour coding).
        show_lidar        : Toggle LiDAR point cloud.
        show_nfz          : Toggle No-Fly Zone cylinders.
        arm_length        : Drone arm length for visualisation.
        camera_eye        : Plotly camera eye dict.
    """
    fig = go.Figure()

    bx, by, bz = environment.bounds

    # ------------------------------------------------------------------
    # Ground plane (two triangles)
    # ------------------------------------------------------------------
    fig.add_trace(go.Mesh3d(
        x=[0, bx, bx, 0],
        y=[0, 0, by, by],
        z=[0, 0, 0, 0],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="#0B3D0B", opacity=0.25,
        showlegend=False,
        hoverinfo="skip",
    ))

    # ------------------------------------------------------------------
    # Obstacles
    # ------------------------------------------------------------------
    for obs in environment.obstacles:
        if obs.obstacle_type == "cylinder":
            fig.add_trace(_cylinder_mesh(obs.center, obs.radius, obs.height, obs.color))
        else:
            fig.add_trace(_box_mesh(obs.center, obs.radius, obs.height, obs.color))

    # ------------------------------------------------------------------
    # No-Fly Zones
    # ------------------------------------------------------------------
    if show_nfz:
        for nfz in environment.no_fly_zones:
            fig.add_trace(_nfz_cylinder(
                nfz.center, nfz.radius, nfz.z_min, nfz.z_max
            ))
            # NFZ label
            fig.add_trace(go.Scatter3d(
                x=[nfz.center[0]], y=[nfz.center[1]], z=[nfz.z_max + 1],
                mode="text",
                text=[f"⛔ {nfz.label}"],
                textfont=dict(color="#FF4060", size=11),
                showlegend=False, hoverinfo="skip",
            ))

    # ------------------------------------------------------------------
    # Planned Path
    # ------------------------------------------------------------------
    if path and len(path) > 1:
        pa = np.array(path)
        fig.add_trace(go.Scatter3d(
            x=pa[:, 0], y=pa[:, 1], z=pa[:, 2],
            mode="lines",
            line=dict(color=PATH_COLOR, width=3, dash="dash"),
            name="Planned Path",
            hoverinfo="skip",
        ))

    # ------------------------------------------------------------------
    # Flight Trail (speed colour coding)
    # ------------------------------------------------------------------
    if trajectory_history and len(trajectory_history) > 1:
        ta = np.array(trajectory_history)
        speeds = trajectory_speeds if trajectory_speeds else [0.0] * len(ta)

        fig.add_trace(go.Scatter3d(
            x=ta[:, 0], y=ta[:, 1], z=ta[:, 2],
            mode="lines+markers",
            line=dict(
                color=speeds,
                colorscale="Inferno",
                width=5,
                cmin=0, cmax=8,
            ),
            marker=dict(
                size=2,
                color=speeds,
                colorscale="Inferno",
                cmin=0, cmax=8,
                showscale=True,
                colorbar=dict(
                    title="Speed (m/s)",
                    x=1.02, len=0.5, thickness=12,
                    tickfont=dict(color="white"),
                    titlefont=dict(color="white"),
                ),
            ),
            name="Flight Trail",
        ))

    # ------------------------------------------------------------------
    # LiDAR Point Cloud
    # ------------------------------------------------------------------
    if show_lidar and lidar_points is not None and len(lidar_points) > 0:
        fig.add_trace(go.Scatter3d(
            x=lidar_points[:, 0], y=lidar_points[:, 1], z=lidar_points[:, 2],
            mode="markers",
            marker=dict(
                size=1.5,
                color=lidar_intensities,
                colorscale="Plasma",
                opacity=0.55,
                showscale=True,
                cmin=0, cmax=1,
                colorbar=dict(
                    title="LiDAR<br>Intensity",
                    x=1.12, len=0.45, thickness=12,
                    tickfont=dict(color="white"),
                    titlefont=dict(color="white"),
                ),
            ),
            name="LiDAR Returns",
            hoverinfo="skip",
        ))

    # ------------------------------------------------------------------
    # Drone Body
    # ------------------------------------------------------------------
    if drone_position is not None:
        dx, dy, dz = drone_position

        # Central body marker
        fig.add_trace(go.Scatter3d(
            x=[dx], y=[dy], z=[dz],
            mode="markers",
            marker=dict(size=10, color=DRONE_COLOR, symbol="diamond",
                        line=dict(color="white", width=1)),
            name="Drone",
            hovertemplate=(
                f"<b>Drone</b><br>X: {dx:.1f} m<br>Y: {dy:.1f} m<br>Alt: {dz:.1f} m"
                "<extra></extra>"
            ),
        ))

        # Motor arms in X configuration (rotated by yaw)
        for angle_offset in (np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4):
            arm_angle = drone_yaw + angle_offset
            ax = dx + arm_length * np.cos(arm_angle)
            ay = dy + arm_length * np.sin(arm_angle)
            # Arm line
            fig.add_trace(go.Scatter3d(
                x=[dx, ax], y=[dy, ay], z=[dz, dz],
                mode="lines",
                line=dict(color="#CBD5E1", width=4),
                showlegend=False, hoverinfo="skip",
            ))
            # Motor disc
            fig.add_trace(go.Scatter3d(
                x=[ax], y=[ay], z=[dz],
                mode="markers",
                marker=dict(size=6, color="#F59E0B",
                            line=dict(color="#FCD34D", width=1)),
                showlegend=False, hoverinfo="skip",
            ))

        # Altitude line to ground
        fig.add_trace(go.Scatter3d(
            x=[dx, dx], y=[dy, dy], z=[0, dz],
            mode="lines",
            line=dict(color=DRONE_COLOR, width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))

    # ------------------------------------------------------------------
    # Start & Goal Markers
    # ------------------------------------------------------------------
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[2],
        mode="markers+text",
        text=["START"],
        textfont=dict(color=START_COLOR, size=12),
        textposition="top center",
        marker=dict(size=14, color=START_COLOR, symbol="square",
                    line=dict(color="white", width=1)),
        name="Start",
    ))
    fig.add_trace(go.Scatter3d(
        x=[bx], y=[by], z=[2],
        mode="markers+text",
        text=["GOAL"],
        textfont=dict(color=GOAL_COLOR, size=12),
        textposition="top center",
        marker=dict(size=14, color=GOAL_COLOR, symbol="square",
                    line=dict(color="white", width=1)),
        name="Goal",
    ))

    # ------------------------------------------------------------------
    # Layout & Camera
    # ------------------------------------------------------------------
    cam = camera_eye or dict(x=1.6, y=-1.6, z=1.0)

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="X (m)", range=[0, bx],
                backgroundcolor=DARK_BG, gridcolor=GRID_COLOR,
                showbackground=True, tickfont=dict(color="#94A3B8"),
                titlefont=dict(color="#94A3B8"),
            ),
            yaxis=dict(
                title="Y (m)", range=[0, by],
                backgroundcolor=DARK_BG, gridcolor=GRID_COLOR,
                showbackground=True, tickfont=dict(color="#94A3B8"),
                titlefont=dict(color="#94A3B8"),
            ),
            zaxis=dict(
                title="Alt (m)", range=[0, bz],
                backgroundcolor=DARK_BG, gridcolor=GRID_COLOR,
                showbackground=True, tickfont=dict(color="#94A3B8"),
                titlefont=dict(color="#94A3B8"),
            ),
            bgcolor=DARK_BG,
            camera=dict(eye=cam),
            aspectmode="manual",
            aspectratio=dict(x=bx / bz, y=by / bz, z=1),
        ),
        paper_bgcolor="#080816",
        plot_bgcolor="#080816",
        font=dict(color="white", family="'JetBrains Mono', 'Courier New', monospace"),
        height=660,
        margin=dict(l=0, r=60, t=10, b=0),
        legend=dict(
            font=dict(color="white", size=11),
            bgcolor="rgba(8,8,22,0.85)",
            bordercolor="#1E40AF",
            borderwidth=1,
            x=0.01, y=0.99,
        ),
        uirevision="stable",   # Keeps camera angle on re-render
    )

    return fig
