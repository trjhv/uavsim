"""
app.py — UAV 3-D Simulation & Mission Planning Dashboard
=========================================================
A production-grade Streamlit application that provides:

  ┌─ SIDEBAR ──────────────────────────────────────────────────────────┐
  │  • Drone hardware parameters (mass, speed, battery)                │
  │  • Environment generation (size, obstacle density, seed)           │
  │  • Mission endpoints (start/goal altitude, flight altitude)        │
  │  • Sensor configuration (LiDAR toggle, range, resolution)         │
  │  • Physics options (wind model, safety margin, timestep)           │
  │  • Run / Reset simulation controls                                 │
  └────────────────────────────────────────────────────────────────────┘

  ┌─ MAIN TABS ────────────────────────────────────────────────────────┐
  │  Tab 1 — 3D Mission View    : Full Plotly 3-D scene with scrubber  │
  │  Tab 2 — Telemetry          : Altitude, speed, battery time-series │
  │  Tab 3 — Collision Monitor  : Proximity alerts, NFZ violations     │
  │  Tab 4 — Mission Report     : Statistics and flight log            │
  └────────────────────────────────────────────────────────────────────┘

Dependencies: streamlit, plotly, numpy, scipy
"""

import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from simulation.drone import DronePhysics, DroneState
from simulation.environment import Environment
from simulation.path_planner import AStarPlanner3D
from simulation.lidar import LiDARSensor
from simulation.visualization import build_3d_scene


# ===========================================================================
# PAGE CONFIG — Must be first Streamlit call
# ===========================================================================

st.set_page_config(
    page_title="UAV Mission Sim",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===========================================================================
# CUSTOM CSS — Aerospace HUD aesthetic
# ===========================================================================

st.markdown("""
<style>
  /* --- Import monospace font --- */
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

  /* --- Global dark background --- */
  .stApp { background-color: #05050F; color: #CBD5E1; }
  .main .block-container { padding-top: 1rem; padding-bottom: 1rem; }

  /* --- Sidebar styling --- */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A0A1E 0%, #070714 100%);
    border-right: 1px solid #1E3A5F;
  }

  /* --- HUD metric cards --- */
  [data-testid="stMetric"] {
    background: rgba(14, 20, 40, 0.85);
    border: 1px solid #1E40AF;
    border-radius: 6px;
    padding: 8px 12px !important;
    backdrop-filter: blur(4px);
    font-family: 'Share Tech Mono', monospace;
  }
  [data-testid="stMetricLabel"] { color: #64748B !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
  [data-testid="stMetricValue"] { color: #00D4FF !important; font-size: 1.3rem !important; font-family: 'Share Tech Mono', monospace !important; }
  [data-testid="stMetricDelta"] { font-size: 0.72rem !important; }

  /* --- Tabs --- */
  [data-testid="stTabs"] button {
    font-family: 'Exo 2', sans-serif;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: #64748B;
    border-bottom: 2px solid transparent;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
    color: #00D4FF;
    border-bottom: 2px solid #00D4FF;
  }

  /* --- Buttons --- */
  .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #1E3A5F, #0F1F3D);
    border: 1px solid #2563EB;
    color: #00D4FF;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.9rem;
    letter-spacing: 0.06em;
    border-radius: 4px;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #2563EB, #1E3A5F);
    border-color: #00D4FF;
    box-shadow: 0 0 12px rgba(0, 212, 255, 0.35);
  }

  /* --- Sliders --- */
  [data-testid="stSlider"] > div > div { background-color: #1E40AF; }
  [data-testid="stSlider"] > div > div > div { background: #00D4FF; }

  /* --- Section headers in sidebar --- */
  .sidebar-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #00D4FF;
    border-bottom: 1px solid #1E40AF;
    padding-bottom: 4px;
    margin-bottom: 10px;
    margin-top: 16px;
  }

  /* --- Alert boxes --- */
  .alert-danger {
    background: rgba(220, 38, 38, 0.12);
    border-left: 3px solid #DC2626;
    padding: 8px 12px;
    border-radius: 4px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    color: #FCA5A5;
    margin: 4px 0;
  }
  .alert-warning {
    background: rgba(245, 158, 11, 0.12);
    border-left: 3px solid #F59E0B;
    padding: 8px 12px;
    border-radius: 4px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    color: #FDE68A;
    margin: 4px 0;
  }
  .alert-ok {
    background: rgba(16, 185, 129, 0.10);
    border-left: 3px solid #10B981;
    padding: 8px 12px;
    border-radius: 4px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    color: #6EE7B7;
    margin: 4px 0;
  }

  /* --- Status badge --- */
  .status-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }

  /* --- HUD title --- */
  .hud-title {
    font-family: 'Exo 2', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #00D4FF;
    letter-spacing: 0.04em;
    text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
  }
  .hud-subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #475569;
    letter-spacing: 0.08em;
  }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# SESSION STATE INITIALISATION
# ===========================================================================

def init_session():
    defaults = dict(
        sim_run=False,
        states=[],
        path=[],
        environment=None,
        frame_idx=0,
        lidar_cache={},    # frame_idx → (points, intensities)
        plan_time=0.0,
        sim_time=0.0,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ===========================================================================
# WIND MODEL
# ===========================================================================

def make_wind_fn(intensity: float, direction_deg: float):
    """Return a callable wind_fn(t) → force vector."""
    if intensity == 0:
        return None
    dir_rad = np.deg2rad(direction_deg)
    base = np.array([
        intensity * np.cos(dir_rad),
        intensity * np.sin(dir_rad),
        0.0,
    ])
    def wind_fn(t):
        # Add turbulence: Dryden-inspired sinusoidal gusts
        gust = 0.3 * intensity * np.array([
            np.sin(0.5 * t + 1.2),
            np.sin(0.7 * t + 2.4),
            np.sin(0.3 * t) * 0.3,
        ])
        return base + gust
    return wind_fn


# ===========================================================================
# SIDEBAR
# ===========================================================================

with st.sidebar:
    st.markdown(
        '<div class="hud-title">🚁 UAV SIM</div>'
        '<div class="hud-subtitle">MISSION CONTROL INTERFACE v2.0</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Drone Hardware ────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-header">▶ Drone Hardware</div>', unsafe_allow_html=True)
    drone_mass      = st.slider("Mass (kg)",        0.5, 5.0,  1.5, 0.1)
    drone_max_speed = st.slider("Max Speed (m/s)", 2.0, 15.0,  8.0, 0.5)
    drone_max_thrust= st.slider("Max Thrust (N)",  10.0, 60.0, 35.0, 1.0)
    drag_coeff      = st.slider("Drag Coefficient",0.05, 0.4,  0.12, 0.01)

    # ── Environment ───────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-header">▶ Environment</div>', unsafe_allow_html=True)
    env_size     = st.slider("World Size (m)",   30, 120, 60, 5)
    env_height   = st.slider("Max Altitude (m)", 20,  60, 35, 5)
    n_cylinders  = st.slider("Cylinder Buildings",  0, 12, 6, 1)
    n_boxes      = st.slider("Box Structures",       0,  8, 4, 1)
    n_nfz        = st.slider("No-Fly Zones",         0,  4, 2, 1)
    env_seed     = st.number_input("Environment Seed", 0, 9999, 42, step=1)

    # ── Mission ───────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-header">▶ Mission</div>', unsafe_allow_html=True)
    takeoff_alt  = st.slider("Takeoff Altitude (m)",   1.0, 10.0,  3.0, 0.5)
    cruise_alt   = st.slider("Cruise Altitude (m)",    5.0, 30.0, 15.0, 1.0)
    planner_res  = st.slider("Planner Resolution (m)", 0.8,  3.0,  1.5, 0.1)
    safety_margin= st.slider("Safety Margin (m)",      0.5,  4.0,  1.5, 0.25)

    # ── Weather ───────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-header">▶ Weather / Wind</div>', unsafe_allow_html=True)
    wind_intensity = st.slider("Wind Force (N)",   0.0, 8.0, 0.0, 0.5)
    wind_direction = st.slider("Wind Direction (°)", 0, 359, 45, 5) if wind_intensity > 0 else 45

    # ── LiDAR ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-header">▶ LiDAR Sensor</div>', unsafe_allow_html=True)
    lidar_on    = st.toggle("Enable LiDAR", value=True)
    lidar_range = st.slider("LiDAR Range (m)",       5.0, 40.0, 20.0, 1.0) if lidar_on else 20.0
    lidar_h_res = st.slider("H-Resolution (rays)",  18, 144,   72,  18)    if lidar_on else 72
    lidar_v_ch  = st.slider("Vertical Channels",      4,  32,   16,   4)   if lidar_on else 16
    show_nfz    = st.toggle("Show No-Fly Zones", value=True)

    # ── Simulation ────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-header">▶ Simulation</div>', unsafe_allow_html=True)
    sim_dt          = st.select_slider("Timestep (s)", [0.02, 0.05, 0.1, 0.2], value=0.05)
    lidar_every_n   = st.slider("LiDAR every N frames", 1, 20, 5, 1)

    st.markdown("")
    run_btn   = st.button("🚀  RUN SIMULATION",   type="primary")
    reset_btn = st.button("⟳  RESET")

    if reset_btn:
        for k in ["sim_run", "states", "path", "environment", "frame_idx", "lidar_cache"]:
            st.session_state[k] = [] if k in ("states", "path") else (
                {} if k == "lidar_cache" else
                False if k == "sim_run" else
                None if k == "environment" else 0
            )
        st.rerun()


# ===========================================================================
# SIMULATION ENGINE — triggered by Run button
# ===========================================================================

if run_btn:
    with st.spinner("🔧  Building environment…"):
        env = Environment(bounds=(float(env_size), float(env_size), float(env_height)))
        env.generate_urban_environment(
            n_cylinders=n_cylinders,
            n_boxes=n_boxes,
            n_nfz=n_nfz,
            seed=int(env_seed),
        )
        st.session_state.environment = env

    with st.spinner("🗺️  Planning path (A*)…"):
        start = np.array([1.0, 1.0, takeoff_alt])
        goal  = np.array([env_size - 1.0, env_size - 1.0, takeoff_alt])

        planner = AStarPlanner3D(env, resolution=planner_res, safety_margin=safety_margin)
        t0 = time.perf_counter()
        path = planner.plan(start, goal)
        st.session_state.plan_time = time.perf_counter() - t0
        st.session_state.path = path

    with st.spinner("⚙️  Running physics simulation…"):
        drone = DronePhysics(
            mass=drone_mass,
            max_speed=drone_max_speed,
            max_thrust=drone_max_thrust,
            drag_coeff=drag_coeff,
        )
        drone.reset(start)

        wind_fn = make_wind_fn(wind_intensity, wind_direction)

        t0 = time.perf_counter()
        states = drone.simulate_mission(
            waypoints=path,
            dt=sim_dt,
            wind_fn=wind_fn,
            arrival_radius=0.8,
            max_steps_per_wp=800,
        )
        st.session_state.sim_time = time.perf_counter() - t0
        st.session_state.states = states
        st.session_state.frame_idx = 0

    if lidar_on and states:
        with st.spinner("📡  Computing LiDAR scans…"):
            lidar = LiDARSensor(
                range_max=lidar_range,
                n_horizontal=lidar_h_res,
                n_vertical=lidar_v_ch,
            )
            lidar_cache = {}
            rng = np.random.RandomState(0)
            for i, s in enumerate(states):
                if i % lidar_every_n == 0:
                    pts, inten = lidar.scan(s.position, s.yaw, env, rng=rng)
                    lidar_cache[i] = (pts, inten)
            st.session_state.lidar_cache = lidar_cache

    st.session_state.sim_run = True
    st.rerun()


# ===========================================================================
# HEADER
# ===========================================================================

col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown(
        '<span class="hud-title">🛩 UAV MISSION SIMULATION</span>  '
        '<span class="hud-subtitle">3-D ENVIRONMENT · A* PLANNER · LiDAR SENSOR FUSION</span>',
        unsafe_allow_html=True,
    )

with col_status:
    if st.session_state.sim_run and st.session_state.states:
        n_states = len(st.session_state.states)
        badge_style = "background:#064E3B;color:#6EE7B7;border:1px solid #10B981;"
        st.markdown(
            f'<div style="text-align:right;margin-top:6px;">'
            f'<span class="status-badge" style="{badge_style}">✔ SIMULATION READY</span><br>'
            f'<span class="hud-subtitle">{n_states} FRAMES · '
            f'PLAN {st.session_state.plan_time*1000:.0f}ms · '
            f'SIM {st.session_state.sim_time*1000:.0f}ms</span></div>',
            unsafe_allow_html=True,
        )
    else:
        badge_style = "background:#1C1917;color:#78716C;border:1px solid #44403C;"
        st.markdown(
            f'<div style="text-align:right;margin-top:6px;">'
            f'<span class="status-badge" style="{badge_style}">⬤ AWAITING LAUNCH</span></div>',
            unsafe_allow_html=True,
        )

st.markdown("---")


# ===========================================================================
# MAIN CONTENT — Only shown after simulation
# ===========================================================================

if not st.session_state.sim_run or not st.session_state.states:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px;">
      <div style="font-size:5rem;">🚁</div>
      <div style="font-family:'Share Tech Mono',monospace; color:#1E40AF; font-size:1.1rem; margin-top:12px;">
        CONFIGURE MISSION PARAMETERS IN SIDEBAR<br>
        <span style="color:#475569; font-size:0.85rem;">THEN PRESS ▶ RUN SIMULATION</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ===========================================================================
# EXTRACT SIMULATION DATA
# ===========================================================================

states: list[DroneState] = st.session_state.states
env: Environment = st.session_state.environment
path = st.session_state.path
n_frames = len(states)

# Build arrays for plotting
positions  = np.array([s.position for s in states])
velocities = np.array([s.velocity for s in states])
speeds     = np.array([s.speed()  for s in states])
altitudes  = positions[:, 2]
batteries  = np.array([s.battery  for s in states])
timestamps = np.array([s.timestamp for s in states])
yaws       = np.array([s.yaw      for s in states])
pitches    = np.array([s.pitch    for s in states])
motor_rpms = np.array([s.motor_rpm for s in states])
thrusts    = np.array([s.thrust   for s in states])

# Nearest obstacle distances
obs_distances = []
nfz_flags = []
for s in states:
    dist, _ = env.nearest_obstacle_info(s.position)
    obs_distances.append(dist)
    nfz_flags.append(env.in_no_fly_zone(s.position))
obs_distances = np.array(obs_distances)
nfz_flags = np.array(nfz_flags)

DANGER_DIST  = safety_margin + 0.5
WARNING_DIST = safety_margin + 3.0


# ===========================================================================
# HUD METRICS ROW
# ===========================================================================

frame_idx = st.session_state.frame_idx
cur = states[frame_idx]
cur_dist = obs_distances[frame_idx]

mc = st.columns(7)
mc[0].metric("Altitude",    f"{cur.altitude():.1f} m",   f"{cur.altitude() - states[0].altitude():.1f} m")
mc[1].metric("Speed",       f"{cur.speed():.2f} m/s",    f"{cur.speed() - states[max(0,frame_idx-1)].speed():.2f}")
mc[2].metric("Battery",     f"{cur.battery:.1f}%",       f"{cur.battery - 100:.1f}%")
mc[3].metric("Heading",     f"{np.rad2deg(cur.yaw):.0f}°")
mc[4].metric("Motor RPM",   f"{cur.motor_rpm:,.0f}")
mc[5].metric("Obs. Dist.",  f"{cur_dist:.1f} m",
             delta=None if cur_dist > WARNING_DIST else "⚠ CLOSE",
             delta_color="inverse" if cur_dist < DANGER_DIST else "normal")
mc[6].metric("Frame",       f"{frame_idx+1}/{n_frames}")

st.markdown("")


# ===========================================================================
# TABS
# ===========================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🌐  3D Mission View",
    "📊  Telemetry",
    "🛡  Collision Monitor",
    "📋  Mission Report",
])


# ---------------------------------------------------------------------------
# TAB 1 — 3D MISSION VIEW
# ---------------------------------------------------------------------------

with tab1:
    # Frame scrubber
    frame_idx = st.slider(
        "🎞  Scrub Timeline",
        min_value=0,
        max_value=n_frames - 1,
        value=st.session_state.frame_idx,
        key="frame_slider",
    )
    st.session_state.frame_idx = frame_idx

    cur_state = states[frame_idx]

    # Get LiDAR for nearest cached frame
    lidar_pts, lidar_int = None, None
    if lidar_on and st.session_state.lidar_cache:
        # Find nearest cached frame
        cached_frames = sorted(st.session_state.lidar_cache.keys())
        nearest_cached = min(cached_frames, key=lambda f: abs(f - frame_idx))
        lidar_pts, lidar_int = st.session_state.lidar_cache[nearest_cached]

    # Build scene
    fig = build_3d_scene(
        environment=env,
        path=path,
        drone_position=cur_state.position,
        drone_yaw=cur_state.yaw,
        lidar_points=lidar_pts if lidar_on else None,
        lidar_intensities=lidar_int if lidar_on else None,
        trajectory_history=list(positions[:frame_idx + 1]),
        trajectory_speeds=list(speeds[:frame_idx + 1]),
        show_lidar=lidar_on,
        show_nfz=show_nfz,
    )
    st.plotly_chart(fig, use_container_width=True, key="scene3d")

    # Inline alert
    if nfz_flags[frame_idx]:
        st.markdown('<div class="alert-danger">⛔  NO-FLY ZONE VIOLATION DETECTED AT THIS FRAME</div>', unsafe_allow_html=True)
    elif obs_distances[frame_idx] < DANGER_DIST:
        st.markdown(f'<div class="alert-danger">🚨  COLLISION RISK — Obstacle at {obs_distances[frame_idx]:.1f} m</div>', unsafe_allow_html=True)
    elif obs_distances[frame_idx] < WARNING_DIST:
        st.markdown(f'<div class="alert-warning">⚠️  Obstacle proximity warning — {obs_distances[frame_idx]:.1f} m</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-ok">✅  Clear — Nearest obstacle {obs_distances[frame_idx]:.1f} m away</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# TAB 2 — TELEMETRY
# ---------------------------------------------------------------------------

with tab2:
    t_arr = timestamps

    c1, c2 = st.columns(2)

    # --- Altitude ---
    with c1:
        fig_alt = go.Figure()
        fig_alt.add_trace(go.Scatter(
            x=t_arr, y=altitudes,
            mode="lines",
            line=dict(color="#00D4FF", width=2),
            name="Altitude",
            fill="tozeroy",
            fillcolor="rgba(0, 212, 255, 0.08)",
        ))
        fig_alt.add_vline(
            x=t_arr[frame_idx], line_dash="dash",
            line_color="#A855F7", line_width=1.5,
        )
        fig_alt.update_layout(
            title=dict(text="Altitude (m)", font=dict(color="#00D4FF", size=13)),
            paper_bgcolor="#080816", plot_bgcolor="#080816",
            font=dict(color="#94A3B8"),
            xaxis=dict(title="Time (s)", gridcolor="#0e0e2c"),
            yaxis=dict(title="m", gridcolor="#0e0e2c"),
            height=260, margin=dict(l=40, r=10, t=40, b=30),
        )
        st.plotly_chart(fig_alt, use_container_width=True)

    # --- Speed ---
    with c2:
        fig_spd = go.Figure()
        fig_spd.add_trace(go.Scatter(
            x=t_arr, y=speeds,
            mode="lines",
            line=dict(color="#F59E0B", width=2),
            name="Speed",
            fill="tozeroy",
            fillcolor="rgba(245, 158, 11, 0.08)",
        ))
        fig_spd.add_hline(
            y=drone_max_speed, line_dash="dot",
            line_color="#EF4444", line_width=1,
            annotation_text="Max Speed",
            annotation_font_color="#EF4444",
        )
        fig_spd.add_vline(
            x=t_arr[frame_idx], line_dash="dash",
            line_color="#A855F7", line_width=1.5,
        )
        fig_spd.update_layout(
            title=dict(text="Speed (m/s)", font=dict(color="#F59E0B", size=13)),
            paper_bgcolor="#080816", plot_bgcolor="#080816",
            font=dict(color="#94A3B8"),
            xaxis=dict(title="Time (s)", gridcolor="#0e0e2c"),
            yaxis=dict(title="m/s", gridcolor="#0e0e2c"),
            height=260, margin=dict(l=40, r=10, t=40, b=30),
        )
        st.plotly_chart(fig_spd, use_container_width=True)

    c3, c4 = st.columns(2)

    # --- Battery ---
    with c3:
        # Colour by level
        bat_colors = ["#10B981" if b > 40 else "#F59E0B" if b > 15 else "#EF4444"
                      for b in batteries]
        fig_bat = go.Figure()
        fig_bat.add_trace(go.Scatter(
            x=t_arr, y=batteries,
            mode="lines",
            line=dict(color="#10B981", width=2),
            name="Battery",
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.08)",
        ))
        fig_bat.add_hline(y=20, line_dash="dot", line_color="#EF4444", line_width=1,
                          annotation_text="Low Battery", annotation_font_color="#EF4444")
        fig_bat.add_vline(x=t_arr[frame_idx], line_dash="dash",
                          line_color="#A855F7", line_width=1.5)
        fig_bat.update_layout(
            title=dict(text="Battery (%)", font=dict(color="#10B981", size=13)),
            paper_bgcolor="#080816", plot_bgcolor="#080816",
            font=dict(color="#94A3B8"),
            xaxis=dict(title="Time (s)", gridcolor="#0e0e2c"),
            yaxis=dict(title="%", range=[0, 105], gridcolor="#0e0e2c"),
            height=260, margin=dict(l=40, r=10, t=40, b=30),
        )
        st.plotly_chart(fig_bat, use_container_width=True)

    # --- Motor RPM + Thrust ---
    with c4:
        fig_rpm = go.Figure()
        fig_rpm.add_trace(go.Scatter(
            x=t_arr, y=motor_rpms,
            mode="lines",
            line=dict(color="#A855F7", width=2),
            name="Motor RPM",
            yaxis="y1",
        ))
        fig_rpm.add_trace(go.Scatter(
            x=t_arr, y=thrusts,
            mode="lines",
            line=dict(color="#EC4899", width=1.5, dash="dot"),
            name="Thrust (N)",
            yaxis="y2",
        ))
        fig_rpm.add_vline(x=t_arr[frame_idx], line_dash="dash",
                          line_color="#00D4FF", line_width=1.5)
        fig_rpm.update_layout(
            title=dict(text="Motor RPM / Thrust", font=dict(color="#A855F7", size=13)),
            paper_bgcolor="#080816", plot_bgcolor="#080816",
            font=dict(color="#94A3B8"),
            xaxis=dict(title="Time (s)", gridcolor="#0e0e2c"),
            yaxis=dict(title="RPM", gridcolor="#0e0e2c"),
            yaxis2=dict(title="Thrust (N)", overlaying="y", side="right", gridcolor="#0e0e2c"),
            legend=dict(bgcolor="rgba(0,0,0,0.5)"),
            height=260, margin=dict(l=40, r=50, t=40, b=30),
        )
        st.plotly_chart(fig_rpm, use_container_width=True)

    # --- 3D Velocity Phase Portrait ---
    st.markdown("#### Velocity Phase Portrait")
    vx = velocities[:, 0]
    vy = velocities[:, 1]
    vz = velocities[:, 2]
    fig_phase = go.Figure(go.Scatter3d(
        x=vx, y=vy, z=vz,
        mode="lines",
        line=dict(color=speeds, colorscale="Plasma", width=3, cmin=0, cmax=float(drone_max_speed)),
        hoverinfo="skip",
    ))
    fig_phase.update_layout(
        scene=dict(
            xaxis=dict(title="Vx", backgroundcolor="#080816", gridcolor="#0e0e2c"),
            yaxis=dict(title="Vy", backgroundcolor="#080816", gridcolor="#0e0e2c"),
            zaxis=dict(title="Vz", backgroundcolor="#080816", gridcolor="#0e0e2c"),
            bgcolor="#080816",
        ),
        paper_bgcolor="#080816", font=dict(color="#94A3B8"),
        height=380, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_phase, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 3 — COLLISION MONITOR
# ---------------------------------------------------------------------------

with tab3:
    col_a, col_b = st.columns([3, 1])

    with col_a:
        # Proximity time-series
        fig_prox = go.Figure()
        fig_prox.add_hrect(
            y0=0, y1=DANGER_DIST,
            fillcolor="rgba(220,38,38,0.12)", line_width=0,
            annotation_text="DANGER ZONE", annotation_font_color="#EF4444",
        )
        fig_prox.add_hrect(
            y0=DANGER_DIST, y1=WARNING_DIST,
            fillcolor="rgba(245,158,11,0.08)", line_width=0,
            annotation_text="WARNING ZONE", annotation_font_color="#F59E0B",
        )
        fig_prox.add_trace(go.Scatter(
            x=t_arr, y=obs_distances,
            mode="lines",
            line=dict(color="#00D4FF", width=2),
            name="Nearest Obstacle",
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.05)",
        ))
        fig_prox.add_vline(
            x=t_arr[frame_idx], line_dash="dash",
            line_color="#A855F7", line_width=2,
        )
        fig_prox.update_layout(
            title=dict(text="Nearest Obstacle Distance (m)", font=dict(color="#00D4FF", size=13)),
            paper_bgcolor="#080816", plot_bgcolor="#080816",
            font=dict(color="#94A3B8"),
            xaxis=dict(title="Time (s)", gridcolor="#0e0e2c"),
            yaxis=dict(title="Distance (m)", gridcolor="#0e0e2c"),
            height=320, margin=dict(l=40, r=10, t=40, b=30),
        )
        st.plotly_chart(fig_prox, use_container_width=True)

        # NFZ violation timeline
        if nfz_flags.any():
            fig_nfz = go.Figure()
            fig_nfz.add_trace(go.Scatter(
                x=t_arr, y=nfz_flags.astype(int),
                mode="lines",
                line=dict(color="#EF4444", width=2),
                fill="tozeroy",
                fillcolor="rgba(239,68,68,0.15)",
                name="NFZ Violation",
            ))
            fig_nfz.update_layout(
                title=dict(text="No-Fly Zone Violations", font=dict(color="#EF4444", size=13)),
                paper_bgcolor="#080816", plot_bgcolor="#080816",
                font=dict(color="#94A3B8"),
                xaxis=dict(title="Time (s)", gridcolor="#0e0e2c"),
                yaxis=dict(title="Violation (0/1)", range=[-0.1, 1.5], gridcolor="#0e0e2c"),
                height=180, margin=dict(l=40, r=10, t=40, b=30),
            )
            st.plotly_chart(fig_nfz, use_container_width=True)

    with col_b:
        st.markdown("#### ⚠ Alert Summary")

        danger_frames  = np.sum(obs_distances < DANGER_DIST)
        warning_frames = np.sum((obs_distances >= DANGER_DIST) & (obs_distances < WARNING_DIST))
        nfz_frames     = np.sum(nfz_flags)
        min_sep        = float(obs_distances.min())

        if danger_frames > 0:
            st.markdown(f'<div class="alert-danger">🚨 COLLISION RISK<br>{danger_frames} frames in danger zone</div>', unsafe_allow_html=True)
        if warning_frames > 0:
            st.markdown(f'<div class="alert-warning">⚠️ PROXIMITY<br>{warning_frames} frames in warning zone</div>', unsafe_allow_html=True)
        if nfz_frames > 0:
            st.markdown(f'<div class="alert-danger">⛔ NFZ VIOLATION<br>{nfz_frames} frames</div>', unsafe_allow_html=True)
        if danger_frames == 0 and warning_frames == 0 and nfz_frames == 0:
            st.markdown('<div class="alert-ok">✅ NO ALERTS<br>Mission within safe parameters</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.metric("Min Separation", f"{min_sep:.2f} m",
                  delta="DANGER" if min_sep < DANGER_DIST else ("WARNING" if min_sep < WARNING_DIST else "SAFE"),
                  delta_color="inverse" if min_sep < DANGER_DIST else "normal")
        st.metric("Danger Frames",  f"{danger_frames}")
        st.metric("Warning Frames", f"{warning_frames}")
        st.metric("NFZ Violations", f"{nfz_frames}")

    # --- Obstacle proximity map (2D top-down)
    st.markdown("#### Top-Down Proximity Map (XY Plane)")
    fig_2d = go.Figure()

    # Plot obstacle footprints
    for obs in env.obstacles:
        theta = np.linspace(0, 2 * np.pi, 60)
        if obs.obstacle_type == "cylinder":
            fig_2d.add_trace(go.Scatter(
                x=obs.center[0] + obs.radius * np.cos(theta),
                y=obs.center[1] + obs.radius * np.sin(theta),
                fill="toself", fillcolor="rgba(220,38,38,0.3)",
                line=dict(color=obs.color, width=1),
                showlegend=False, hoverinfo="skip",
            ))
        else:
            hs = obs.radius
            cx, cy = obs.center[0], obs.center[1]
            fig_2d.add_trace(go.Scatter(
                x=[cx-hs, cx+hs, cx+hs, cx-hs, cx-hs],
                y=[cy-hs, cy-hs, cy+hs, cy+hs, cy-hs],
                fill="toself", fillcolor="rgba(180,100,0,0.3)",
                line=dict(color=obs.color, width=1),
                showlegend=False, hoverinfo="skip",
            ))

    # NFZs
    for nfz in env.no_fly_zones:
        theta = np.linspace(0, 2 * np.pi, 60)
        fig_2d.add_trace(go.Scatter(
            x=nfz.center[0] + nfz.radius * np.cos(theta),
            y=nfz.center[1] + nfz.radius * np.sin(theta),
            fill="toself", fillcolor="rgba(239,68,68,0.08)",
            line=dict(color="#EF4444", width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))

    # Path
    if path:
        pa = np.array(path)
        fig_2d.add_trace(go.Scatter(
            x=pa[:, 0], y=pa[:, 1],
            mode="lines",
            line=dict(color="#00D4FF", width=2, dash="dash"),
            name="Path",
        ))

    # Trail up to current frame
    fig_2d.add_trace(go.Scatter(
        x=positions[:frame_idx+1, 0], y=positions[:frame_idx+1, 1],
        mode="lines",
        line=dict(color="#F59E0B", width=3),
        name="Flown",
    ))

    # Current drone position
    fig_2d.add_trace(go.Scatter(
        x=[positions[frame_idx, 0]], y=[positions[frame_idx, 1]],
        mode="markers",
        marker=dict(size=12, color="#A855F7", symbol="diamond"),
        name="Drone",
    ))

    fig_2d.update_layout(
        paper_bgcolor="#080816", plot_bgcolor="#080816",
        font=dict(color="#94A3B8"),
        xaxis=dict(title="X (m)", range=[0, env_size], gridcolor="#0e0e2c",
                   scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Y (m)", range=[0, env_size], gridcolor="#0e0e2c"),
        height=420, margin=dict(l=40, r=10, t=10, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
    )
    st.plotly_chart(fig_2d, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 4 — MISSION REPORT
# ---------------------------------------------------------------------------

with tab4:
    total_time  = float(timestamps[-1])
    total_dist  = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    avg_speed   = float(speeds.mean())
    max_speed_  = float(speeds.max())
    max_alt     = float(altitudes.max())
    bat_used    = float(100.0 - batteries[-1])
    path_len    = float(sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
                            for i in range(len(path)-1))) if path else 0.0

    st.markdown("### 📋 Mission Summary")
    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown("**Flight Statistics**")
        st.markdown(f"""
| Parameter | Value |
|---|---|
| Total Flight Time | {total_time:.1f} s |
| Distance Covered | {total_dist:.1f} m |
| Avg Speed | {avg_speed:.2f} m/s |
| Max Speed | {max_speed_:.2f} m/s |
| Max Altitude | {max_alt:.1f} m |
| Battery Used | {bat_used:.1f}% |
""")

    with r2:
        st.markdown("**Path Planning**")
        st.markdown(f"""
| Parameter | Value |
|---|---|
| Planning Time | {st.session_state.plan_time*1000:.0f} ms |
| Path Waypoints | {len(path)} |
| Path Length | {path_len:.1f} m |
| Planner Resolution | {planner_res} m |
| Safety Margin | {safety_margin} m |
| Algorithm | A* + Cubic Spline |
""")

    with r3:
        st.markdown("**Safety Assessment**")
        mission_status = "✅ SAFE" if (danger_frames == 0 and nfz_frames == 0) else "⚠️ VIOLATIONS"
        st.markdown(f"""
| Parameter | Value |
|---|---|
| Mission Status | {mission_status} |
| Min Obstacle Sep. | {min_sep:.2f} m |
| Danger Frames | {danger_frames} ({danger_frames/n_frames*100:.1f}%) |
| Warning Frames | {warning_frames} ({warning_frames/n_frames*100:.1f}%) |
| NFZ Violations | {nfz_frames} frames |
| Obstacles in Env | {len(env.obstacles)} |
""")

    st.markdown("---")
    st.markdown("### 🔧 Configuration Used")
    cfg_col1, cfg_col2, cfg_col3 = st.columns(3)

    with cfg_col1:
        st.markdown("**Drone Hardware**")
        st.json({
            "mass_kg": drone_mass,
            "max_speed_ms": drone_max_speed,
            "max_thrust_N": drone_max_thrust,
            "drag_coefficient": drag_coeff,
        })

    with cfg_col2:
        st.markdown("**Environment**")
        st.json({
            "world_size_m": env_size,
            "max_altitude_m": env_height,
            "cylinder_buildings": n_cylinders,
            "box_structures": n_boxes,
            "no_fly_zones": n_nfz,
            "seed": int(env_seed),
        })

    with cfg_col3:
        st.markdown("**Simulation**")
        st.json({
            "timestep_s": sim_dt,
            "wind_force_N": wind_intensity,
            "lidar_enabled": lidar_on,
            "lidar_range_m": lidar_range if lidar_on else "N/A",
            "total_sim_frames": n_frames,
        })

    # Download flight log
    st.markdown("---")
    st.markdown("### 📥 Export Flight Log")
    log_df = pd.DataFrame({
        "time_s":     timestamps,
        "x_m":        positions[:, 0],
        "y_m":        positions[:, 1],
        "z_m":        positions[:, 2],
        "speed_ms":   speeds,
        "yaw_rad":    yaws,
        "pitch_rad":  pitches,
        "battery_pct":batteries,
        "motor_rpm":  motor_rpms,
        "thrust_N":   thrusts,
        "obs_dist_m": obs_distances,
        "nfz_flag":   nfz_flags.astype(int),
    })
    st.download_button(
        label="⬇️  Download Flight Log (CSV)",
        data=log_df.to_csv(index=False),
        file_name="uav_flight_log.csv",
        mime="text/csv",
    )
