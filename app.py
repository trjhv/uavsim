"""
app.py — SAR UAV Multi-Drone Simulation Dashboard
===================================================
Urban Flood Search & Rescue with Hybrid APF-PSO-Vortex Navigation

Deploy: streamlit run app.py
GitHub: Place all files in repo root; Streamlit Cloud deploys automatically.

Three-panel layout:
  ┌────────────────────────────┬──────────────────────┐
  │                            │  TACTICAL MAP        │
  │   MAIN 3D SCENE            │  (top-down, FPV)     │
  │   Full isometric world     ├──────────────────────┤
  │   + drone models + trails  │  LiDAR + APF FIELD   │
  │                            │  (sensor panel)      │
  └────────────────────────────┴──────────────────────┘

Keyboard control (UAV-1):
  WASD / Arrow keys — directional flight
  R / Space — Ascend    |    F / Shift — Descend
  Q — Yaw Left          |    E — Yaw Right
  H — Hold position     |    TAB — Toggle auto/manual
"""

import time
import json
import math
import numpy as np
from numpy.linalg import norm
import streamlit as st
import streamlit.components.v1 as components

# ── Simulation modules ────────────────────────────────────────────────────────
from simulation.environment import (
    create_flood_scene, get_all_obstacle_centers,
    FloodScene, DRONE_STARTS, DETECTION_RADIUS, FLOOD_Z,
    WORLD_X, WORLD_Y
)
from simulation.drone        import DroneState, DronePhysics, create_drone_state
from simulation.algorithms   import (
    PSOSwarm, compute_hybrid_force, compute_manual_force,
    select_target, DETECTION_RADIUS as _
)
from simulation.lidar        import LiDARSensor
from simulation.visualization import (
    build_main_scene, build_tactical_map, build_sensor_panel
)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIGURATION — must be first Streamlit call
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title  = "SAR UAV Sim | APF-PSO-Vortex",
    page_icon   = "🚁",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)


# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS — Military HUD aesthetic
#  Font: "Share Tech Mono" (sci-fi monospace) + "Barlow Condensed" (titles)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700;900&display=swap');

/* ── Base ── */
.stApp { background-color: #030A12; color: #8FBAD0; }
.main .block-container { padding: 0.6rem 0.8rem 0.8rem 0.8rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #040D18 0%, #030A12 100%);
  border-right: 1px solid #0A2540;
}
section[data-testid="stSidebar"] * { font-family: 'Share Tech Mono', monospace !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
  background: rgba(8,20,40,0.90);
  border: 1px solid #0A3060;
  border-radius: 3px;
  padding: 6px 10px !important;
}
[data-testid="stMetricLabel"] {
  color: #2E5A80 !important; font-size: 0.64rem !important;
  text-transform: uppercase; letter-spacing: 0.12em;
  font-family: 'Share Tech Mono', monospace !important;
}
[data-testid="stMetricValue"] {
  color: #00C8FF !important; font-size: 1.15rem !important;
  font-family: 'Share Tech Mono', monospace !important;
}
[data-testid="stMetricDelta"] { font-size: 0.65rem !important; }

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, #0A2040, #061428);
  border: 1px solid #1A4880;
  color: #00C8FF;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.82rem;
  letter-spacing: 0.06em;
  border-radius: 3px;
  padding: 4px 12px;
  transition: all 0.15s;
  width: 100%;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #1A4880, #0A2040);
  border-color: #00C8FF;
  box-shadow: 0 0 10px rgba(0,200,255,0.3);
}
.stButton > button:active {
  background: #00C8FF;
  color: #030A12;
  box-shadow: 0 0 20px rgba(0,200,255,0.6);
}

/* ── Sliders ── */
[data-testid="stSlider"] > div > div { background: #1A4880 !important; }
[data-testid="stSlider"] > div > div > div { background: #00C8FF !important; }

/* ── Toggle ── */
[data-testid="stCheckbox"] span { color: #8FBAD0; }

/* ── Section dividers ── */
hr { border-color: #0A2540 !important; }

/* ── D-PAD keyboard control widget ── */
.dpad-container {
  display: grid;
  grid-template-areas:
    ". up ."
    "left stop right"
    ". down .";
  grid-template-columns: 1fr 1fr 1fr;
  gap: 4px;
  width: 120px;
  margin: 0 auto;
}
.dpad-btn {
  background: linear-gradient(145deg, #0E2A45, #071A30);
  border: 1px solid #1A4880;
  border-radius: 4px;
  color: #00C8FF;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.75rem;
  padding: 6px 0;
  text-align: center;
  cursor: pointer;
  transition: all 0.1s;
  user-select: none;
  -webkit-user-select: none;
}
.dpad-btn:hover { border-color: #00C8FF; box-shadow: 0 0 8px rgba(0,200,255,0.4); }
.dpad-btn:active { background: #00C8FF; color: #030A12; }
.dpad-btn.active { background: #00C8FF !important; color: #030A12 !important; box-shadow: 0 0 12px rgba(0,200,255,0.7); }

/* ── Panel title bars ── */
.panel-title {
  font-family: 'Barlow Condensed', sans-serif;
  font-weight: 700;
  font-size: 0.72rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #1A6090;
  border-bottom: 1px solid #0A3060;
  padding-bottom: 3px;
  margin-bottom: 4px;
}

/* ── Status badges ── */
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 2px;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.badge-ok   { background: rgba(0,180,80,0.15); color: #00E060; border: 1px solid #00A040; }
.badge-warn { background: rgba(255,140,0,0.15); color: #FFA020; border: 1px solid #CC6600; }
.badge-crit { background: rgba(220,20,30,0.15); color: #FF3040; border: 1px solid #CC1020; }
.badge-info { background: rgba(0,160,220,0.15); color: #00C8FF; border: 1px solid #006090; }

/* ── Alert box ── */
.alert-box {
  padding: 6px 10px;
  border-radius: 3px;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.72rem;
  margin: 3px 0;
}
.alert-found { background: rgba(0,200,80,0.12); border-left: 3px solid #00C050; color: #60FF90; }
.alert-warn  { background: rgba(255,120,0,0.12); border-left: 3px solid #FF7000; color: #FFAA40; }

/* ── App title ── */
.app-title {
  font-family: 'Barlow Condensed', sans-serif;
  font-weight: 900;
  font-size: 1.45rem;
  letter-spacing: 0.08em;
  color: #00C8FF;
  text-shadow: 0 0 20px rgba(0,200,255,0.45);
  line-height: 1.1;
}
.app-sub {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.62rem;
  color: #2A5A80;
  letter-spacing: 0.15em;
  text-transform: uppercase;
}

/* ── Progress bar custom ── */
.stProgress > div > div > div > div {
  background: linear-gradient(90deg, #006090, #00C8FF) !important;
}

/* ── Hide Streamlit branding ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  KEYBOARD LISTENER — JavaScript injection
#  Captures WASD/Arrow key presses from the browser and stores them in
#  the Streamlit session via a hidden text_input.
#  Works in all modern browsers; updates every 80ms via polling.
# ═══════════════════════════════════════════════════════════════════════════════

KEYBOARD_JS = """
<script>
(function() {
  // Avoid double-injection
  if (window._sarKeysInit) return;
  window._sarKeysInit = true;
  window._sarActiveKeys = {};

  const TRACKED = new Set(['w','a','s','d','q','e','r','f','h',
    'arrowup','arrowdown','arrowleft','arrowright',' ','shift','tab']);

  document.addEventListener('keydown', function(e) {
    const k = e.key.toLowerCase();
    if (TRACKED.has(k)) {
      e.preventDefault();
      window._sarActiveKeys[k] = 1;
    }
  });

  document.addEventListener('keyup', function(e) {
    delete window._sarActiveKeys[e.key.toLowerCase()];
  });

  // Poll and inject into the hidden text input every 80ms
  setInterval(function() {
    const keys = Object.keys(window._sarActiveKeys).join(',');
    // Find our specific hidden input by placeholder attribute
    const allInputs = window.parent.document.querySelectorAll('input[type="text"]');
    for (const inp of allInputs) {
      if (inp.getAttribute('placeholder') === '__sar_keys__') {
        if (inp.value !== keys) {
          const setter = Object.getOwnPropertyDescriptor(
            window.HTMLInputElement.prototype, 'value').set;
          setter.call(inp, keys);
          inp.dispatchEvent(new Event('input', {bubbles: true}));
        }
        break;
      }
    }
  }, 80);
})();
</script>
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE — Initialise all simulation state on first run
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _get_scene_and_obstacles():
    """Cache the flood scene (never changes between reruns)."""
    scene = create_flood_scene(seed=42)
    obstacles = get_all_obstacle_centers(scene)
    return scene, obstacles


@st.cache_resource
def _get_lidar():
    return LiDARSensor(n_horizontal=72, n_vertical=16,
                       fov_vert_deg=30, range_max=25, noise_std=0.08)


def _init_session():
    """Initialise session state on first load."""
    scene, obstacles = _get_scene_and_obstacles()

    if "drones" not in st.session_state:
        st.session_state.drones = [create_drone_state(i) for i in range(3)]

    if "physics" not in st.session_state:
        st.session_state.physics = [
            DronePhysics(st.session_state.drones[i])
            for i in range(3)
        ]

    if "pso" not in st.session_state:
        surv_positions = [s.position for s in scene.survivors]
        st.session_state.pso = [
            PSOSwarm(DRONE_STARTS[i], surv_positions, seed=i*7)
            for i in range(3)
        ]

    if "found_idx" not in st.session_state:
        st.session_state.found_idx = set()

    if "sim_running" not in st.session_state:
        st.session_state.sim_running = False

    if "frame_count" not in st.session_state:
        st.session_state.frame_count = 0

    if "sim_speed" not in st.session_state:
        st.session_state.sim_speed = 1.0

    if "dt" not in st.session_state:
        st.session_state.dt = 0.08

    if "lidar_pts" not in st.session_state:
        st.session_state.lidar_pts   = None
        st.session_state.lidar_dists = None

    if "lidar_frame_counter" not in st.session_state:
        st.session_state.lidar_frame_counter = 0

    if "active_keys" not in st.session_state:
        st.session_state.active_keys = set()

    if "manual_mode" not in st.session_state:
        st.session_state.manual_mode = True   # Drone 0 manual by default

    if "mission_start_time" not in st.session_state:
        st.session_state.mission_start_time = None

    if "alerts" not in st.session_state:
        st.session_state.alerts = []

    if "cam_eye" not in st.session_state:
        st.session_state.cam_eye = dict(x=-0.8, y=-1.6, z=0.9)


_init_session()


# ═══════════════════════════════════════════════════════════════════════════════
#  SIMULATION STEP — Runs on every Streamlit rerun when sim_running=True
# ═══════════════════════════════════════════════════════════════════════════════

def simulation_step():
    """
    One simulation tick: update all 3 drones.

      1. Parse keyboard input → drone 0 force (if in manual mode)
      2. Run PSO swarm step for each autonomous drone
      3. Compute hybrid APF-PSO-Vortex force for each autonomous drone
      4. Apply physics integration to all drones
      5. Check survivor detection
      6. Update LiDAR (every 4 frames for performance)
    """
    scene, obstacles = _get_scene_and_obstacles()
    lidar_sensor     = _get_lidar()
    dt               = st.session_state.dt * st.session_state.sim_speed

    # Increment frame counter
    st.session_state.frame_count += 1

    # ── Start mission timer on first run ──────────────────────────────────────
    if st.session_state.mission_start_time is None:
        st.session_state.mission_start_time = time.time()

    drones   = st.session_state.drones
    physics_ = st.session_state.physics
    pso_list = st.session_state.pso

    # ── Collect other-drone positions for separation ──────────────────────────
    all_positions = [d.position for d in drones]

    # ── Per-drone update ──────────────────────────────────────────────────────
    for idx, (drone, phys, pso) in enumerate(zip(drones, physics_, pso_list)):

        # ── Manual drone (idx 0 when manual_mode is on) ───────────────────────
        if idx == 0 and st.session_state.manual_mode:
            nav_force = compute_manual_force(
                st.session_state.active_keys,
                drone.velocity,
                drone.yaw,
            )
            drone.mode = "manual"
            force_components = None

        else:
            # ── Autonomous: PSO step + Hybrid APF-PSO-Vortex ──────────────────
            drone.mode = "autonomous"

            # PSO cross-swarm knowledge sharing (global best from swarm 0)
            shared_best = pso_list[0].g_best if idx > 0 else None
            pso.step(shared_g_best=shared_best)

            # Select current search target (nearest unvisited survivor)
            target_idx = select_target(
                drone.position,
                scene.survivors,
                st.session_state.found_idx,
            )
            drone.target_idx = target_idx

            # If all survivors found, return to base
            if target_idx is None:
                goal = np.array([42.0, 42.0, 8.0])   # Rescue base
            else:
                goal = scene.survivors[target_idx].position.copy()
                goal[2] = max(goal[2], 8.0)   # Approach from above

            other_pos = [all_positions[j] for j in range(3) if j != idx]
            nav_force, force_comps = compute_hybrid_force(
                pos          = drone.position,
                velocity     = drone.velocity,
                goal         = goal,
                obstacles    = obstacles,
                other_drones = other_pos,
                pso_g_best   = pso.g_best,
                alpha        = 0.55,
                beta         = 0.30,
                gamma        = 0.15,
            )
            force_components = force_comps

        # ── Physics step ──────────────────────────────────────────────────────
        phys.step(nav_force, dt=dt, force_components=force_components)

    # ── Survivor detection ────────────────────────────────────────────────────
    newly_found = []
    for surv in scene.survivors:
        if surv.idx in st.session_state.found_idx:
            continue
        for drone in drones:
            dist = norm(drone.position - surv.position)
            if dist < DETECTION_RADIUS:
                st.session_state.found_idx.add(surv.idx)
                scene.survivors[surv.idx].found = True
                newly_found.append(surv.name)
                # Inform all PSO swarms
                for pso in pso_list:
                    pso.mark_visited(surv.idx)
                break

    if newly_found:
        for name in newly_found:
            st.session_state.alerts.insert(0,
                f"✅  SURVIVOR {name} LOCATED  —  Frame {st.session_state.frame_count}")
        st.session_state.alerts = st.session_state.alerts[:6]

    # ── LiDAR update (every 4 frames to save compute) ─────────────────────────
    st.session_state.lidar_frame_counter += 1
    if st.session_state.lidar_frame_counter % 4 == 0:
        rng = np.random.RandomState(st.session_state.frame_count)
        pts, dists = lidar_sensor.scan(
            drones[0].position, drones[0].yaw, scene, rng=rng
        )
        # Add ground returns for visual richness
        g_pts, g_dists = lidar_sensor.get_ground_returns(drones[0].position, n_rays=80, rng=rng)
        if len(g_pts) > 0 and len(pts) > 0:
            pts   = np.vstack([pts, g_pts])
            dists = np.concatenate([dists, g_dists])
        st.session_state.lidar_pts   = pts
        st.session_state.lidar_dists = dists


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Mission Control
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        '<div class="app-title">🛩  SAR UAV SIM</div>'
        '<div class="app-sub">APF · PSO · VORTEX · HYBRID</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ── Mission control ───────────────────────────────────────────────────────
    st.markdown('<div class="panel-title">▶ MISSION CONTROL</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🚀  LAUNCH" if not st.session_state.sim_running else "⏸  PAUSE"):
            st.session_state.sim_running = not st.session_state.sim_running
    with col_b:
        if st.button("⟳  RESET"):
            for key in ["drones","physics","pso","found_idx","sim_running",
                        "frame_count","lidar_pts","lidar_dists","alerts",
                        "active_keys","mission_start_time","lidar_frame_counter"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    run_status = "● RUNNING" if st.session_state.sim_running else "◼ STOPPED"
    status_class = "badge-ok" if st.session_state.sim_running else "badge-warn"
    st.markdown(f'<span class="badge {status_class}">{run_status}</span>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Simulation settings ───────────────────────────────────────────────────
    st.markdown('<div class="panel-title">▶ SIM SETTINGS</div>', unsafe_allow_html=True)
    st.session_state.sim_speed = st.slider(
        "Sim Speed ×", 0.25, 4.0, st.session_state.sim_speed, 0.25
    )
    st.session_state.dt = st.select_slider(
        "Physics dt (s)", [0.04, 0.06, 0.08, 0.10, 0.15], value=0.08
    )

    scene_seed = st.number_input("Environment Seed", 0, 9999, 42, step=1)
    show_trails   = st.toggle("Show Flight Trails",  value=True)
    show_nfz      = st.toggle("Show PSO Particles",  value=True)
    show_lidar    = st.toggle("Show LiDAR Cloud",    value=True)

    st.markdown("---")

    # ── Drone control ─────────────────────────────────────────────────────────
    st.markdown('<div class="panel-title">▶ DRONE CONTROL</div>', unsafe_allow_html=True)

    st.session_state.manual_mode = st.toggle(
        "🎮  Manual Control — UAV-1",
        value=st.session_state.manual_mode
    )

    if st.session_state.manual_mode:
        st.markdown(
            '<div class="badge badge-info">UAV-1 IN MANUAL MODE</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="badge badge-ok">UAV-1 AUTONOMOUS</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Algorithm weights ─────────────────────────────────────────────────────
    st.markdown('<div class="panel-title">▶ ALGORITHM WEIGHTS</div>', unsafe_allow_html=True)
    alpha_apf = st.slider("α  APF Weight",  0.0, 1.0, 0.55, 0.05)
    beta_pso  = st.slider("β  PSO Weight",  0.0, 1.0, 0.30, 0.05)
    gamma_sep = st.slider("γ  Sep Weight",  0.0, 1.0, 0.15, 0.05)

    st.markdown("---")

    # ── Alerts log ────────────────────────────────────────────────────────────
    st.markdown('<div class="panel-title">▶ MISSION LOG</div>', unsafe_allow_html=True)
    for alert in st.session_state.alerts:
        st.markdown(f'<div class="alert-box alert-found">{alert}</div>',
                    unsafe_allow_html=True)
    if not st.session_state.alerts:
        st.markdown(
            '<div class="alert-box alert-warn">◌  Awaiting contact…</div>',
            unsafe_allow_html=True
        )

    # ── Mission progress ──────────────────────────────────────────────────────
    st.markdown("---")
    scene_ref, _ = _get_scene_and_obstacles()
    n_total  = len(scene_ref.survivors)
    n_found  = len(st.session_state.found_idx)
    st.markdown(
        f'<div class="panel-title">▶ MISSION PROGRESS  '
        f'<span style="color:#00C8FF">{n_found}/{n_total}</span></div>',
        unsafe_allow_html=True
    )
    st.progress(n_found / max(n_total, 1))

    elapsed = (
        int(time.time() - st.session_state.mission_start_time)
        if st.session_state.mission_start_time else 0
    )
    st.markdown(
        f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.68rem;'
        f'color:#2A5A80;">ELAPSED: {elapsed//60:02d}:{elapsed%60:02d}  |  '
        f'FRAME: {st.session_state.frame_count:05d}</div>',
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  KEYBOARD LISTENER INJECTION + HIDDEN INPUT READER
# ═══════════════════════════════════════════════════════════════════════════════

# Inject the JS keyboard listener (runs in browser iframe)
components.html(KEYBOARD_JS, height=0)

# Hidden text input that JS writes into (placeholder acts as an ID)
raw_keys = st.text_input(
    label="keys",
    value="",
    key="raw_key_input",
    label_visibility="hidden",
    placeholder="__sar_keys__",
)

# Parse the key string into a set
if raw_keys:
    st.session_state.active_keys = set(k.strip() for k in raw_keys.split(",") if k.strip())
else:
    st.session_state.active_keys = set()


# ═══════════════════════════════════════════════════════════════════════════════
#  ON-SCREEN D-PAD (reliable fallback + complement to keyboard)
# ═══════════════════════════════════════════════════════════════════════════════

def render_dpad():
    """
    Render the on-screen D-pad controller using styled Streamlit buttons.
    Each button adds/removes a virtual key from active_keys.
    """
    if not st.session_state.manual_mode:
        return

    active = st.session_state.active_keys

    row0 = st.columns([1, 1, 1, 1, 1, 1, 1])
    with row0[0]:
        st.markdown("**🎮 D-PAD**", help="Click buttons OR use WASD/Arrow keys")

    # Row 1: Ascend + Forward
    r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns([1, 1, 1, 1, 1])
    with r1c2:
        if st.button("▲\nASC", key="dp_asc", help="Ascend [R/Space]"):
            st.session_state.active_keys ^= {'r'}
    with r1c3:
        if st.button("↑\nFWD", key="dp_fwd", help="Forward [W/↑]"):
            st.session_state.active_keys ^= {'w'}
    with r1c4:
        if st.button("↺\nYAW L", key="dp_yawr", help="Yaw Left [Q]"):
            st.session_state.active_keys ^= {'q'}

    # Row 2: Left + Hold + Right
    r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns([1, 1, 1, 1, 1])
    with r2c1:
        pass
    with r2c2:
        if st.button("←\nLFT", key="dp_left", help="Strafe Left [A/←]"):
            st.session_state.active_keys ^= {'a'}
    with r2c3:
        if st.button("■\nHLD", key="dp_hold", help="Hold Position [H]"):
            st.session_state.active_keys = set()
    with r2c4:
        if st.button("→\nRGT", key="dp_right", help="Strafe Right [D/→]"):
            st.session_state.active_keys ^= {'d'}
    with r2c5:
        pass

    # Row 3: Descend + Backward
    r3c1, r3c2, r3c3, r3c4, r3c5 = st.columns([1, 1, 1, 1, 1])
    with r3c2:
        if st.button("▼\nDSC", key="dp_dsc", help="Descend [F/Shift]"):
            st.session_state.active_keys ^= {'f'}
    with r3c3:
        if st.button("↓\nBCK", key="dp_bck", help="Backward [S/↓]"):
            st.session_state.active_keys ^= {'s'}
    with r3c4:
        if st.button("↻\nYAW R", key="dp_yawr2", help="Yaw Right [E]"):
            st.session_state.active_keys ^= {'e'}

    # Key state indicator
    if active:
        keys_str = " + ".join(k.upper()[:3] for k in sorted(active))
        st.markdown(
            f'<div class="badge badge-info">KEYS: {keys_str}</div>',
            unsafe_allow_html=True
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN SIMULATION STEP (if running)
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.sim_running:
    simulation_step()


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER ROW — Title + HUD Metrics
# ═══════════════════════════════════════════════════════════════════════════════

header_left, header_right = st.columns([2, 3])

with header_left:
    st.markdown(
        '<div class="app-title">🛩 SAR MULTI-UAV SIMULATION</div>'
        '<div class="app-sub">FLOOD RESCUE · APF-PSO-VORTEX HYBRID NAVIGATION · 3-DRONE FLEET</div>',
        unsafe_allow_html=True
    )

with header_right:
    scene_ref, _ = _get_scene_and_obstacles()
    drones_now   = st.session_state.drones
    mc = st.columns(6)
    mc[0].metric("UAV-1 Alt",  f"{drones_now[0].altitude:.1f}m",
                 f"{drones_now[0].speed:.1f}m/s")
    mc[1].metric("UAV-2 Alt",  f"{drones_now[1].altitude:.1f}m",
                 f"{drones_now[1].speed:.1f}m/s")
    mc[2].metric("UAV-3 Alt",  f"{drones_now[2].altitude:.1f}m",
                 f"{drones_now[2].speed:.1f}m/s")
    mc[3].metric("Survivors",
                 f"{len(st.session_state.found_idx)}/{len(scene_ref.survivors)}",
                 delta=f"+{len(st.session_state.found_idx)}" if st.session_state.found_idx else None)
    mc[4].metric("UAV-1 Bat",  f"{drones_now[0].battery:.0f}%")
    mc[5].metric("Mode",
                 "MANUAL" if st.session_state.manual_mode else "AUTO",
                 delta="UAV-1")

st.markdown("<hr style='margin:4px 0 6px 0;'>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  THREE-PANEL LAYOUT
#  Left (60%): Main 3D Scene + D-Pad
#  Right (40%): Top — Tactical Map | Bottom — Sensor Panel
# ═══════════════════════════════════════════════════════════════════════════════

col_left, col_right = st.columns([3, 2], gap="small")

# ── Retrieve current state ─────────────────────────────────────────────────────
scene_now, obstacles_now = _get_scene_and_obstacles()
drones_now   = st.session_state.drones
pso_now      = st.session_state.pso
found_now    = st.session_state.found_idx


# ── LEFT PANEL: Main 3D Scene ──────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="panel-title">◈ MAIN 3D SCENE — FLOOD SAR ENVIRONMENT</div>',
                unsafe_allow_html=True)

    main_fig = build_main_scene(
        scene    = scene_now,
        drones   = drones_now,
        found_idx= found_now,
        cam_eye  = st.session_state.cam_eye,
    )
    st.plotly_chart(main_fig, use_container_width=True, key="main_scene")

    st.markdown("<hr style='margin:4px 0;'>", unsafe_allow_html=True)

    # ── D-Pad + telemetry row ──────────────────────────────────────────────────
    dp_col, telem_col = st.columns([2, 3])

    with dp_col:
        if st.session_state.manual_mode:
            st.markdown('<div class="panel-title">🎮 MANUAL CONTROL — UAV-1</div>',
                        unsafe_allow_html=True)
            render_dpad()
        else:
            st.markdown('<div class="panel-title">⚙ ALGORITHM WEIGHTS</div>',
                        unsafe_allow_html=True)
            # Show force magnitude bars for drone 0
            d0 = drones_now[0]
            for fname, fvec, col_ in [
                ("F_att",  d0.f_apf_att, "#00FF88"),
                ("F_rep",  d0.f_apf_rep, "#FF4040"),
                ("F_vort", d0.f_vortex,  "#FF8C00"),
                ("F_pso",  d0.f_pso,     "#A0A0FF"),
            ]:
                mag = float(norm(fvec))
                bar = int(min(mag / 12.0 * 100, 100))
                st.markdown(
                    f'<div style="font-family:\'Share Tech Mono\',monospace;'
                    f'font-size:0.68rem;color:{col_};">'
                    f'{fname}: {mag:5.2f}N '
                    f'<span style="background:{col_};width:{bar}px;height:6px;'
                    f'display:inline-block;border-radius:2px;vertical-align:middle;"></span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    with telem_col:
        st.markdown('<div class="panel-title">📡 TELEMETRY</div>', unsafe_allow_html=True)
        # Per-drone telemetry table
        telem_cols = st.columns(3)
        drone_icons = ["🔵", "🟢", "🟡"]
        for d_idx, (dc, icon) in enumerate(zip(drones_now, drone_icons)):
            with telem_cols[d_idx]:
                target_str = (
                    f"→ SUR-{dc.target_idx+1:02d}"
                    if dc.target_idx is not None else "→ BASE"
                )
                mode_str = "MAN" if dc.mode == "manual" else "APF"
                bat_color = "#00FF80" if dc.battery > 40 else ("#FFA020" if dc.battery > 15 else "#FF3040")
                st.markdown(
                    f'<div style="background:rgba(8,20,40,0.9);border:1px solid #0A3060;'
                    f'border-radius:3px;padding:6px;font-family:\'Share Tech Mono\',monospace;'
                    f'font-size:0.68rem;">'
                    f'<div style="color:#00C8FF;">{icon} UAV-{d_idx+1} [{mode_str}]</div>'
                    f'<div style="color:#7BAFC8;">X: {dc.position[0]:+6.1f}m</div>'
                    f'<div style="color:#7BAFC8;">Y: {dc.position[1]:+6.1f}m</div>'
                    f'<div style="color:#7BAFC8;">Z: {dc.position[2]:5.1f}m</div>'
                    f'<div style="color:#7BAFC8;">V: {dc.speed:5.2f}m/s</div>'
                    f'<div style="color:{bat_color};">⚡ {dc.battery:4.0f}%</div>'
                    f'<div style="color:#FFA020;">{target_str}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ── RIGHT PANEL: Top = Tactical Map, Bottom = Sensor ──────────────────────────
with col_right:

    # ── TOP RIGHT: Tactical Map (top-down view) ────────────────────────────────
    st.markdown('<div class="panel-title">◈ TACTICAL MAP — TOP-DOWN / FPV</div>',
                unsafe_allow_html=True)

    tac_fig = build_tactical_map(
        scene      = scene_now,
        drones     = drones_now,
        found_idx  = found_now,
        pso_states = pso_now if show_nfz else [],
        obstacles  = obstacles_now,
    )
    st.plotly_chart(tac_fig, use_container_width=True, key="tac_map")

    # ── BOTTOM RIGHT: Sensor Panel (LiDAR + APF/Vortex Field) ─────────────────
    st.markdown('<div class="panel-title">◈ SENSOR DATA — LiDAR · APF · VORTEX FIELD</div>',
                unsafe_allow_html=True)

    sensor_fig = build_sensor_panel(
        drones     = drones_now,
        lidar_pts  = st.session_state.lidar_pts  if show_lidar else None,
        lidar_dists= st.session_state.lidar_dists if show_lidar else None,
        obstacles  = obstacles_now,
        found_idx  = found_now,
        scene      = scene_now,
    )
    st.plotly_chart(sensor_fig, use_container_width=True, key="sensor_panel")


# ═══════════════════════════════════════════════════════════════════════════════
#  KEYBOARD LEGEND (full-width, below panels)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("<hr style='margin:4px 0;'>", unsafe_allow_html=True)

legend_html = """
<div style="
  font-family:'Share Tech Mono',monospace;
  font-size:0.65rem;
  color:#1A4870;
  display:flex;
  gap:20px;
  flex-wrap:wrap;
  padding:4px 0;
">
  <span>🎮 <span style="color:#00C8FF">UAV-1 MANUAL:</span></span>
  <span><kbd style="background:#0A2040;border:1px solid #1A4880;border-radius:2px;padding:1px 4px;color:#00C8FF;">W</kbd> / <kbd style="background:#0A2040;border:1px solid #1A4880;border-radius:2px;padding:1px 4px;color:#00C8FF;">↑</kbd> Forward</span>
  <span><kbd style="background:#0A2040;border:1px solid #1A4880;border-radius:2px;padding:1px 4px;color:#00C8FF;">S</kbd> / <kbd style="background:#0A2040;border:1px solid #1A4880;border-radius:2px;padding:1px 4px;color:#00C8FF;">↓</kbd> Backward</span>
  <span><kbd style="background:#0A2040;border:1px solid #1A4880;border-radius:2px;padding:1px 4px;color:#00C8FF;">A</kbd> / <kbd style="background:#0A2040;border:1px solid #1A4880;border-radius:2px;padding:1px 4px;color:#00C8FF;">←</kbd> Left</span>
  <span><kbd style="background:#0A2040;border:1px solid #1A4880;border-radius:2px;padding:1px 4px;color:#00C8FF;">D</kbd> / <kbd style="background:#0A2040;border:1px solid #1A4880;border-radius:2px;padding:1px 4px;color:#00C8FF;">→</kbd> Right</span>
  <span><kbd style="background:#0A2040;border:1px solid #1A4880;border-radius:2px;padding:1px 4px;color:#00C8FF;">R</kbd> / <kbd style="background:#0A2040;border:1px solid #1A4880;border-radius:2px;padding:1px 4px;color:#00C8FF;">SPACE</kbd> Ascend</span>
  <span><kbd style="background:#0A2040;border:1px solid #1A4880;border-radius:2px;padding:1px 4px;color:#00C8FF;">F</kbd> / <kbd style="background:#0A2040;border:1px solid #1A4880;border-radius:2px;padding:1px 4px;color:#00C8FF;">SHIFT</kbd> Descend</span>
  <span><kbd style="background:#0A2040;border:1px solid #1A4880;border-radius:2px;padding:1px 4px;color:#00C8FF;">Q</kbd>/<kbd style="background:#0A2040;border:1px solid #1A4880;border-radius:2px;padding:1px 4px;color:#00C8FF;">E</kbd> Yaw</span>
  <span><kbd style="background:#0A2040;border:1px solid #1A4880;border-radius:2px;padding:1px 4px;color:#00C8FF;">H</kbd> Hold</span>
  <span style="margin-left:auto;color:#0A3060;">UAV-2 &amp; UAV-3 run APF-PSO-VORTEX autonomously</span>
</div>
"""
st.markdown(legend_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTO-REFRESH — Drive the simulation loop
#  Uses time.sleep + st.rerun() — standard Streamlit simulation pattern.
#  Refresh interval is set by sim_speed (0.25x → ~200ms, 4x → ~25ms).
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.sim_running:
    base_interval = 0.08    # seconds at 1× speed
    interval = base_interval / max(st.session_state.sim_speed, 0.25)
    time.sleep(max(interval, 0.04))   # floor at 25 FPS
    st.rerun()
