import math
from typing import Dict, List
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Ultimate Underground Simulator", layout="wide")

# =========================================================
# Constants & Scenarios
# =========================================================
OUTDOOR_CO2_PPM = 420.0
OUTDOOR_AH_G_M3 = 8.0 
OUTDOOR_TEMP_C = 20.0 
OUTDOOR_O2_PCT = 21.0

MAX_HEATMAP_CELLS = 12000

EXTERNAL_SCENARIOS = {
    "Normal": {"co2": 420.0, "toxin": 0, "radiation": 0.1, "label": "🟢 평상시"},
    "Wildfire": {"co2": 900.0, "toxin": 200, "radiation": 0.2, "label": "🔥 화재 (연기)"},
    "Chemical": {"co2": 420.0, "toxin": 800, "radiation": 0.5, "label": "☣️ 화학 사고"},
    "Fallout": {"co2": 420.0, "toxin": 50, "radiation": 500.0, "label": "☢️ 방사능 낙진"}
}

EQUIPMENT_TYPES = {
    "Supply": {"color": "rgba(0,140,255,0.95)", "symbol": "S", "radius_m": 8.0, "ppm_red": 120.0, "ah_red": 3.0, "temp_red": 3.0, "label": "🟦"},
    "Exhaust": {"color": "rgba(255,90,90,0.95)", "symbol": "E", "radius_m": 10.0, "ppm_red": 150.0, "ah_red": 4.0, "temp_red": 2.0, "label": "🟥"},
    "Purifier": {"color": "rgba(0,180,120,0.95)", "symbol": "P", "radius_m": 6.0, "ppm_red": 90.0, "ah_red": 0.5, "temp_red": -0.5, "label": "🟩"}, 
}

if "equipment_map" not in st.session_state:
    st.session_state.equipment_map = {}

# =========================================================
# Advanced Physics Engine
# =========================================================
def compute_transient_baseline(room_w, room_d, room_h, ach, n_st, n_si, n_ly, t, panic_mode, initial_blackout_t, depth_m, grid_destroyed, ext_cfg, filter_on):
    vol = room_w * room_d * room_h
    area = 2 * (room_w*room_d + room_w*room_h + room_d*room_h) 
    ground_t = 20.0 - min(depth_m, 10.0) * 0.5 
    
    # 전력망 및 필터 부하 계산
    filter_load = 1.3 if filter_on else 1.0
    power_multiplier = (1.0 + (depth_m / 25.0)) * filter_load
    
    if grid_destroyed:
        actual_blackout_t = initial_blackout_t / power_multiplier
    else:
        actual_blackout_t = float('inf')

    multiplier = 3.0 if panic_mode else 1.0
    g_co2 = (n_st*0.021 + n_si*0.018 + n_ly*0.015) * multiplier
    g_ah = (n_st*75.0 + n_si*50.0 + n_ly*40.0) * multiplier
    g_heat_w = (n_st*75.0 + n_si*60.0 + n_ly*45.0) * multiplier
    
    heat_rate_c_per_h = (g_heat_w * 3600) / (1200 * vol) 
    k_wall = (2.0 * area * 3600) / (1200 * vol) 

    def get_co2_ah_deltas(current_t, current_ach):
        if current_ach > 0:
            q_vent = current_ach * vol
            decay = (1.0 - math.exp(-current_ach * current_t))
            return (1e6 * g_co2 / q_vent) * decay, (g_ah / q_vent) * decay
        else:
            return (1e6 * g_co2 * current_t)/vol, (g_ah * current_t)/vol

    def get_temp(start_t, duration, current_ach):
        eff_ach = current_ach + k_wall
        if eff_ach > 0:
            t_eq = (current_ach * OUTDOOR_TEMP_C + k_wall * ground_t + heat_rate_c_per_h) / eff_ach
            return t_eq + (start_t - t_eq) * math.exp(-eff_ach * duration)
        else:
            return start_t + heat_rate_c_per_h * duration

    # 메인 시뮬레이션 루프
    if t <= actual_blackout_t:
        d_c, d_a = get_co2_ah_deltas(t, ach)
        base_t = get_temp(ground_t, t, ach)
        curr_ach = ach
    else:
        fail_c, fail_a = get_co2_ah_deltas(actual_blackout_t, ach) 
        post_c, post_a = get_co2_ah_deltas(t - actual_blackout_t, 0.0) 
        d_c, d_a = fail_c + post_c, fail_a + post_a
        temp_at_fail = get_temp(ground_t, actual_blackout_t, ach)
        base_t = get_temp(temp_at_fail, t - actual_blackout_t, 0.0)
        curr_ach = 0.0

    base_c = ext_cfg["co2"] + d_c
    base_a = OUTDOOR_AH_G_M3 + d_a
    base_o2 = OUTDOOR_O2_PCT - (d_c * 1.2 / 10000.0) 
    
    # 독소 및 방사능 계산 (라돈 포함)
    filter_eff = 0.98 if filter_on else 0.0
    if curr_ach > 0:
        res_toxin = ext_cfg["toxin"] * (1 - filter_eff) * (1 - math.exp(-curr_ach * t))
        res_rad = ext_cfg["radiation"] * (1 - filter_eff) * (1 - math.exp(-curr_ach * t))
    else:
        res_toxin = 0.0
        radon_gen = (depth_m * 0.2) # 지표면 라돈 발생 가정
        res_rad = radon_gen * t 

    sat_ah_wall = 5.018 + 0.3232*ground_t + 0.0081847*(ground_t**2) + 0.00031243*(ground_t**3)
    condensation = base_a > sat_ah_wall

    return vol, base_c, base_a, base_t, base_o2, actual_blackout_t, condensation, res_toxin, res_rad

# --- Helper Functions (Grid, Charts, Points) ---
@st.cache_data(show_spinner=False)
def get_grid(room_w, room_d):
    step = max(1.0, math.sqrt(room_w * room_d / MAX_HEATMAP_CELLS))
    nx, ny = int(room_w/step)+1, int(room_d/step)+1
    return np.meshgrid(np.linspace(0, room_w, nx), np.linspace(0, room_d, ny)), step

def make_trend_charts(room_w, room_d, room_h, ach, n_st, n_si, n_ly, current_t, panic_mode, initial_blackout_t, depth_m, grid_destroyed, ext_cfg, filter_on):
    times = np.linspace(0, 24, 49)
    co2_v, rad_v, tox_v, temp_v = [], [], [], []
    act_blackout = float('inf')
    for t in times:
        _, c, _, temp, _, act_b, _, tox, rad = compute_transient_baseline(room_w, room_d, room_h, ach, n_st, n_si, n_ly, t, panic_mode, initial_blackout_t, depth_m, grid_destroyed, ext_cfg, filter_on)
        co2_v.append(c); rad_v.append(rad); tox_v.append(tox); temp_v.append(temp)
        act_blackout = act_b
        
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=times, y=co2_v, name="CO2 (ppm)", yaxis="y1", line=dict(color="orangered", width=3)))
    fig1.add_trace(go.Scatter(x=times, y=rad_v, name="Rad (mSv/h)", yaxis="y2", line=dict(color="purple", width=3, dash='dot')))
    fig1.add_vline(x=current_t, line_width=2, line_dash="dash")
    fig1.update_layout(title="Air & Radiation", height=250, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified",
                       yaxis=dict(title="CO2 (ppm)"), yaxis2=dict(title="Rad", overlaying="y", side="right"))
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=times, y=temp_v, name="Temp (°C)", yaxis="y1", line=dict(color="firebrick", width=3)))
    fig2.add_trace(go.Scatter(x=times, y=tox_v, name="Toxin (ppm)", yaxis="y2", line=dict(color="lime", width=3, dash='dot')))
    fig2.add_vline(x=current_t, line_width=2, line_dash="dash")
    fig2.update_layout(title="Thermal & Toxicity", height=250, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified",
                       yaxis=dict(title="Temp (°C)"), yaxis2=dict(title="Toxin", overlaying="y", side="right"))
    return fig1, fig2

def equipment_list_from_map(equipment_map, room_w, room_d):
    eqs = []
    dx, dy = room_w / 10, room_d / 10
    for (ix, iy), eq_type in equipment_map.items():
        if eq_type in EQUIPMENT_TYPES:
            eq = EQUIPMENT_TYPES[eq_type]
            eqs.append({"ix": ix, "iy": iy, "x": (ix*dx)-0.5, "y": (iy*dy)-0.5, "w": 1.0, "d": 1.0, "type": eq_type, **eq})
    return eqs

def get_non_overlapping_points(room_w, room_d, n_st, n_si, n_ly, seed=42):
    total_n = n_st + n_si + n_ly
    if total_n == 0: return [], [], [], [], [], []
    nx = max(1, int(math.ceil(math.sqrt(total_n * (room_w / room_d)))))
    ny = max(1, int(math.ceil(total_n / nx)))
    while nx * ny < total_n:
        if nx / room_w < ny / room_d: nx += 1
        else: ny += 1
    dx, dy = room_w / nx, room_d / ny
    points = [(ix * dx + dx / 2, iy * dy + dy / 2) for ix in range(nx) for iy in range(ny)]
    rng = np.random.default_rng(seed)
    rng.shuffle(points)
    pts = points[:total_n]
    fx = [p[0] + jx for p, jx in zip(pts, rng.uniform(-dx/3, dx/3, total_n))]
    fy = [p[1] + jy for p, jy in zip(pts, rng.uniform(-dy/3, dy/3, total_n))]
    return fx[:n_st], fy[:n_st], fx[n_st:n_st+n_si], fy[n_st:n_st+n_si], fx[n_st+n_si:], fy[n_st+n_si:]

def make_heatmap_fig(room_w, room_d, X, Y, Z, eqs, t, n_st, n_si, n_ly, title, scale, zmin, zmax, cbtitle):
    fig = go.Figure(go.Heatmap(x=X[0], y=Y[:,0], z=Z, colorscale=scale, zmin=zmin, zmax=zmax, colorbar=dict(title=cbtitle), zsmooth="best"))
    for eq in eqs:
        fig.add_shape(type="rect", x0=eq["x"]+0.15, y0=eq["y"]+0.15, x1=eq["x"]+0.85, y1=eq["y"]+0.85, line=dict(color="black", width=1), fillcolor=eq["color"])
        fig.add_trace(go.Scatter(x=[eq["x"]+0.5], y=[eq["y"]+0.5], mode="text", text=[eq["symbol"]], textfont=dict(color="white"), showlegend=False))
    st_x, st_y, si_x, si_y, ly_x, ly_y = get_non_overlapping_points(room_w, room_d, n_st, n_si, n_ly, seed=99)
    if n_st > 0: fig.add_trace(go.Scatter(x=st_x, y=st_y, mode="markers", name=f"Standing", marker=dict(size=6, color="black")))
    fig.update_layout(width=900, height=500, title=f"{title} (T={t:.1f}h)")
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    return fig

# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("🏢 Space & Geometry")
depth_m = st.sidebar.slider("Underground Depth (m)", 0.0, 100.0, 20.0)
room_w = st.sidebar.slider("Width (m)", 5.0, 500.0, 100.0)
room_d = st.sidebar.slider("Depth (m)", 5.0, 500.0, 50.0)
room_h = st.sidebar.slider("Height (m)", 2.0, 6.0, 3.0)

st.sidebar.header("🌍 External Threats")
use_external = st.sidebar.toggle("Enable Threats", value=True)
env_mode = st.sidebar.selectbox("Condition", list(EXTERNAL_SCENARIOS.keys())) if use_external else "Normal"
ext_cfg = EXTERNAL_SCENARIOS[env_mode]
filter_on = st.sidebar.toggle("🛡️ HEPA Filter", value=False)

st.sidebar.header("🚨 Scenario Setup")
grid_destroyed = st.sidebar.checkbox("💥 Blackout Mode", value=True)
initial_blackout_t = st.sidebar.slider("⚡ Fuel Time (h)", 0.0, 24.0, 24.0) if grid_destroyed else 24.0
ach = st.sidebar.slider("Ventilation (ACH)", 0.0, 15.0, 2.0)
panic_mode = st.sidebar.checkbox("😱 Panic Mode")

st.sidebar.header("👥 Population")
n_st = st.sidebar.number_input("Standing", 0, 5000, 100)
n_si = st.sidebar.number_input("Sitting", 0, 5000, 50)
n_ly = st.sidebar.number_input("Lying", 0, 5000, 10)

tool = st.sidebar.radio("Tool", ["Supply", "Exhaust", "Purifier", "Eraser"])

# =========================================================
# Main UI
# =========================================================
st.caption("@sicsmria")
st.title("Ultimate Underground Simulator (Peacetime vs Wartime)")

elapsed_t = st.select_slider("🕒 Time Machine", options=np.round(np.linspace(0, 24, 241), 1), value=2.0)

vol, base_c, base_a, base_t, base_o2, act_blackout, is_condensation, res_toxin, res_rad = compute_transient_baseline(
    room_w, room_d, room_h, ach, n_st, n_si, n_ly, elapsed_t, panic_mode, initial_blackout_t, depth_m, grid_destroyed, ext_cfg, filter_on
)

# 알림창
if res_rad > 100: st.error(f"☢️ 방사능 수치 위험! ({res_rad:.1f} mSv/h)")
if res_toxin > 100: st.error(f"☣️ 독소 유입 중! ({res_toxin:.1f} ppm)")
if elapsed_t > act_blackout: st.error("⚡ 전력 고갈: 모든 환기 시스템 정지")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("CO2 Level", f"{base_c:.0f} ppm")
m2.metric("Oxygen", f"{base_o2:.1f} %")
m3.metric("Radiation", f"{res_rad:.1f}")
m4.metric("Toxin", f"{res_toxin:.1f}")
m5.metric("Temp", f"{base_t:.1f} °C")

# 차트 및 히트맵
fig_air, fig_therm = make_trend_charts(room_w, room_d, room_h, ach, n_st, n_si, n_ly, elapsed_t, panic_mode, initial_blackout_t, depth_m, grid_destroyed, ext_cfg, filter_on)
c1, c2 = st.columns(2)
c1.plotly_chart(fig_air, use_container_width=True)
c2.plotly_chart(fig_therm, use_container_width=True)

# 설비 로직
eqs = equipment_list_from_map(st.session_state.equipment_map, room_w, room_d)
(X, Y), g_step = get_grid(room_w, room_d)
Z_c = np.ones_like(X)*base_c

t1, t2 = st.tabs(["🔥 CO2 Heatmap", "📍 Editor"])
with t1: st.plotly_chart(make_heatmap_fig(room_w, room_d, X, Y, Z_c, eqs, elapsed_t, n_st, n_si, n_ly, "CO2 Distribution", "Turbo", 400, 3000, "ppm"))
with t2:
    nx, ny = 11, 11
    for iy in range(ny):
        cols = st.columns(nx)
        for ix in range(nx):
            et = st.session_state.equipment_map.get((ix, iy))
            lbl = EQUIPMENT_TYPES[et]["label"] if et in EQUIPMENT_TYPES else "·"
            if cols[ix].button(lbl, key=f"b_{ix}_{iy}"):
                if tool == "Eraser": st.session_state.equipment_map.pop((ix, iy), None)
                else: st.session_state.equipment_map[(ix, iy)] = tool
                st.rerun()
