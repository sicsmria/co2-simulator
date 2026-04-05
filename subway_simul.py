import math
from typing import Dict, List
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="CO2 & Humidity Time-Traveler", layout="wide")

# =========================================================
# Constants
# =========================================================
OUTDOOR_CO2_PPM = 420.0
OUTDOOR_AH_G_M3 = 8.0 
MAX_HEATMAP_CELLS = 12000

CO2_GEN_M3_PER_H = {"standing": 0.021, "sitting": 0.018, "lying": 0.015}
MOISTURE_GEN_G_PER_H = {"standing": 75.0, "sitting": 50.0, "lying": 40.0}

EQUIPMENT_TYPES = {
    "Supply": {"color": "rgba(0,140,255,0.95)", "symbol": "S", "radius_m": 8.0, "ppm_reduction": 120.0, "ah_reduction": 3.0, "label": "🟦"},
    "Exhaust": {"color": "rgba(255,90,90,0.95)", "symbol": "E", "radius_m": 10.0, "ppm_reduction": 150.0, "ah_reduction": 4.0, "label": "🟥"},
    "Purifier": {"color": "rgba(0,180,120,0.95)", "symbol": "P", "radius_m": 6.0, "ppm_reduction": 90.0, "ah_reduction": 0.5, "label": "🟩"},
}

EMPTY_CELL_LABEL = "·"

if "equipment_map" not in st.session_state:
    st.session_state.equipment_map = {}

# =========================================================
# Helpers
# =========================================================
def compute_transient_baseline(room_w, room_d, room_h, ach, n_st, n_si, n_ly, t, panic_mode):
    vol = room_w * room_d * room_h
    q_vent = max(ach * vol, 0.001)
    
    # 패닉 모드 시 호흡량, 땀 배출량 3배 증가
    multiplier = 3.0 if panic_mode else 1.0
    
    g_co2 = (n_st*0.021 + n_si*0.018 + n_ly*0.015) * multiplier
    g_ah = (n_st*75.0 + n_si*50.0 + n_ly*40.0) * multiplier

    if ach > 0:
        decay = (1.0 - math.exp(-ach * t))
        c_delta = (1e6 * g_co2 / q_vent) * decay
        a_delta = (g_ah / q_vent) * decay
    else:
        c_delta = (1e6 * g_co2 * t) / max(vol, 1e-9)
        a_delta = (g_ah * t) / max(vol, 1e-9)
    return vol, q_vent, OUTDOOR_CO2_PPM + c_delta, OUTDOOR_AH_G_M3 + a_delta

@st.cache_data(show_spinner=False)
def get_grid(room_w, room_d):
    room_area = room_w * room_d
    step = max(1.0, math.sqrt(room_area / MAX_HEATMAP_CELLS))
    nx, ny = int(room_w/step)+1, int(room_d/step)+1
    xs, ys = np.linspace(0, room_w, nx), np.linspace(0, room_d, ny)
    return np.meshgrid(xs, ys), step

def make_trend_chart(room_w, room_d, room_h, ach, n_st, n_si, n_ly, current_t, panic_mode):
    times = np.linspace(0, 24, 49) # 0.5시간 단위
    co2_vals, ah_vals = [], []
    for t in times:
        _, _, c, a = compute_transient_baseline(room_w, room_d, room_h, ach, n_st, n_si, n_ly, t, panic_mode)
        co2_vals.append(c)
        ah_vals.append(a)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=co2_vals, name="CO2 (ppm)", yaxis="y1", line=dict(color="orangered", width=3)))
    fig.add_trace(go.Scatter(x=times, y=ah_vals, name="AH (g/m³)", yaxis="y2", line=dict(color="royalblue", width=3, dash='dot')))
    
    fig.add_vline(x=current_t, line_width=2, line_dash="dash", line_color="black")
    
    fig.update_layout(
        title="24-Hour Concentration Trend",
        xaxis=dict(title="Hours Elapsed", range=[0, 24]),
        yaxis=dict(title=dict(text="CO2 Concentration (ppm)", font=dict(color="orangered")), tickfont=dict(color="orangered")),
        yaxis2=dict(title=dict(text="Absolute Humidity (g/m³)", font=dict(color="royalblue")), tickfont=dict(color="royalblue"), overlaying="y", side="right"),
        height=300, margin=dict(l=20, r=20, t=50, b=20), hovermode="x unified"
    )
    return fig

def equipment_list_from_map(equipment_map, room_w, room_d):
    eqs = []
    nx, ny = 11, 11
    dx, dy = room_w / 10, room_d / 10
    for (ix, iy), eq_type in equipment_map.items():
        if eq_type in EQUIPMENT_TYPES:
            eq = EQUIPMENT_TYPES[eq_type]
            eqs.append({"ix": ix, "iy": iy, "x": (ix*dx)-0.5, "y": (iy*dy)-0.5, "w": 1.0, "d": 1.0, "type": eq_type, **eq})
    return eqs

def get_non_overlapping_points(room_w, room_d, n_st, n_si, n_ly, seed=42):
    total_n = n_st + n_si + n_ly
    if total_n == 0:
        return [], [], [], [], [], []

    aspect_ratio = room_w / room_d
    nx = max(1, int(math.ceil(math.sqrt(total_n * aspect_ratio))))
    ny = max(1, int(math.ceil(total_n / nx)))

    while nx * ny < total_n:
        if nx / room_w < ny / room_d: nx += 1
        else: ny += 1

    dx = room_w / nx
    dy = room_d / ny

    points = [(ix * dx + dx / 2, iy * dy + dy / 2) for ix in range(nx) for iy in range(ny)]
    rng = np.random.default_rng(seed)
    rng.shuffle(points)
    selected_points = points[:total_n]

    jitter_x = rng.uniform(-dx/3, dx/3, total_n)
    jitter_y = rng.uniform(-dy/3, dy/3, total_n)

    final_x = [p[0] + jx for p, jx in zip(selected_points, jitter_x)]
    final_y = [p[1] + jy for p, jy in zip(selected_points, jitter_y)]

    return (
        final_x[:n_st], final_y[:n_st],
        final_x[n_st:n_st+n_si], final_y[n_st:n_st+n_si],
        final_x[n_st+n_si:], final_y[n_st+n_si:]
    )

def make_heatmap_fig(room_w, room_d, X, Y, Z, eqs, t, n_st, n_si, n_ly, title, scale, zmin, zmax, cbtitle):
    fig = go.Figure(go.Heatmap(x=X[0], y=Y[:,0], z=Z, colorscale=scale, zmin=zmin, zmax=zmax, colorbar=dict(title=cbtitle), zsmooth="best"))
    
    for eq in eqs:
        fig.add_shape(type="rect", x0=eq["x"]+0.15, y0=eq["y"]+0.15, x1=eq["x"]+0.85, y1=eq["y"]+0.85, line=dict(color="black", width=1), fillcolor=eq["color"])
        fig.add_trace(go.Scatter(x=[eq["x"]+0.5], y=[eq["y"]+0.5], mode="text", text=[eq["symbol"]], textfont=dict(color="white"), showlegend=False))
    
    st_x, st_y, si_x, si_y, ly_x, ly_y = get_non_overlapping_points(room_w, room_d, n_st, n_si, n_ly, seed=99)

    if n_st > 0:
        fig.add_trace(go.Scatter(x=st_x, y=st_y, mode="markers", name=f"Standing ({n_st})", 
                                 marker=dict(size=6, color="rgba(20,20,20,0.9)", symbol="circle", line=dict(width=0.5, color="white")), hoverinfo="name"))
    if n_si > 0:
        fig.add_trace(go.Scatter(x=si_x, y=si_y, mode="markers", name=f"Sitting ({n_si})", 
                                 marker=dict(size=8, color="rgba(40,90,255,0.9)", symbol="square", line=dict(width=0.5, color="white")), hoverinfo="name"))
    if n_ly > 0:
        fig.add_trace(go.Scatter(x=ly_x, y=ly_y, mode="markers", name=f"Lying ({n_ly})", 
                                 marker=dict(size=11, color="rgba(0,150,90,0.9)", symbol="diamond-wide", line=dict(width=0.5, color="white")), hoverinfo="name"))

    fig.update_layout(width=900, height=500, title=f"{title} (T={t:.1f}h)", margin=dict(l=20, r=20, t=40, b=20))
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    return fig

# =========================================================
# Sidebar & Main UI
# =========================================================
st.sidebar.header("🏢 Room Physics")
room_w = st.sidebar.slider("Width (m)", 5.0, 500.0, 100.0)
room_d = st.sidebar.slider("Depth (m)", 5.0, 500.0, 50.0)
room_h = st.sidebar.slider("Height (m)", 2.0, 6.0, 3.0)
ach = st.sidebar.slider("Ventilation (ACH)", 0.0, 15.0, 2.0)

st.sidebar.header("👥 Population (Max 5,000 per type)")
n_st = st.sidebar.number_input("Standing", 0, 5000, 100)
n_si = st.sidebar.number_input("Sitting", 0, 5000, 50)
n_ly = st.sidebar.number_input("Lying", 0, 5000, 10)

st.sidebar.header("🚨 Emergency Scenario")
panic_mode = st.sidebar.checkbox("Panic Mode (CO2/Humidity x3)")

st.sidebar.header("🛠️ Tools")
tool = st.sidebar.radio("Equipment", ["Supply", "Exhaust", "Purifier", "Eraser"])
if st.sidebar.button("Clear All"): st.session_state.equipment_map = {}

# --- Main Page ---
st.title("CO2 & Humidity Time-Traveler")

if panic_mode:
    st.error("⚠️ **비상 사태 모드 가동 중:** 극도의 긴장과 신체 활동으로 인해 사람들의 호흡량과 발한량이 평소의 3배로 폭증합니다.")

st.subheader("🕒 Drag to see the future")
elapsed_t = st.select_slider("Select Time (Hours Elapsed)", options=np.round(np.linspace(0, 24, 241), 1), value=2.0)

trend_fig = make_trend_chart(room_w, room_d, room_h, ach, n_st, n_si, n_ly, elapsed_t, panic_mode)
st.plotly_chart(trend_fig, use_container_width=True)

vol, q_vent, base_c, base_a = compute_transient_baseline(room_w, room_d, room_h, ach, n_st, n_si, n_ly, elapsed_t, panic_mode)
eqs = equipment_list_from_map(st.session_state.equipment_map, room_w, room_d)
(X, Y), g_step = get_grid(room_w, room_d)

Z_c, Z_a = np.ones_like(X)*base_c, np.ones_like(X)*base_a
t_factor = (1.0 - math.exp(-ach*elapsed_t)) if (elapsed_t>0 and ach>0) else (1.0 if elapsed_t>0 else 0)
for eq in eqs:
    dist2 = (X - (eq['x']+0.5))**2 + (Y - (eq['y']+0.5))**2
    gauss = np.exp(-dist2 / (2.0 * max(eq['radius_m'], g_step)**2))
    Z_c -= (eq['ppm_reduction']/max(1+0.1*ach,1)) * t_factor * gauss
    Z_a -= (eq['ah_reduction']/max(1+0.1*ach,1)) * t_factor * gauss

t1, t2 = st.tabs(["🔥 CO2 Distribution", "💧 Humidity Distribution"])
with t1:
    st.plotly_chart(make_heatmap_fig(room_w, room_d, X, Y, Z_c, eqs, elapsed_t, n_st, n_si, n_ly, "CO2 (ppm)", "Turbo", 400, 2500, "ppm"), use_container_width=False)
with t2:
    st.plotly_chart(make_heatmap_fig(room_w, room_d, X, Y, Z_a, eqs, elapsed_t, n_st, n_si, n_ly, "Abs. Humidity (g/m³)", "Tealrose", 5, 30, "g/m³"), use_container_width=False)

st.divider()
nx, ny = 11, 11
dx, dy = room_w/10, room_d/10
st.markdown(f"### 📍 Equipment Placement Grid (Node Spacing: {dx:.1f}m x {dy:.1f}m)")
legend_cols = st.columns(5)
for i, (k, v) in enumerate(EQUIPMENT_TYPES.items()):
    legend_cols[i].write(f"{v['label']} {k}")

for iy in range(ny):
    cols = st.columns([0.8] + [1]*nx)
    cols[0].write(f"Y{iy}\n({iy*dy:.0f}m)")
    for ix in range(nx):
        et = st.session_state.equipment_map.get((ix, iy))
        lbl = EQUIPMENT_TYPES[et]["label"] if et in EQUIPMENT_TYPES else "·"
        if cols[ix+1].button(lbl, key=f"c_{ix}_{iy}", use_container_width=True):
            if tool == "Eraser": st.session_state.equipment_map.pop((ix, iy), None)
            else: st.session_state.equipment_map[(ix, iy)] = tool
            st.rerun()
