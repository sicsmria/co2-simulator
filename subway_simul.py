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
def compute_transient_baseline(room_w, room_d, room_h, ach, n_st, n_si, n_ly, t):
    vol = room_w * room_d * room_h
    q_vent = max(ach * vol, 0.001)
    g_co2 = n_st*0.021 + n_si*0.018 + n_ly*0.015
    g_ah = n_st*75.0 + n_si*50.0 + n_ly*40.0

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

def make_trend_chart(room_w, room_d, room_h, ach, n_st, n_si, n_ly, current_t):
    times = np.linspace(0, 24, 49) # 0.5시간 단위
    co2_vals, ah_vals = [], []
    for t in times:
        _, _, c, a = compute_transient_baseline(room_w, room_d, room_h, ach, n_st, n_si, n_ly, t)
        co2_vals.append(c)
        ah_vals.append(a)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=co2_vals, name="CO2 (ppm)", yaxis="y1", line=dict(color="orangered", width=3)))
    fig.add_trace(go.Scatter(x=times, y=ah_vals, name="AH (g/m³)", yaxis="y2", line=dict(color="royalblue", width=3, dash='dot')))
    
    # 현재 선택 시간 수직선 표시
    fig.add_vline(x=current_t, line_width=2, line_dash="dash", line_color="black")
    
    # 최신 Plotly 문법에 맞게 yaxis, yaxis2의 title 및 font 구조 수정
    fig.update_layout(
        title="24-Hour Concentration Trend",
        xaxis=dict(title="Hours Elapsed", range=[0, 24]),
        yaxis=dict(
            title=dict(text="CO2 Concentration (ppm)", font=dict(color="orangered")), 
            tickfont=dict(color="orangered")
        ),
        yaxis2=dict(
            title=dict(text="Absolute Humidity (g/m³)", font=dict(color="royalblue")), 
            tickfont=dict(color="royalblue"), 
            overlaying="y", 
            side="right"
        ),
        height=300, 
        margin=dict(l=20, r=20, t=50, b=20), 
        hovermode="x unified"
    )
    return fig

# (기타 Heatmap 및 Editor 함수는 이전과 동일한 로직을 유지하되 가독성을 위해 최적화)
# ... [중략: equipment_list_from_map, add_equipment_shapes, add_people_markers 등] ...

def equipment_list_from_map(equipment_map, room_w, room_d):
    eqs = []
    nx, ny = 11, 11
    dx, dy = room_w / 10, room_d / 10
    for (ix, iy), eq_type in equipment_map.items():
        if eq_type in EQUIPMENT_TYPES:
            eq = EQUIPMENT_TYPES[eq_type]
            eqs.append({"ix": ix, "iy": iy, "x": (ix*dx)-0.5, "y": (iy*dy)-0.5, "w": 1.0, "d": 1.0, "type": eq_type, **eq})
    return eqs

def make_heatmap_fig(room_w, room_d, X, Y, Z, eqs, t, n_st, n_si, n_ly, title, scale, zmin, zmax, cbtitle):
    fig = go.Figure(go.Heatmap(x=X[0], y=Y[:,0], z=Z, colorscale=scale, zmin=zmin, zmax=zmax, colorbar=dict(title=cbtitle), zsmooth="best"))
    for eq in eqs:
        fig.add_shape(type="rect", x0=eq["x"]+0.15, y0=eq["y"]+0.15, x1=eq["x"]+0.85, y1=eq["y"]+0.85, line=dict(color="black", width=1), fillcolor=eq["color"])
        fig.add_trace(go.Scatter(x=[eq["x"]+0.5], y=[eq["y"]+0.5], mode="text", text=[eq["symbol"]], textfont=dict(color="white"), showlegend=False))
    
    # 사람 마커 추가 (이전 함수 통합 호출)
    if n_st > 0: 
        rx, ry = np.random.default_rng(11).uniform(0.5, room_w-0.5, n_st), np.random.default_rng(11).uniform(0.5, room_d-0.5, n_st)
        fig.add_trace(go.Scatter(x=rx, y=ry, mode="markers", marker=dict(size=6, color="black"), name="Standing"))
    
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

st.sidebar.header("👥 Population")
n_st = st.sidebar.number_input("Standing", 0, 2000, 50)
n_si = st.sidebar.number_input("Sitting", 0, 2000, 30)
n_ly = st.sidebar.number_input("Lying", 0, 2000, 10)

st.sidebar.header("🛠️ Tools")
tool = st.sidebar.radio("Equipment", ["Supply", "Exhaust", "Purifier", "Eraser"])
if st.sidebar.button("Clear All"): st.session_state.equipment_map = {}

# --- Main Page ---
st.title("CO2 & Humidity Time-Traveler")

# 1. 시간 슬라이더 (메인 상단 배치)
st.subheader("🕒 Drag to see the future")
elapsed_t = st.select_slider("Select Time (Hours Elapsed)", options=np.round(np.linspace(0, 24, 241), 1), value=2.0)

# 2. 실시간 트렌드 그래프
trend_fig = make_trend_chart(room_w, room_d, room_h, ach, n_st, n_si, n_ly, elapsed_t)
st.plotly_chart(trend_fig, use_container_width=True)

# 3. 데이터 계산
vol, q_vent, base_c, base_a = compute_transient_baseline(room_w, room_d, room_h, ach, n_st, n_si, n_ly, elapsed_t)
eqs = equipment_list_from_map(st.session_state.equipment_map, room_w, room_d)
(X, Y), g_step = get_grid(room_w, room_d)

# 설비 필드 연산 (간소화 버전)
Z_c, Z_a = np.ones_like(X)*base_c, np.ones_like(X)*base_a
t_factor = (1.0 - math.exp(-ach*elapsed_t)) if (elapsed_t>0 and ach>0) else (1.0 if elapsed_t>0 else 0)
for eq in eqs:
    dist2 = (X - (eq['x']+0.5))**2 + (Y - (eq['y']+0.5))**2
    gauss = np.exp(-dist2 / (2.0 * max(eq['radius_m'], g_step)**2))
    Z_c -= (eq['ppm_reduction']/max(1+0.1*ach,1)) * t_factor * gauss
    Z_a -= (eq['ah_reduction']/max(1+0.1*ach,1)) * t_factor * gauss

# 4. 히트맵 탭 배치
t1, t2 = st.tabs(["🔥 CO2 Distribution", "💧 Humidity Distribution"])
with t1:
    st.plotly_chart(make_heatmap_fig(room_w, room_d, X, Y, Z_c, eqs, elapsed_t, n_st, n_si, n_ly, "CO2 (ppm)", "Turbo", 400, 2500, "ppm"), use_container_width=False)
with t2:
    st.plotly_chart(make_heatmap_fig(room_w, room_d, X, Y, Z_a, eqs, elapsed_t, n_st, n_si, n_ly, "Abs. Humidity (g/m³)", "Tealrose", 5, 30, "g/m³"), use_container_width=False)

# 5. 그리드 에디터
# (이전의 render_equipment_editor 로직 동일 사용)
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
