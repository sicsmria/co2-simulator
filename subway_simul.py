import math
from typing import Dict, List
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="CO2 Room Simulator", layout="wide")

# =========================================================
# Constants
# =========================================================
OUTDOOR_CO2_PPM = 420.0
MAX_HEATMAP_CELLS = 15000

CO2_GEN_M3_PER_H = {
    "standing": 0.021,
    "sitting": 0.018,
    "lying": 0.015,
}

EQUIPMENT_TYPES = {
    "Supply": {
        "color": "rgba(0,140,255,0.95)",
        "symbol": "S",
        "radius_m": 8.0,
        "ppm_reduction": 120.0,
        "label": "🟦",
    },
    "Exhaust": {
        "color": "rgba(255,90,90,0.95)",
        "symbol": "E",
        "radius_m": 10.0,
        "ppm_reduction": 150.0,
        "label": "🟥",
    },
    "Purifier": {
        "color": "rgba(0,180,120,0.95)",
        "symbol": "P",
        "radius_m": 6.0,
        "ppm_reduction": 90.0,
        "label": "🟩",
    },
}

EMPTY_CELL_LABEL = "·"

if "equipment_map" not in st.session_state:
    st.session_state.equipment_map = {}

# =========================================================
# Helpers
# =========================================================
def get_adaptive_grid_step(room_w: float, room_d: float, target_max_cells: int = MAX_HEATMAP_CELLS) -> float:
    room_area = max(room_w * room_d, 1e-9)
    raw_step = math.sqrt(room_area / target_max_cells)
    
    # 너무 작은 값 방지 (최소 단위를 1.0m로 축소)
    step = max(1.0, raw_step)
    
    # 보기 좋은 구간으로 스냅 (1.0 옵션 추가)
    candidate_steps = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    for s in candidate_steps:
        if s >= step:
            return s
    return candidate_steps[-1]

@st.cache_data(show_spinner=False)
def make_grid(room_w: float, room_d: float, step_m: float):
    nx = max(2, int(math.ceil(room_w / step_m)) + 1)
    ny = max(2, int(math.ceil(room_d / step_m)) + 1)
    xs = np.linspace(0.0, room_w, nx)
    ys = np.linspace(0.0, room_d, ny)
    return np.meshgrid(xs, ys)

def equipment_list_from_map(equipment_map: Dict, room_w: float, room_d: float) -> List[Dict]:
    equipments = []
    # 항상 11개의 점(10칸)으로 공간을 나눔
    nx, ny = 11, 11
    dx = room_w / (nx - 1)
    dy = room_d / (ny - 1)

    for (ix, iy), eq_type in equipment_map.items():
        if 0 <= ix < nx and 0 <= iy < ny and eq_type in EQUIPMENT_TYPES:
            eq = EQUIPMENT_TYPES[eq_type]
            equipments.append({
                "ix": ix, "iy": iy,
                "x": (ix * dx) - 0.5, # 1m 크기 설비의 중심을 정확한 좌표에 맞춤
                "y": (iy * dy) - 0.5,
                "w": 1.0,             # 설비 자체의 크기는 1m * 1m로 고정
                "d": 1.0,
                "type": eq_type,
                **eq
            })
    return equipments
    
def compute_transient_baseline_ppm(
    room_w: float, room_d: float, room_h: float, ach: float,
    n_standing: int, n_sitting: int, n_lying: int, elapsed_time_h: float
):
    volume_m3 = room_w * room_d * room_h
    q_vent_m3ph = max(ach * volume_m3, 0.001)

    g_total = (
        n_standing * CO2_GEN_M3_PER_H["standing"] +
        n_sitting * CO2_GEN_M3_PER_H["sitting"] +
        n_lying * CO2_GEN_M3_PER_H["lying"]
    )

    steady_state_delta = 1_000_000.0 * g_total / q_vent_m3ph

    if ach > 0:
        current_delta = steady_state_delta * (1.0 - math.exp(-ach * elapsed_time_h))
    else:
        current_delta = 1_000_000.0 * g_total * elapsed_time_h / max(volume_m3, 1e-9)

    baseline_ppm = OUTDOOR_CO2_PPM + current_delta
    return volume_m3, q_vent_m3ph, baseline_ppm

@st.cache_data(show_spinner=False)
def compute_equipment_field(
    room_w: float, room_d: float, room_h: float, ach: float,
    n_standing: int, n_sitting: int, n_lying: int,
    equipments_key: tuple, grid_step_m: float, elapsed_time_h: float
):
    X, Y = make_grid(room_w, room_d, grid_step_m)
    volume_m3, q_vent_m3ph, baseline_ppm = compute_transient_baseline_ppm(
        room_w, room_d, room_h, ach,
        n_standing, n_sitting, n_lying, elapsed_time_h
    )

    Z = np.ones_like(X, dtype=float) * baseline_ppm

    for eq_type, x, y, w, d in equipments_key:
        eq = EQUIPMENT_TYPES[eq_type]
        cx, cy = x + w / 2.0, y + d / 2.0
        sigma = max(eq["radius_m"], grid_step_m * 1.2)
        dist2 = (X - cx) ** 2 + (Y - cy) ** 2

        if elapsed_time_h > 0 and ach > 0:
            time_factor = (1.0 - math.exp(-ach * elapsed_time_h))
        elif elapsed_time_h > 0:
            time_factor = 1.0
        else:
            time_factor = 0.0

        reduction = (eq["ppm_reduction"] / max(1.0 + 0.1 * ach, 1.0)) * time_factor
        Z -= reduction * np.exp(-dist2 / (2.0 * sigma ** 2))

    Z = np.clip(Z, OUTDOOR_CO2_PPM, 5000.0)
    return X, Y, Z, volume_m3, q_vent_m3ph, baseline_ppm

def add_equipment_shapes(fig: go.Figure, equipments: List[Dict]) -> None:
    for eq in equipments:
        fig.add_shape(
            type="rect",
            x0=eq["x"] + eq["w"] * 0.15,
            y0=eq["y"] + eq["d"] * 0.15,
            x1=eq["x"] + eq["w"] * 0.85,
            y1=eq["y"] + eq["d"] * 0.85,
            line=dict(color="black", width=1),
            fillcolor=eq["color"],
        )
        fig.add_trace(go.Scatter(
            x=[eq["x"] + eq["w"] / 2.0],
            y=[eq["y"] + eq["d"] / 2.0],
            mode="text",
            text=[eq["symbol"]],
            textfont=dict(color="white", size=12),
            showlegend=False,
            hoverinfo="skip"
        ))

def generate_people_points(room_w: float, room_d: float, n: int, seed: int):
    if n <= 0:
        return np.array([]), np.array([])
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.5, max(0.5, room_w - 0.5), n)
    y = rng.uniform(0.5, max(0.5, room_d - 0.5), n)
    return x, y

def add_people_markers(fig: go.Figure, room_w: float, room_d: float, n_standing: int, n_sitting: int, n_lying: int) -> None:
    if n_standing > 0:
        x, y = generate_people_points(room_w, room_d, n_standing, seed=11)
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name=f"Standing ({n_standing})",
            marker=dict(
                size=7,
                color="rgba(20,20,20,0.95)",
                symbol="circle",
                line=dict(width=0.5, color="white")
            ),
            hovertemplate="Standing person<br>x=%{x:.1f} m<br>y=%{y:.1f} m<extra></extra>"
        ))

    if n_sitting > 0:
        x, y = generate_people_points(room_w, room_d, n_sitting, seed=22)
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name=f"Sitting ({n_sitting})",
            marker=dict(
                size=8,
                color="rgba(40,90,255,0.95)",
                symbol="square",
                line=dict(width=0.5, color="white")
            ),
            hovertemplate="Sitting person<br>x=%{x:.1f} m<br>y=%{y:.1f} m<extra></extra>"
        ))

    if n_lying > 0:
        x, y = generate_people_points(room_w, room_d, n_lying, seed=33)
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name=f"Lying ({n_lying})",
            marker=dict(
                size=10,
                color="rgba(0,150,90,0.95)",
                symbol="diamond-wide",
                line=dict(width=0.5, color="white")
            ),
            hovertemplate="Lying person<br>x=%{x:.1f} m<br>y=%{y:.1f} m<extra></extra>"
        ))

def make_heatmap_figure(
    room_w, room_d, X, Y, Z, equipments, elapsed_time_h,
    n_standing, n_sitting, n_lying
):
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=X[0],
        y=Y[:, 0],
        z=Z,
        colorscale="Turbo",
        zmin=400,
        zmax=2500,
        colorbar=dict(title="CO2 (ppm)"),
        hovertemplate="x=%{x:.1f} m<br>y=%{y:.1f} m<br>CO2=%{z:.0f} ppm<extra></extra>",
        zsmooth="best"
    ))

    add_equipment_shapes(fig, equipments)
    add_people_markers(fig, room_w, room_d, n_standing, n_sitting, n_lying)

    fig.update_layout(
        width=950,
        height=540,
        autosize=False,
        margin=dict(l=20, r=20, t=45, b=20),
        title=f"CO2 Heatmap (T = {elapsed_time_h:.1f} Hours)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0
        )
    )
    fig.update_xaxes(
        title="Width (m)",
        range=[0.0, room_w],
        constrain="domain",
        scaleanchor="y",
        scaleratio=1
    )
    fig.update_yaxes(
        title="Depth (m)",
        range=[0.0, room_d],
        autorange="reversed"
    )
    return fig

def toggle_equipment(ix: int, iy: int, selected_tool: str) -> None:
    key = (ix, iy)
    if selected_tool == "Eraser":
        st.session_state.equipment_map.pop(key, None)
    else:
        st.session_state.equipment_map[key] = selected_tool

def trim_equipment_map(room_w: float, room_d: float) -> None:
    # 방 크기에 상관없이 항상 10등분(11개 교차점) 기준 내의 데이터만 유지
    st.session_state.equipment_map = {
        k: v for k, v in st.session_state.equipment_map.items()
        if k[0] < 11 and k[1] < 11
    }

def recommend_cell_size(room_w: float, room_d: float, current: float) -> float:
    area = room_w * room_d
    if area > 2_000_000:
        return max(current, 100.0)
    if area > 500_000:
        return max(current, 50.0)
    if area > 100_000:
        return max(current, 20.0)
    if area > 20_000:
        return max(current, 10.0)
    if area > 5_000:
        return max(current, 5.0) # 추가된 기준
    return current

def cell_label(ix: int, iy: int) -> str:
    eq_type = st.session_state.equipment_map.get((ix, iy))
    if eq_type in EQUIPMENT_TYPES:
        return EQUIPMENT_TYPES[eq_type]["label"]
    return EMPTY_CELL_LABEL

def render_equipment_editor(room_w: float, room_d: float, selected_tool: str):
    st.markdown("### Equipment Placement Grid")
    
    # 그리드 화면 무조건 10등분 고정
    DIVISIONS = 10
    nx = DIVISIONS + 1
    ny = DIVISIONS + 1
    
    dx = room_w / DIVISIONS
    dy = room_d / DIVISIONS

    st.caption(f"💡 모서리 기준 자동 분할: 공간 전체가 {DIVISIONS}등분 되었습니다. (간격: 가로 {dx:.1f}m, 세로 {dy:.1f}m)")

    legend_cols = st.columns(4)
    legend_cols[0].markdown(f"**{EQUIPMENT_TYPES['Supply']['label']} Supply**")
    legend_cols[1].markdown(f"**{EQUIPMENT_TYPES['Exhaust']['label']} Exhaust**")
    legend_cols[2].markdown(f"**{EQUIPMENT_TYPES['Purifier']['label']} Purifier**")
    legend_cols[3].markdown(f"**{EMPTY_CELL_LABEL} Empty**")

    col_ratios = [0.8] + [1.0] * nx

    # X축 (상단 라벨)
    header_cols = st.columns(col_ratios)
    header_cols[0].write("") 
    for ix in range(nx):
        actual_x = ix * dx
        header_cols[ix + 1].markdown(f"<div style='text-align: center; color: gray; font-size: 0.8em; padding-bottom: 5px;'>X{ix}<br>({actual_x:.1f}m)</div>", unsafe_allow_html=True)

    # Y축 및 버튼 배치
    for iy in range(ny):
        cols = st.columns(col_ratios)
        actual_y = iy * dy
        cols[0].markdown(f"<div style='padding-top: 8px; font-weight: bold; color: gray; font-size: 0.85em;'>Y{iy}<br>({actual_y:.1f}m)</div>", unsafe_allow_html=True)
        
        for ix in range(nx):
            label = cell_label(ix, iy)
            if cols[ix + 1].button(label, key=f"cell_{ix}_{iy}", use_container_width=True):
                toggle_equipment(ix, iy, selected_tool)
                st.rerun()
# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("Room & Time Settings")

elapsed_time_h = st.sidebar.slider(
    "Elapsed Time (Hours)", min_value=0.0, max_value=48.0, value=2.0, step=0.5
)
room_w = st.sidebar.slider(
    "Room Width (m)", min_value=2.0, max_value=5000.0, value=100.0, step=1.0
)
room_d = st.sidebar.slider(
    "Room Depth (m)", min_value=2.0, max_value=1000.0, value=50.0, step=1.0
)
room_h = st.sidebar.slider(
    "Room Height (m)", min_value=2.0, max_value=6.0, value=3.0, step=0.1
)
ach = st.sidebar.slider(
    "Ventilation ACH (1/h)", min_value=0.0, max_value=20.0, value=3.0, step=0.1
)

st.sidebar.header("Population")
n_standing = st.sidebar.number_input(
    "Standing people", min_value=0, max_value=3000, value=100, step=1
)
n_sitting = st.sidebar.number_input(
    "Sitting people", min_value=0, max_value=3000, value=50, step=1
)
n_lying = st.sidebar.number_input(
    "Lying people", min_value=0, max_value=3000, value=10, step=1
)

st.sidebar.header("Equipment Editor")
selected_tool = st.sidebar.radio(
    "Equipment Tool", ["Supply", "Exhaust", "Purifier", "Eraser"], index=0
)

# 옵션에 1.0, 2.0 추가 및 기본값을 1.0으로 변경
editor_step = st.sidebar.select_slider(
    "Cell Size (m)", 
    options=[1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0], 
    value=1.0
)

if st.sidebar.button("Clear Equipments"):
    st.session_state.equipment_map = {}

trim_equipment_map(room_w, room_d, editor_step)

# =========================================================
# Main compute
# =========================================================
equipments = equipment_list_from_map(
    st.session_state.equipment_map, room_w, room_d, editor_step
)
equipments_key = tuple(
    (e["type"], e["x"], e["y"], e["w"], e["d"]) for e in equipments
)

grid_step_m = get_adaptive_grid_step(room_w, room_d)

X, Y, Z, volume_m3, q_vent_m3ph, baseline_ppm = compute_equipment_field(
    room_w, room_d, room_h, ach,
    int(n_standing), int(n_sitting), int(n_lying),
    equipments_key, grid_step_m, elapsed_time_h
)

fig = make_heatmap_figure(
    room_w, room_d, X, Y, Z, equipments, elapsed_time_h,
    int(n_standing), int(n_sitting), int(n_lying)
)

recommended_cell = recommend_cell_size(room_w, room_d, editor_step)

# =========================================================
# Main UI
# =========================================================
st.title("CO2 Transient Simulator")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Width", f"{room_w:.1f} m")
m2.metric("Depth", f"{room_d:.1f} m")
m3.metric("Height", f"{room_h:.1f} m")
m4.metric("Volume", f"{volume_m3:.1f} m³")
m5.metric("Population", f"{int(n_standing + n_sitting + n_lying)}")

s1, s2, s3 = st.columns(3)
s1.metric("Ventilation Flow", f"{q_vent_m3ph:.1f} m³/h")
s2.metric(f"CO2 at {elapsed_time_h}h", f"{baseline_ppm:.0f} ppm")
s3.metric("Grid Step", f"{grid_step_m:.1f} m")

if recommended_cell > editor_step:
    st.warning(f"This room is large. Recommended Cell Size: {recommended_cell:.1f} m or larger.")

st.plotly_chart(fig, use_container_width=False)

render_equipment_editor(room_w, room_d, editor_step, selected_tool)

with st.expander("Current equipment map"):
    st.write(st.session_state.equipment_map)
