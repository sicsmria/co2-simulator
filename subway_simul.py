import math
from typing import Dict, Tuple, List

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="CO2 Room Simulator Lite", layout="wide")

# =========================================================
# Constants
# =========================================================
OUTDOOR_CO2_PPM = 420.0
MAX_HEATMAP_CELLS = 20000

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
    },
    "Exhaust": {
        "color": "rgba(255,90,90,0.95)",
        "symbol": "E",
        "radius_m": 10.0,
        "ppm_reduction": 150.0,
    },
    "Purifier": {
        "color": "rgba(0,180,120,0.95)",
        "symbol": "P",
        "radius_m": 6.0,
        "ppm_reduction": 90.0,
    },
}

if "equipment_map" not in st.session_state:
    st.session_state.equipment_map = {}


# =========================================================
# Helpers
# =========================================================
def get_adaptive_grid_step(room_w: float, room_d: float, target_max_cells: int = MAX_HEATMAP_CELLS) -> float:
    room_area = max(room_w * room_d, 1e-9)
    raw_step = math.sqrt(room_area / target_max_cells)
    step = max(2.0, raw_step)

    candidate_steps = [2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
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
    X, Y = np.meshgrid(xs, ys)
    return X, Y


def equipment_list_from_map(
    equipment_map: Dict[Tuple[int, int], str],
    room_w: float,
    room_d: float,
    cell_size_m: float,
) -> List[Dict]:
    equipments = []
    nx = int(room_w // cell_size_m)
    ny = int(room_d // cell_size_m)

    for (ix, iy), eq_type in equipment_map.items():
        if 0 <= ix < nx and 0 <= iy < ny and eq_type in EQUIPMENT_TYPES:
            x0 = ix * cell_size_m
            y0 = iy * cell_size_m
            eq = EQUIPMENT_TYPES[eq_type]
            equipments.append(
                {
                    "ix": ix,
                    "iy": iy,
                    "x": x0,
                    "y": y0,
                    "w": cell_size_m,
                    "d": cell_size_m,
                    "type": eq_type,
                    "radius_m": eq["radius_m"],
                    "ppm_reduction": eq["ppm_reduction"],
                    "symbol": eq["symbol"],
                    "color": eq["color"],
                }
            )
    return equipments


def compute_baseline_ppm(
    room_w: float,
    room_d: float,
    room_h: float,
    ach: float,
    n_standing: int,
    n_sitting: int,
    n_lying: int,
):
    volume_m3 = room_w * room_d * room_h
    q_vent_m3ph = max(ach * volume_m3, 0.001)

    g_total = (
        n_standing * CO2_GEN_M3_PER_H["standing"]
        + n_sitting * CO2_GEN_M3_PER_H["sitting"]
        + n_lying * CO2_GEN_M3_PER_H["lying"]
    )

    delta_ppm = 1_000_000.0 * g_total / q_vent_m3ph
    baseline_ppm = OUTDOOR_CO2_PPM + delta_ppm
    return volume_m3, q_vent_m3ph, baseline_ppm


@st.cache_data(show_spinner=False)
def compute_equipment_field(
    room_w: float,
    room_d: float,
    room_h: float,
    ach: float,
    n_standing: int,
    n_sitting: int,
    n_lying: int,
    equipments_key: tuple,
    grid_step_m: float,
):
    X, Y = make_grid(room_w, room_d, grid_step_m)
    volume_m3, q_vent_m3ph, baseline_ppm = compute_baseline_ppm(
        room_w, room_d, room_h, ach, n_standing, n_sitting, n_lying
    )

    Z = np.ones_like(X, dtype=float) * baseline_ppm

    for eq_type, x, y, w, d in equipments_key:
        eq = EQUIPMENT_TYPES[eq_type]
        cx = x + w / 2.0
        cy = y + d / 2.0
        sigma = max(eq["radius_m"], grid_step_m * 1.2)
        dist2 = (X - cx) ** 2 + (Y - cy) ** 2

        reduction = eq["ppm_reduction"] / max(1.0 + 0.1 * ach, 1.0)
        Z -= reduction * np.exp(-dist2 / (2.0 * sigma**2))

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
        fig.add_trace(
            go.Scatter(
                x=[eq["x"] + eq["w"] / 2.0],
                y=[eq["y"] + eq["d"] / 2.0],
                mode="text",
                text=[eq["symbol"]],
                textfont=dict(color="white", size=12),
                showlegend=False,
                hoverinfo="skip",
            )
        )


def make_heatmap_figure(room_w, room_d, X, Y, Z, equipments):
    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=X[0],
            y=Y[:, 0],
            z=Z,
            colorscale="Turbo",
            zmin=400,
            zmax=2500,
            colorbar=dict(title="CO2 (ppm)"),
            hovertemplate="x=%{x:.1f} m<br>y=%{y:.1f} m<br>CO2=%{z:.0f} ppm<extra></extra>",
        )
    )

    add_equipment_shapes(fig, equipments)

    fig.update_layout(
        width=950,
        height=540,
        autosize=False,
        margin=dict(l=20, r=20, t=45, b=20),
        title="CO2 Heatmap (Lite Mode)",
    )
    fig.update_xaxes(
        title="Width (m)",
        range=[0.0, room_w],
        constrain="domain",
        scaleanchor="y",
        scaleratio=1,
    )
    fig.update_yaxes(
        title="Depth (m)",
        range=[0.0, room_d],
        autorange="reversed",
    )
    return fig


def toggle_equipment(ix: int, iy: int, selected_tool: str) -> None:
    key = (ix, iy)
    if selected_tool == "Eraser":
        st.session_state.equipment_map.pop(key, None)
    else:
        st.session_state.equipment_map[key] = selected_tool


def trim_equipment_map(room_w: float, room_d: float, cell_size_m: float) -> None:
    max_ix = int(room_w // cell_size_m)
    max_iy = int(room_d // cell_size_m)
    st.session_state.equipment_map = {
        (ix, iy): eq
        for (ix, iy), eq in st.session_state.equipment_map.items()
        if ix < max_ix and iy < max_iy
    }


# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("Room Settings")
room_w = st.sidebar.slider("Room Width (m)", min_value=2.0, max_value=5000.0, value=100.0, step=1.0)
room_d = st.sidebar.slider("Room Depth (m)", min_value=2.0, max_value=1000.0, value=50.0, step=1.0)
room_h = st.sidebar.slider("Room Height (m)", min_value=2.0, max_value=6.0, value=3.0, step=0.1)
ach = st.sidebar.slider("Ventilation ACH (1/h)", min_value=0.1, max_value=20.0, value=3.0, step=0.1)

st.sidebar.header("Population")
n_standing = st.sidebar.number_input("Standing people", min_value=0, max_value=3000, value=100, step=1)
n_sitting = st.sidebar.number_input("Sitting people", min_value=0, max_value=3000, value=50, step=1)
n_lying = st.sidebar.number_input("Lying people", min_value=0, max_value=3000, value=10, step=1)

st.sidebar.header("Equipment Editor")
selected_tool = st.sidebar.radio("Equipment Tool", ["Supply", "Exhaust", "Purifier", "Eraser"], index=0)
editor_step = st.sidebar.select_slider("Cell Size (m)", options=[5.0, 10.0, 20.0, 50.0, 100.0], value=20.0)

if st.sidebar.button("Clear Equipments"):
    st.session_state.equipment_map = {}

trim_equipment_map(room_w, room_d, editor_step)

# =========================================================
# Main compute
# =========================================================
equipments = equipment_list_from_map(st.session_state.equipment_map, room_w, room_d, editor_step)
equipments_key = tuple((e["type"], e["x"], e["y"], e["w"], e["d"]) for e in equipments)

grid_step_m = get_adaptive_grid_step(room_w, room_d)

X, Y, Z, volume_m3, q_vent_m3ph, baseline_ppm = compute_equipment_field(
    room_w, room_d, room_h, ach,
    int(n_standing), int(n_sitting), int(n_lying),
    equipments_key,
    grid_step_m
)

fig = make_heatmap_figure(room_w, room_d, X, Y, Z, equipments)

# =========================================================
# Main UI
# =========================================================
st.title("CO2 Room Simulator Lite")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Width", f"{room_w:.1f} m")
m2.metric("Depth", f"{room_d:.1f} m")
m3.metric("Height", f"{room_h:.1f} m")
m4.metric("Volume", f"{volume_m3:.1f} m³")
m5.metric("Population", f"{int(n_standing+n_sitting+n_lying)}")

s1, s2, s3 = st.columns(3)
s1.metric("Ventilation Flow", f"{q_vent_m3ph:.1f} m³/h")
s2.metric("Baseline CO2", f"{baseline_ppm:.0f} ppm")
s3.metric("Grid Step", f"{grid_step_m:.1f} m")

st.info("Lite mode: occupants are reflected in total CO2 generation only. Individual human placement is disabled for speed.")

st.plotly_chart(fig, use_container_width=False)

# =========================================================
# Coordinate-based equipment editor
# =========================================================
st.subheader("Equipment Placement")
st.caption("Use coordinates instead of button grids for large spaces.")

nx = max(1, int(room_w // editor_step))
ny = max(1, int(room_d // editor_step))

c1, c2, c3 = st.columns(3)
with c1:
    place_x = st.number_input("Cell X", min_value=0, max_value=max(nx - 1, 0), value=0, step=1)
with c2:
    place_y = st.number_input("Cell Y", min_value=0, max_value=max(ny - 1, 0), value=0, step=1)
with c3:
    if st.button("Apply Equipment"):
        toggle_equipment(int(place_x), int(place_y), selected_tool)
        st.rerun()

with st.expander("Placed Equipments"):
    if equipments:
        for i, eq in enumerate(equipments, 1):
            st.write(
                f"{i}. {eq['type']} | cell=({eq['ix']}, {eq['iy']}) | "
                f"x={eq['x']:.1f} m, y={eq['y']:.1f} m"
            )
    else:
        st.write("No equipments placed.")
