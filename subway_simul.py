import math
import random
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="CO2 Room Simulator", layout="wide")

# =========================================================
# Constants (meter-based)
# =========================================================
OUTDOOR_CO2_PPM = 420.0
DEFAULT_EDITOR_CELL_M = 1.0

CO2_GEN_M3_PER_H = {
    "standing": 0.021,
    "sitting": 0.018,
    "lying": 0.015,
}

HUMAN_TYPES = {
    "standing": {
        "w": 0.40,
        "d": 0.40,
        "label": "Standing",
        "co2_gen": CO2_GEN_M3_PER_H["standing"],
    },
    "sitting": {
        "w": 0.50,
        "d": 0.50,
        "label": "Sitting",
        "co2_gen": CO2_GEN_M3_PER_H["sitting"],
    },
    "lying": {
        "w": 0.60,
        "d": 1.70,
        "label": "Lying",
        "co2_gen": CO2_GEN_M3_PER_H["lying"],
    },
}

EQUIPMENT_TYPES = {
    "Supply": {
        "color": "rgba(0,140,255,0.95)",
        "symbol": "S",
        "radius_m": 2.2,
        "ppm_reduction": 180.0,
    },
    "Exhaust": {
        "color": "rgba(255,90,90,0.95)",
        "symbol": "E",
        "radius_m": 2.8,
        "ppm_reduction": 220.0,
    },
    "Purifier": {
        "color": "rgba(0,180,120,0.95)",
        "symbol": "P",
        "radius_m": 1.8,
        "ppm_reduction": 130.0,
    },
}

MAX_HEATMAP_CELLS = 180_000
MAX_DRAWN_PEOPLE = 800
MAX_EDITOR_BUTTONS = 1200


# =========================================================
# Session state
# =========================================================
if "people_layout" not in st.session_state:
    st.session_state.people_layout = []

if "layout_seed" not in st.session_state:
    st.session_state.layout_seed = 0

if "last_layout_signature" not in st.session_state:
    st.session_state.last_layout_signature = None

if "equipment_map" not in st.session_state:
    st.session_state.equipment_map = {}


# =========================================================
# Utility functions
# =========================================================
def rects_overlap(a: Dict, b: Dict) -> bool:
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["d"]

    bx1, by1 = b["x"], b["y"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["d"]

    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def can_place(candidate: Dict, placed: List[Dict]) -> bool:
    for p in placed:
        if rects_overlap(candidate, p):
            return False
    return True


def get_adaptive_grid_step(room_w: float, room_d: float, target_max_cells: int = MAX_HEATMAP_CELLS) -> float:
    """
    Adaptive grid step so huge rooms do not explode memory/time.
    """
    room_area = max(room_w * room_d, 1e-9)
    raw_step = math.sqrt(room_area / target_max_cells)

    # Lower bound to preserve detail in small rooms
    step = max(0.20, raw_step)

    # Snap to reasonable values
    candidate_steps = [0.2, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 25.0, 50.0]
    for s in candidate_steps:
        if s >= step:
            return s
    return candidate_steps[-1]


def make_grid(room_w: float, room_d: float, step_m: float):
    nx = max(2, int(math.ceil(room_w / step_m)) + 1)
    ny = max(2, int(math.ceil(room_d / step_m)) + 1)
    xs = np.linspace(0.0, room_w, nx)
    ys = np.linspace(0.0, room_d, ny)
    X, Y = np.meshgrid(xs, ys)
    return X, Y


def generate_random_people(
    room_w: float,
    room_d: float,
    n_standing: int,
    n_sitting: int,
    n_lying: int,
    seed: int = 0,
) -> List[Dict]:
    rng = random.Random(seed)
    placed = []

    request = (
        [("standing", i) for i in range(n_standing)]
        + [("sitting", i) for i in range(n_sitting)]
        + [("lying", i) for i in range(n_lying)]
    )
    rng.shuffle(request)

    for kind, _ in request:
        spec = HUMAN_TYPES[kind]
        base_w, base_d = spec["w"], spec["d"]

        orientations = [(base_w, base_d)]
        if kind == "lying":
            orientations.append((base_d, base_w))

        success = False
        for _ in range(120):
            w, d = rng.choice(orientations)

            if room_w <= w or room_d <= d:
                continue

            x = rng.uniform(0.0, room_w - w)
            y = rng.uniform(0.0, room_d - d)

            candidate = {
                "type": kind,
                "x": x,
                "y": y,
                "w": w,
                "d": d,
                "co2_gen": spec["co2_gen"],
                "label": spec["label"],
            }

            if can_place(candidate, placed):
                placed.append(candidate)
                success = True
                break

        if not success:
            continue

    return placed


def ensure_people_layout(
    room_w: float,
    room_d: float,
    n_standing: int,
    n_sitting: int,
    n_lying: int,
) -> None:
    signature = (
        round(room_w, 3),
        round(room_d, 3),
        n_standing,
        n_sitting,
        n_lying,
        st.session_state.layout_seed,
    )
    if st.session_state.last_layout_signature != signature:
        st.session_state.people_layout = generate_random_people(
            room_w, room_d, n_standing, n_sitting, n_lying, st.session_state.layout_seed
        )
        st.session_state.last_layout_signature = signature


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


def compute_well_mixed_baseline_ppm(
    room_w: float,
    room_d: float,
    room_h: float,
    ach: float,
    people: List[Dict],
) -> Tuple[float, float, float]:
    volume_m3 = room_w * room_d * room_h
    q_vent_m3ph = max(ach * volume_m3, 0.001)
    g_total_m3ph = sum(p["co2_gen"] for p in people)

    delta_ppm = 1_000_000.0 * g_total_m3ph / q_vent_m3ph
    baseline_ppm = OUTDOOR_CO2_PPM + delta_ppm

    return volume_m3, q_vent_m3ph, baseline_ppm


def compute_co2_field(
    room_w: float,
    room_d: float,
    room_h: float,
    ach: float,
    people: List[Dict],
    equipments: List[Dict],
    grid_step_m: float,
):
    X, Y = make_grid(room_w, room_d, grid_step_m)

    volume_m3, q_vent_m3ph, baseline_ppm = compute_well_mixed_baseline_ppm(
        room_w, room_d, room_h, ach, people
    )

    Z = np.ones_like(X, dtype=float) * baseline_ppm

    for p in people:
        cx = p["x"] + p["w"] / 2.0
        cy = p["y"] + p["d"] / 2.0

        sigma = max((p["w"] + p["d"]) / 2.0, max(0.35, grid_step_m * 1.2))
        dist2 = (X - cx) ** 2 + (Y - cy) ** 2

        local_amp = 90.0 / max(math.sqrt(ach + 0.2), 0.5)
        if p["type"] == "standing":
            local_amp *= 1.05
        elif p["type"] == "lying":
            local_amp *= 0.9

        Z += local_amp * np.exp(-dist2 / (2.0 * sigma**2))

    for eq in equipments:
        cx = eq["x"] + eq["w"] / 2.0
        cy = eq["y"] + eq["d"] / 2.0
        sigma = max(eq["radius_m"], max(0.5, grid_step_m * 1.2))
        dist2 = (X - cx) ** 2 + (Y - cy) ** 2

        reduction = eq["ppm_reduction"] / max(1.0 + 0.15 * ach, 1.0)
        Z -= reduction * np.exp(-dist2 / (2.0 * sigma**2))

    Z = np.clip(Z, OUTDOOR_CO2_PPM, 5000.0)
    return X, Y, Z, volume_m3, q_vent_m3ph, baseline_ppm


def sample_people_for_drawing(people: List[Dict], max_draw: int = MAX_DRAWN_PEOPLE) -> Tuple[List[Dict], bool]:
    if len(people) <= max_draw:
        return people, False

    rng = random.Random(12345)
    sampled = rng.sample(people, max_draw)
    return sampled, True


def add_people_shapes(fig: go.Figure, people: List[Dict]) -> None:
    color_map = {
        "standing": "rgba(40,120,255,0.95)",
        "sitting": "rgba(50,170,80,0.95)",
        "lying": "rgba(255,150,40,0.95)",
    }

    for p in people:
        fig.add_shape(
            type="rect",
            x0=p["x"],
            y0=p["y"],
            x1=p["x"] + p["w"],
            y1=p["y"] + p["d"],
            line=dict(color="black", width=1),
            fillcolor=color_map[p["type"]],
        )

        fig.add_trace(
            go.Scatter(
                x=[p["x"] + p["w"] / 2.0],
                y=[p["y"] + p["d"] / 2.0],
                mode="text",
                text=[p["label"][0]],
                textfont=dict(color="white", size=9),
                showlegend=False,
                hoverinfo="skip",
            )
        )


def add_equipment_shapes(fig: go.Figure, equipments: List[Dict]) -> None:
    for eq in equipments:
        fig.add_shape(
            type="rect",
            x0=eq["x"] + eq["w"] * 0.18,
            y0=eq["y"] + eq["d"] * 0.18,
            x1=eq["x"] + eq["w"] * 0.82,
            y1=eq["y"] + eq["d"] * 0.82,
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


def add_editor_grid(fig: go.Figure, room_w: float, room_d: float, cell_size_m: float) -> None:
    xs = np.arange(0.0, room_w + 1e-9, cell_size_m)
    ys = np.arange(0.0, room_d + 1e-9, cell_size_m)

    for x in xs:
        fig.add_shape(
            type="line",
            x0=x,
            y0=0.0,
            x1=x,
            y1=room_d,
            line=dict(color="rgba(255,255,255,0.14)", width=1),
        )
    for y in ys:
        fig.add_shape(
            type="line",
            x0=0.0,
            y0=y,
            x1=room_w,
            y1=y,
            line=dict(color="rgba(255,255,255,0.14)", width=1),
        )


def make_heatmap_figure(
    room_w: float,
    room_d: float,
    X,
    Y,
    Z,
    drawn_people: List[Dict],
    equipments: List[Dict],
    cell_size_m: float,
) -> go.Figure:
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
            hovertemplate="x=%{x:.2f} m<br>y=%{y:.2f} m<br>CO2=%{z:.0f} ppm<extra></extra>",
        )
    )

    add_editor_grid(fig, room_w, room_d, cell_size_m)
    add_people_shapes(fig, drawn_people)
    add_equipment_shapes(fig, equipments)

    fig.update_layout(
        width=950,
        height=540,
        autosize=False,
        margin=dict(l=20, r=20, t=45, b=20),
        title="CO2 Heatmap (Fixed Size, adaptive resolution)",
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
        if key in st.session_state.equipment_map:
            del st.session_state.equipment_map[key]
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


def recommend_editor_cell_size(room_w: float, room_d: float, current_cell: float) -> float:
    area = room_w * room_d
    if area > 2_000_000:
        return max(current_cell, 50.0)
    if area > 500_000:
        return max(current_cell, 20.0)
    if area > 100_000:
        return max(current_cell, 10.0)
    if area > 20_000:
        return max(current_cell, 5.0)
    return current_cell


# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("Room Settings")

room_w = st.sidebar.slider("Room Width (m)", min_value=2.0, max_value=5000.0, value=10.0, step=1.0)
room_d = st.sidebar.slider("Room Depth (m)", min_value=2.0, max_value=1000.0, value=5.0, step=1.0)
room_h = st.sidebar.slider("Room Height (m)", min_value=2.0, max_value=6.0, value=3.0, step=0.1)

ach = st.sidebar.slider("Ventilation ACH (1/h)", min_value=0.1, max_value=20.0, value=3.0, step=0.1)

st.sidebar.header("Population")
n_standing = st.sidebar.number_input("Standing people", min_value=0, max_value=3000, value=5, step=1)
n_sitting = st.sidebar.number_input("Sitting people", min_value=0, max_value=3000, value=3, step=1)
n_lying = st.sidebar.number_input("Lying people", min_value=0, max_value=3000, value=1, step=1)

st.sidebar.header("Equipment Editor")
selected_tool = st.sidebar.radio(
    "Equipment Tool",
    ["Supply", "Exhaust", "Purifier", "Eraser"],
    index=0,
)

editor_step = st.sidebar.select_slider(
    "Cell Size (m)",
    options=[0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0, 50.0],
    value=DEFAULT_EDITOR_CELL_M,
)

btn_col1, btn_col2 = st.sidebar.columns(2)

with btn_col1:
    if st.button("Randomize Population"):
        st.session_state.layout_seed += 1

with btn_col2:
    if st.button("Clear Equipments"):
        st.session_state.equipment_map = {}

recommended_editor_step = recommend_editor_cell_size(room_w, room_d, editor_step)
trim_equipment_map(room_w, room_d, editor_step)

# =========================================================
# Main computation
# =========================================================
ensure_people_layout(room_w, room_d, n_standing, n_sitting, n_lying)

people = st.session_state.people_layout
equipments = equipment_list_from_map(st.session_state.equipment_map, room_w, room_d, editor_step)

grid_step_m = get_adaptive_grid_step(room_w, room_d)

X, Y, Z, volume_m3, q_vent_m3ph, baseline_ppm = compute_co2_field(
    room_w, room_d, room_h, ach, people, equipments, grid_step_m
)

drawn_people, people_sampled = sample_people_for_drawing(people, MAX_DRAWN_PEOPLE)

fig = make_heatmap_figure(room_w, room_d, X, Y, Z, drawn_people, equipments, editor_step)

# =========================================================
# Main UI
# =========================================================
st.title("CO2 Room Simulator")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Width", f"{room_w:.1f} m")
m2.metric("Depth", f"{room_d:.1f} m")
m3.metric("Height", f"{room_h:.1f} m")
m4.metric("Room Volume", f"{volume_m3:.1f} m³")
m5.metric("Population", f"{len(people)}")

s1, s2, s3 = st.columns(3)
s1.metric("Ventilation Flow", f"{q_vent_m3ph:.1f} m³/h")
s2.metric("Well-mixed baseline", f"{baseline_ppm:.0f} ppm")
s3.metric("Grid Step", f"{grid_step_m:.2f} m")

if len(people) < (n_standing + n_sitting + n_lying):
    st.warning("Room is crowded, so some people could not be placed without overlap.")

if people_sampled:
    st.info(f"Too many occupants to draw all at once, so only {MAX_DRAWN_PEOPLE} people are shown on the heatmap.")

if recommended_editor_step > editor_step:
    st.warning(
        f"This room is very large. Current editor cell size may feel too heavy. "
        f"Recommended Cell Size: {recommended_editor_step:.1f} m"
    )

st.plotly_chart(fig, use_container_width=False)

# =========================================================
# Equipment editor
# =========================================================
st.subheader("Equipment Cell Editor")
st.caption("Choose a tool in the sidebar, then click a cell below to place or erase equipment.")

legend_cols = st.columns(4)
legend_cols[0].markdown("**S** = Supply")
legend_cols[1].markdown("**E** = Exhaust")
legend_cols[2].markdown("**P** = Purifier")
legend_cols[3].markdown("**·** = Empty")

nx = max(1, int(room_w // editor_step))
ny = max(1, int(room_d // editor_step))
editor_buttons = nx * ny

if editor_buttons > MAX_EDITOR_BUTTONS:
    st.warning(
        f"Editor grid is too large to render safely ({editor_buttons} cells). "
        f"Increase Cell Size to reduce the number of cells."
    )
else:
    for iy in range(ny):
        cols = st.columns(nx + 1)
        cols[0].markdown(f"**{iy}**")
        for ix in range(nx):
            key = (ix, iy)
            current = st.session_state.equipment_map.get(key)

            if current is None:
                label = "·"
            else:
                label = EQUIPMENT_TYPES[current]["symbol"]

            if cols[ix + 1].button(
                label,
                key=f"cell_{ix}_{iy}_{editor_step}_{room_w}_{room_d}",
                use_container_width=True,
            ):
                toggle_equipment(ix, iy, selected_tool)
                st.rerun()

# =========================================================
# Details
# =========================================================
with st.expander("Placed People Details"):
    if people:
        show_limit = min(len(people), 300)
        st.write(f"Showing first {show_limit} of {len(people)} occupants.")
        for i, p in enumerate(people[:show_limit], 1):
            st.write(
                f"{i}. {p['label']} | x={p['x']:.2f} m, y={p['y']:.2f} m, "
                f"w={p['w']:.2f} m, d={p['d']:.2f} m"
            )
    else:
        st.write("No people placed.")

with st.expander("Placed Equipments"):
    if equipments:
        for i, eq in enumerate(equipments, 1):
            st.write(
                f"{i}. {eq['type']} | cell=({eq['ix']}, {eq['iy']}) | "
                f"x={eq['x']:.2f} m, y={eq['y']:.2f} m"
            )
    else:
        st.write("No equipments placed.")

with st.expander("Model Notes"):
    st.markdown(
        """
- All geometric inputs are in **meters**.
- The baseline CO2 is estimated with a simplified steady-state relation:
  - room volume,
  - ACH,
  - total occupant CO2 generation.
- The heatmap adds:
  - local increases around occupants,
  - local reductions around equipment.
- Large rooms use an **adaptive grid step** to prevent slowdown.
- If occupants are too many, only part of them are drawn, but all are still included in the calculation.
- This is a **comparative simulation model**, not CFD.
"""
    )
