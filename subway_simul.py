import streamlit as st
import numpy as np
import plotly.graph_objects as go
import random
import math

st.set_page_config(page_title="CO2 Room Simulator", layout="wide")

# =========================================================
# Constants
# =========================================================
GRID_STEP = 25
EDITOR_STEP = 100   # 설비 배치용 셀 크기
OUTDOOR_CO2 = 420

# 사람 크기 축소
HUMAN_TYPES = {
    "standing": {
        "w": 25,
        "d": 25,
        "label": "Standing",
        "co2": 0.020,
    },
    "sitting": {
        "w": 30,
        "d": 35,
        "label": "Sitting",
        "co2": 0.018,
    },
    "lying": {
        "w": 30,
        "d": 90,
        "label": "Lying",
        "co2": 0.016,
    },
}

EQUIPMENT_TYPES = {
    "Supply": {
        "color": "rgba(0,150,255,0.95)",
        "symbol": "S",
        "effect": -180,
    },
    "Exhaust": {
        "color": "rgba(255,80,80,0.95)",
        "symbol": "E",
        "effect": -220,
    },
    "Purifier": {
        "color": "rgba(0,180,120,0.95)",
        "symbol": "P",
        "effect": -130,
    },
}

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
    # {(ix, iy): "Supply"/"Exhaust"/"Purifier"}
    st.session_state.equipment_map = {}

# =========================================================
# Helpers
# =========================================================
def rects_overlap(a, b):
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["d"]

    bx1, by1 = b["x"], b["y"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["d"]

    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def can_place(candidate, placed):
    for p in placed:
        if rects_overlap(candidate, p):
            return False
    return True


def generate_random_people(room_w, room_d, n_standing, n_sitting, n_lying, seed=0):
    rng = random.Random(seed)
    placed = []

    request = (
        [("standing", i) for i in range(n_standing)] +
        [("sitting", i) for i in range(n_sitting)] +
        [("lying", i) for i in range(n_lying)]
    )
    rng.shuffle(request)

    for kind, _ in request:
        spec = HUMAN_TYPES[kind]
        pw, pd = spec["w"], spec["d"]

        orientations = [(pw, pd)]
        if kind == "lying":
            orientations.append((pd, pw))

        success = False
        for _ in range(300):
            ow, od = rng.choice(orientations)

            if room_w <= ow or room_d <= od:
                continue

            x = rng.uniform(0, room_w - ow)
            y = rng.uniform(0, room_d - od)

            candidate = {
                "type": kind,
                "x": x,
                "y": y,
                "w": ow,
                "d": od,
                "co2": spec["co2"],
                "label": spec["label"],
            }

            if can_place(candidate, placed):
                placed.append(candidate)
                success = True
                break

        if not success:
            pass

    return placed


def ensure_people_layout(room_w, room_d, n_standing, n_sitting, n_lying):
    signature = (room_w, room_d, n_standing, n_sitting, n_lying, st.session_state.layout_seed)
    if st.session_state.last_layout_signature != signature:
        st.session_state.people_layout = generate_random_people(
            room_w, room_d, n_standing, n_sitting, n_lying, seed=st.session_state.layout_seed
        )
        st.session_state.last_layout_signature = signature


def make_grid(room_w, room_d, step=GRID_STEP):
    nx = max(2, int(math.ceil(room_w / step)) + 1)
    ny = max(2, int(math.ceil(room_d / step)) + 1)
    xs = np.linspace(0, room_w, nx)
    ys = np.linspace(0, room_d, ny)
    X, Y = np.meshgrid(xs, ys)
    return X, Y


def equipment_list_from_map(equipment_map, room_w, room_d, cell_size):
    equipments = []
    nx = int(room_w // cell_size)
    ny = int(room_d // cell_size)

    for (ix, iy), eq_type in equipment_map.items():
        if 0 <= ix < nx and 0 <= iy < ny:
            x0 = ix * cell_size
            y0 = iy * cell_size
            equipments.append({
                "ix": ix,
                "iy": iy,
                "x": x0,
                "y": y0,
                "w": cell_size,
                "d": cell_size,
                "type": eq_type,
                "effect": EQUIPMENT_TYPES[eq_type]["effect"],
                "symbol": EQUIPMENT_TYPES[eq_type]["symbol"],
                "color": EQUIPMENT_TYPES[eq_type]["color"],
            })
    return equipments


def compute_co2_field(room_w, room_d, room_h, ach, people, equipments, outdoor_co2=OUTDOOR_CO2):
    X, Y = make_grid(room_w, room_d, GRID_STEP)

    n_people = len(people)
    baseline_rise = (n_people * 45) / max(ach, 0.2)
    Z = np.ones_like(X, dtype=float) * (outdoor_co2 + baseline_rise)

    # 사람 영향
    for p in people:
        cx = p["x"] + p["w"] / 2
        cy = p["y"] + p["d"] / 2

        sigma = max((p["w"] + p["d"]) / 2.2, 25)
        dist2 = (X - cx) ** 2 + (Y - cy) ** 2

        local_strength = 350 * p["co2"] / max(ach, 0.3)
        Z += local_strength * np.exp(-dist2 / (2 * sigma ** 2))

    # 설비 영향 (아주 러프)
    for eq in equipments:
        cx = eq["x"] + eq["w"] / 2
        cy = eq["y"] + eq["d"] / 2

        if eq["type"] == "Supply":
            sigma = eq["w"] * 1.4
        elif eq["type"] == "Exhaust":
            sigma = eq["w"] * 1.6
        else:  # Purifier
            sigma = eq["w"] * 1.2

        dist2 = (X - cx) ** 2 + (Y - cy) ** 2
        Z += eq["effect"] * np.exp(-dist2 / (2 * sigma ** 2))

    Z = np.clip(Z, outdoor_co2, 5000)
    return X, Y, Z


def add_people_shapes(fig, people):
    color_map = {
        "standing": "rgba(30,144,255,0.90)",
        "sitting": "rgba(34,139,34,0.90)",
        "lying": "rgba(255,140,0,0.90)",
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
                x=[p["x"] + p["w"] / 2],
                y=[p["y"] + p["d"] / 2],
                mode="text",
                text=[p["label"][0]],
                textfont=dict(color="white", size=10),
                showlegend=False,
                hoverinfo="skip",
            )
        )


def add_equipment_shapes(fig, equipments):
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
                x=[eq["x"] + eq["w"] / 2],
                y=[eq["y"] + eq["d"] / 2],
                mode="text",
                text=[eq["symbol"]],
                textfont=dict(color="white", size=12),
                showlegend=False,
                hoverinfo="skip",
            )
        )


def add_editor_grid(fig, room_w, room_d, cell_size):
    for x in np.arange(0, room_w + 0.1, cell_size):
        fig.add_shape(
            type="line",
            x0=x, y0=0,
            x1=x, y1=room_d,
            line=dict(color="rgba(255,255,255,0.15)", width=1)
        )

    for y in np.arange(0, room_d + 0.1, cell_size):
        fig.add_shape(
            type="line",
            x0=0, y0=y,
            x1=room_w, y1=y,
            line=dict(color="rgba(255,255,255,0.15)", width=1)
        )


def make_heatmap_figure(room_w, room_d, X, Y, Z, people, equipments, cell_size):
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
            hovertemplate="x=%{x:.0f}<br>y=%{y:.0f}<br>CO2=%{z:.0f} ppm<extra></extra>",
        )
    )

    add_editor_grid(fig, room_w, room_d, cell_size)
    add_people_shapes(fig, people)
    add_equipment_shapes(fig, equipments)

    fig.update_layout(
        width=950,
        height=520,
        autosize=False,
        margin=dict(l=20, r=20, t=40, b=20),
        title="CO2 Heatmap (Fixed Size)",
    )

    fig.update_xaxes(
        title="Width",
        range=[0, room_w],
        constrain="domain",
        scaleanchor="y",
        scaleratio=1,
    )
    fig.update_yaxes(
        title="Depth",
        range=[0, room_d],
        autorange="reversed",
    )
    return fig


def toggle_equipment(ix, iy, selected_tool):
    key = (ix, iy)
    if selected_tool == "Eraser":
        if key in st.session_state.equipment_map:
            del st.session_state.equipment_map[key]
    else:
        st.session_state.equipment_map[key] = selected_tool


# =========================================================
# Sidebar controls
# =========================================================
st.sidebar.header("Room Settings")
room_w = st.sidebar.slider("Room Width", min_value=100, max_value=3000, value=1000, step=50)
room_d = st.sidebar.slider("Room Depth", min_value=100, max_value=1000, value=500, step=25)
room_h = st.sidebar.slider("Room Height", min_value=200, max_value=1000, value=300, step=10)

ach = st.sidebar.slider("Ventilation (ACH)", min_value=0.1, max_value=20.0, value=3.0, step=0.1)

st.sidebar.header("Population")
n_standing = st.sidebar.number_input("Standing people", min_value=0, max_value=200, value=5, step=1)
n_sitting = st.sidebar.number_input("Sitting people", min_value=0, max_value=200, value=3, step=1)
n_lying = st.sidebar.number_input("Lying people", min_value=0, max_value=200, value=1, step=1)

st.sidebar.header("Equipment Editor")
selected_tool = st.sidebar.radio(
    "Equipment Tool",
    ["Supply", "Exhaust", "Purifier", "Eraser"],
    index=0
)

editor_step = st.sidebar.select_slider(
    "Cell Size",
    options=[50, 100, 150, 200],
    value=100
)

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("Randomize Population"):
        st.session_state.layout_seed += 1

with col_btn2:
    if st.button("Clear Equipments"):
        st.session_state.equipment_map = {}

# 방 크기 줄었을 때 범위 밖 설비 삭제
max_ix = int(room_w // editor_step)
max_iy = int(room_d // editor_step)
st.session_state.equipment_map = {
    (ix, iy): eq
    for (ix, iy), eq in st.session_state.equipment_map.items()
    if ix < max_ix and iy < max_iy
}

ensure_people_layout(room_w, room_d, n_standing, n_sitting, n_lying)
people = st.session_state.people_layout
equipments = equipment_list_from_map(st.session_state.equipment_map, room_w, room_d, editor_step)

# =========================================================
# Main
# =========================================================
st.title("CO2 Room Simulator")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Width", f"{room_w}")
m2.metric("Depth", f"{room_d}")
m3.metric("Height", f"{room_h}")
m4.metric("Population", f"{len(people)}")

if len(people) < (n_standing + n_sitting + n_lying):
    st.warning("Some people could not be placed because the room is too crowded.")

X, Y, Z = compute_co2_field(room_w, room_d, room_h, ach, people, equipments)
fig = make_heatmap_figure(room_w, room_d, X, Y, Z, people, equipments, editor_step)

st.plotly_chart(fig, use_container_width=False)

st.subheader("Equipment Cell Editor")
st.caption("Select a tool in the sidebar, then click a cell below.")

nx = max(1, int(room_w // editor_step))
ny = max(1, int(room_d // editor_step))

legend_cols = st.columns(4)
legend_cols[0].markdown("**S** = Supply")
legend_cols[1].markdown("**E** = Exhaust")
legend_cols[2].markdown("**P** = Purifier")
legend_cols[3].markdown("**·** = Empty")

# 너무 많으면 약간 경고
if nx > 35:
    st.info("The room is wide, so the editor may feel a bit heavy. Increase Cell Size if it becomes slow.")

# 위에서 아래로 보기 좋게 y행 순서대로 표시
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

        if cols[ix + 1].button(label, key=f"cell_{ix}_{iy}_{editor_step}_{room_w}_{room_d}"):
            toggle_equipment(ix, iy, selected_tool)
            st.rerun()

with st.expander("Placed People Details"):
    if people:
        for i, p in enumerate(people, 1):
            st.write(
                f"{i}. {p['label']} | x={p['x']:.1f}, y={p['y']:.1f}, w={p['w']}, d={p['d']}"
            )
    else:
        st.write("No people placed.")

with st.expander("Placed Equipments"):
    if equipments:
        for i, eq in enumerate(equipments, 1):
            st.write(
                f"{i}. {eq['type']} | cell=({eq['ix']}, {eq['iy']}) | "
                f"x={eq['x']:.0f}, y={eq['y']:.0f}"
            )
    else:
        st.write("No equipments placed.")
