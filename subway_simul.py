import streamlit as st
import numpy as np
import plotly.graph_objects as go
import random
import math

st.set_page_config(page_title="CO2 Room Simulator", layout="wide")

# =========================================================
# Basic constants
# =========================================================
GRID_STEP = 25  # spatial resolution
OUTDOOR_CO2 = 420
AIR_DENSITY_FACTOR = 1.0  # placeholder if you later want density correction

# Human footprint sizes (same unit as room width/depth)
# You can tune these if needed.
HUMAN_TYPES = {
    "standing": {
        "w": 45,
        "d": 45,
        "label": "Standing",
        "co2": 0.020,   # arbitrary per-step strength for heatmap source
    },
    "sitting": {
        "w": 45,
        "d": 60,
        "label": "Sitting",
        "co2": 0.018,
    },
    "lying": {
        "w": 50,
        "d": 180,
        "label": "Lying",
        "co2": 0.016,
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

        # lying people can rotate
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
            # If space is too crowded, just skip the remaining person
            # instead of crashing.
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


def compute_co2_field(room_w, room_d, room_h, ach, people, outdoor_co2=OUTDOOR_CO2):
    """
    Simple demonstrative field model:
    - Base concentration from occupancy and ventilation
    - Localized source bumps around people
    """
    X, Y = make_grid(room_w, room_d, GRID_STEP)

    volume = max(room_w * room_d * room_h, 1.0)

    total_source = sum(p["co2"] for p in people)
    n_people = len(people)

    # Global baseline rise
    # You can replace this with your own research formula later.
    baseline_rise = (n_people * 45) / max(ach, 0.2)

    Z = np.ones_like(X, dtype=float) * (outdoor_co2 + baseline_rise)

    # Add local contributions around each person
    for p in people:
        cx = p["x"] + p["w"] / 2
        cy = p["y"] + p["d"] / 2

        sigma = max((p["w"] + p["d"]) / 2.2, 40)
        dist2 = (X - cx) ** 2 + (Y - cy) ** 2

        local_strength = 350 * p["co2"] / max(ach, 0.3)
        Z += local_strength * np.exp(-dist2 / (2 * sigma ** 2))

    # Mild clipping for readable heatmap
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
                textfont=dict(color="white", size=12),
                showlegend=False,
                hoverinfo="skip",
            )
        )


def make_heatmap_figure(room_w, room_d, X, Y, Z, people):
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

    add_people_shapes(fig, people)

    # Fixed size: this is the important part
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

col_btn1, col_btn2 = st.sidebar.columns(2)

with col_btn1:
    if st.button("Randomize Population"):
        st.session_state.layout_seed += 1

with col_btn2:
    if st.button("Clear Population"):
        st.session_state.people_layout = []
        st.session_state.last_layout_signature = ("cleared",)

# Automatically regenerate when room size / counts / seed changes
ensure_people_layout(room_w, room_d, n_standing, n_sitting, n_lying)
people = st.session_state.people_layout

# =========================================================
# Main view
# =========================================================
st.title("CO2 Room Simulator")

info1, info2, info3, info4 = st.columns(4)
info1.metric("Width", f"{room_w}")
info2.metric("Depth", f"{room_d}")
info3.metric("Height", f"{room_h}")
info4.metric("Placed Population", f"{len(people)}")

if len(people) < (n_standing + n_sitting + n_lying):
    st.warning("Some people could not be placed because the room is too crowded for non-overlapping random placement.")

X, Y, Z = compute_co2_field(room_w, room_d, room_h, ach, people)

fig = make_heatmap_figure(room_w, room_d, X, Y, Z, people)

# Fixed size on screen
st.plotly_chart(fig, use_container_width=False)

with st.expander("Placed People Details"):
    if people:
        for i, p in enumerate(people, 1):
            st.write(
                f"{i}. {p['label']} | x={p['x']:.1f}, y={p['y']:.1f}, "
                f"w={p['w']}, d={p['d']}"
            )
    else:
        st.write("No people placed.")
