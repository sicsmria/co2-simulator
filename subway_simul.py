import io
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide", page_title="CO2 Grid Simulation")

# =========================================================
# Config
# =========================================================
PX_PER_METER = 45  # image scaling for click map


# =========================================================
# Helpers
# =========================================================
def get_activity_co2_rate(activity_type: str) -> float:
    rates = {
        "Stable": 0.018,             # m3/h/person
        "Crowded / Uneasy": 0.024,
        "Moving": 0.032
    }
    return rates.get(activity_type, 0.018)


def create_grid(room_w, room_d, cell_size):
    nx = max(1, int(room_w / cell_size))
    ny = max(1, int(room_d / cell_size))
    return nx, ny


def pos_to_idx(x, y, cell_size, nx, ny):
    ix = min(max(int(x / cell_size), 0), nx - 1)
    iy = min(max(int(y / cell_size), 0), ny - 1)
    return ix, iy


def diffuse(C, diffusion_strength=0.15):
    C_new = C.copy()
    nx, ny = C.shape

    for i in range(nx):
        for j in range(ny):
            neighbors = []
            if i > 0:
                neighbors.append(C[i - 1, j])
            if i < nx - 1:
                neighbors.append(C[i + 1, j])
            if j > 0:
                neighbors.append(C[i, j - 1])
            if j < ny - 1:
                neighbors.append(C[i, j + 1])

            if neighbors:
                neighbor_avg = np.mean(neighbors)
                C_new[i, j] += diffusion_strength * (neighbor_avg - C[i, j])

    return C_new


def apply_people_source(C, people, cell_size, nx, ny, dt_h, room_h):
    cell_volume = cell_size * cell_size * room_h

    for p in people:
        ix, iy = pos_to_idx(p["x"], p["y"], cell_size, nx, ny)
        G = get_activity_co2_rate(p["activity"])  # m3/h
        delta_ppm = (G / cell_volume) * 1e6 * dt_h
        C[ix, iy] += delta_ppm

    return C


def apply_vents(C, vents, outdoor_co2, cell_size, nx, ny, dt_h, room_h):
    cell_volume = cell_size * cell_size * room_h

    for v in vents:
        ix, iy = pos_to_idx(v["x"], v["y"], cell_size, nx, ny)
        local_exchange = (v["flow"] / cell_volume) * dt_h
        local_exchange = min(local_exchange, 1.0)

        # simple local effect
        if v["type"] == "supply":
            C[ix, iy] = C[ix, iy] - local_exchange * (C[ix, iy] - outdoor_co2)

            # small neighborhood support
            for dx, dy, factor in [(-1, 0, 0.35), (1, 0, 0.35), (0, -1, 0.35), (0, 1, 0.35)]:
                ni, nj = ix + dx, iy + dy
                if 0 <= ni < nx and 0 <= nj < ny:
                    C[ni, nj] = C[ni, nj] - local_exchange * factor * (C[ni, nj] - outdoor_co2)

        elif v["type"] == "exhaust":
            C[ix, iy] = C[ix, iy] - local_exchange * (C[ix, iy] - outdoor_co2)

            # slightly stronger draw from nearby cells
            for dx, dy, factor in [(-1, 0, 0.45), (1, 0, 0.45), (0, -1, 0.45), (0, 1, 0.45)]:
                ni, nj = ix + dx, iy + dy
                if 0 <= ni < nx and 0 <= nj < ny:
                    C[ni, nj] = C[ni, nj] - local_exchange * factor * (C[ni, nj] - outdoor_co2)

    return C


def simulate_grid(
    room_w, room_d, room_h,
    cell_size,
    initial_co2,
    outdoor_co2,
    people,
    vents,
    sim_hours,
    dt_minutes
):
    nx, ny = create_grid(room_w, room_d, cell_size)
    C = np.full((nx, ny), initial_co2, dtype=float)

    dt_h = dt_minutes / 60.0
    steps = max(1, int(sim_hours / dt_h))

    history = []

    for _ in range(steps):
        C = apply_people_source(C, people, cell_size, nx, ny, dt_h, room_h)
        C = apply_vents(C, vents, outdoor_co2, cell_size, nx, ny, dt_h, room_h)
        C = diffuse(C, diffusion_strength=0.20)
        history.append(C.copy())

    return C, history, nx, ny


def auto_generate_people(room_w, room_d, standing_n, sitting_n, lying_n,
                         standing_state, sitting_state, lying_state):
    people = []

    # standing zone
    standing_cols = max(1, int(max(1.0, room_w - 1.0) // 1.0))
    for i in range(standing_n):
        row = i // standing_cols
        col = i % standing_cols
        x = min(0.8 + col * 1.0, room_w - 0.6)
        y = min(1.2 + row * 1.0, max(1.2, room_d * 0.25))
        people.append({
            "type": "standing",
            "x": x,
            "y": y,
            "activity": standing_state
        })

    # sitting zone
    sitting_cols = max(1, int(max(1.0, room_w - 1.0) // 1.1))
    for i in range(sitting_n):
        row = i // sitting_cols
        col = i % sitting_cols
        x = min(0.8 + col * 1.1, room_w - 0.6)
        y = min(room_d * 0.50 + row * 1.0, room_d - 1.0)
        people.append({
            "type": "sitting",
            "x": x,
            "y": y,
            "activity": sitting_state
        })

    # lying zone
    lying_cols = max(1, int(max(1.0, room_w - 1.0) // 1.3))
    for i in range(lying_n):
        row = i // lying_cols
        col = i % lying_cols
        x = min(1.0 + col * 1.3, room_w - 0.8)
        y = min(room_d * 0.78 + row * 0.9, room_d - 0.8)
        people.append({
            "type": "lying",
            "x": x,
            "y": y,
            "activity": lying_state
        })

    return people


def plot_heatmap(C, room_w, room_d, people, vents, vmin=400, vmax=3000):
    fig, ax = plt.subplots(figsize=(8.5, 6.2))

    im = ax.imshow(
        C.T,
        origin="lower",
        extent=[0, room_w, 0, room_d],
        aspect="auto",
        vmin=vmin,
        vmax=vmax
    )

    # red risk contour
    risk = np.where(C.T > 1000, 1, 0)
    ax.contour(
        np.linspace(0, room_w, C.shape[0]),
        np.linspace(0, room_d, C.shape[1]),
        risk,
        levels=[0.5],
        linewidths=1.5
    )

    # people
    for p in people:
        if p["type"] == "standing":
            ax.scatter(p["x"], p["y"], marker="o", s=70)
        elif p["type"] == "sitting":
            ax.scatter(p["x"], p["y"], marker="s", s=70)
        else:
            ax.scatter(p["x"], p["y"], marker="_", s=220)

    # vents
    for idx, v in enumerate(vents, start=1):
        if v["type"] == "supply":
            ax.scatter(v["x"], v["y"], marker="^", s=170)
            ax.text(v["x"], v["y"] + 0.22, f"S{idx}", ha="center", fontsize=9)
        else:
            ax.scatter(v["x"], v["y"], marker="v", s=170)
            ax.text(v["x"], v["y"] + 0.22, f"E{idx}", ha="center", fontsize=9)

    ax.set_title("CO2 Distribution Heatmap")
    ax.set_xlabel("Width (m)")
    ax.set_ylabel("Depth (m)")
    plt.colorbar(im, ax=ax, label="CO2 (ppm)")
    return fig


def render_clickable_layout(room_w, room_d, people, vents):
    fig_w = max(5.5, room_w * 0.55)
    fig_h = max(4.0, room_d * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)

    ax.set_xlim(0, room_w)
    ax.set_ylim(0, room_d)
    ax.set_aspect("equal")
    ax.set_title("Click Map: add a vent by clicking")
    ax.set_xlabel("Width (m)")
    ax.set_ylabel("Depth (m)")
    ax.grid(True, alpha=0.25)

    # room outline
    ax.plot([0, room_w, room_w, 0, 0], [0, 0, room_d, room_d, 0], linewidth=2)

    # people
    for p in people:
        if p["type"] == "standing":
            ax.scatter(p["x"], p["y"], marker="o", s=80)
        elif p["type"] == "sitting":
            ax.scatter(p["x"], p["y"], marker="s", s=80)
        else:
            ax.scatter(p["x"], p["y"], marker="_", s=220)

    # vents
    for idx, v in enumerate(vents, start=1):
        if v["type"] == "supply":
            ax.scatter(v["x"], v["y"], marker="^", s=180)
            ax.text(v["x"], v["y"] + 0.18, f"S{idx}", ha="center", fontsize=9)
        else:
            ax.scatter(v["x"], v["y"], marker="v", s=180)
            ax.text(v["x"], v["y"] + 0.18, f"E{idx}", ha="center", fontsize=9)

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    image = Image.open(buf)
    return image


def pixel_to_room_coords(px, py, image_w, image_h, room_w, room_d):
    # image origin: top-left
    # room origin: bottom-left
    x = (px / image_w) * room_w
    y = room_d - (py / image_h) * room_d
    x = min(max(x, 0.0), room_w)
    y = min(max(y, 0.0), room_d)
    return x, y


def add_clicked_vent(click_data, image_size, room_w, room_d, vent_type, vent_flow):
    if not click_data:
        return

    if "vent_click_guard" not in st.session_state:
        st.session_state.vent_click_guard = None

    click_stamp = (click_data.get("x"), click_data.get("y"), click_data.get("time"))
    if st.session_state.vent_click_guard == click_stamp:
        return

    st.session_state.vent_click_guard = click_stamp

    img_w, img_h = image_size
    x_m, y_m = pixel_to_room_coords(
        click_data["x"], click_data["y"], img_w, img_h, room_w, room_d
    )

    st.session_state.vents.append({
        "type": vent_type,
        "x": round(x_m, 2),
        "y": round(y_m, 2),
        "flow": int(vent_flow)
    })


# =========================================================
# Session state
# =========================================================
if "vents" not in st.session_state:
    st.session_state.vents = []

if "vent_click_guard" not in st.session_state:
    st.session_state.vent_click_guard = None


# =========================================================
# Layout
# =========================================================
st.title("Indoor CO2 Grid Simulation")

left, right = st.columns([1, 2])

with left:
    st.subheader("Settings")

    st.markdown("### Room")
    room_w = st.slider("Room width (m)", 4.0, 30.0, 12.0, 0.5)
    room_d = st.slider("Room depth (m)", 4.0, 30.0, 10.0, 0.5)
    room_h = st.slider("Room height (m)", 2.0, 6.0, 3.0, 0.1)
    cell_size = st.slider("Grid cell size (m)", 0.5, 2.0, 1.0, 0.5)

    st.markdown("### Air / Simulation")
    initial_co2 = st.number_input("Initial indoor CO2 (ppm)", 300, 3000, 450)
    outdoor_co2 = st.number_input("Outdoor CO2 (ppm)", 300, 700, 420)
    sim_hours = st.slider("Simulation time (h)", 0.5, 8.0, 2.0, 0.5)
    dt_minutes = st.slider("Time step (min)", 1, 20, 5)

    st.markdown("### Standing")
    standing_n = st.slider("Standing people", 0, 40, 2)
    standing_state = st.selectbox("Standing activity", ["Stable", "Crowded / Uneasy", "Moving"])

    st.markdown("### Sitting")
    sitting_n = st.slider("Sitting people", 0, 40, 2)
    sitting_state = st.selectbox("Sitting activity", ["Stable", "Crowded / Uneasy", "Moving"])

    st.markdown("### Lying")
    lying_n = st.slider("Lying people", 0, 40, 1)
    lying_state = st.selectbox("Lying activity", ["Stable", "Crowded / Uneasy", "Moving"])

    st.markdown("### Click-to-add vent")
    pending_vent_type = st.radio("New vent type", ["supply", "exhaust"], horizontal=True)
    pending_vent_flow = st.slider("New vent flow (m3/h)", 50, 3000, 400, 50)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Undo last vent", use_container_width=True):
            if st.session_state.vents:
                st.session_state.vents.pop()
    with c2:
        if st.button("Clear vents", use_container_width=True):
            st.session_state.vents = []

    st.markdown("### Current vents")
    if st.session_state.vents:
        for i, v in enumerate(st.session_state.vents, start=1):
            st.caption(
                f"{i}. {v['type']} | x={v['x']:.2f} m | y={v['y']:.2f} m | flow={v['flow']} m³/h"
            )
    else:
        st.caption("No vents yet. Click on the layout at right to add one.")


# =========================================================
# Recompute every rerun
# =========================================================
people = auto_generate_people(
    room_w, room_d,
    standing_n, sitting_n, lying_n,
    standing_state, sitting_state, lying_state
)

# clickable layout image
layout_img = render_clickable_layout(room_w, room_d, people, st.session_state.vents)

with right:
    st.subheader("Results")

    st.markdown("### Layout editor")
    click_data = streamlit_image_coordinates(
        layout_img,
        key=f"layout_click_{room_w}_{room_d}_{len(st.session_state.vents)}"
    )

    add_clicked_vent(
        click_data=click_data,
        image_size=layout_img.size,
        room_w=room_w,
        room_d=room_d,
        vent_type=pending_vent_type,
        vent_flow=pending_vent_flow
    )

    # use updated vents after possible click
    vents = st.session_state.vents

    C, history, nx, ny = simulate_grid(
        room_w, room_d, room_h,
        cell_size,
        initial_co2,
        outdoor_co2,
        people,
        vents,
        sim_hours,
        dt_minutes
    )

    avg_co2 = float(np.mean(C))
    max_co2 = float(np.max(C))
    risk_ratio = float(np.sum(C > 1000) / C.size * 100.0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Average CO2", f"{avg_co2:.0f} ppm")
    m2.metric("Max CO2", f"{max_co2:.0f} ppm")
    m3.metric("Area >1000 ppm", f"{risk_ratio:.1f}%")

    st.markdown("### CO2 heatmap")
    fig = plot_heatmap(C, room_w, room_d, people, vents)
    st.pyplot(fig, use_container_width=True)

    st.markdown("### Notes")
    st.caption("Standing=o, Sitting=square, Lying=horizontal mark")
    st.caption("Supply=triangle up, Exhaust=triangle down")
