import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide", page_title="CO2 Grid Simulation")


# =========================================================
# Basic helpers
# =========================================================
def get_activity_co2_rate(activity_type: str) -> float:
    rates = {
        "Stable": 0.018,             # m3/h/person
        "Crowded / Uneasy": 0.024,
        "Moving": 0.032
    }
    return rates.get(activity_type, 0.018)


def create_grid(room_w, room_d, cell_size):
    nx = max(1, int(round(room_w / cell_size)))
    ny = max(1, int(round(room_d / cell_size)))
    return nx, ny


def pos_to_idx(x, y, cell_size, nx, ny):
    ix = min(max(int(x / cell_size), 0), nx - 1)
    iy = min(max(int(y / cell_size), 0), ny - 1)
    return ix, iy


def idx_to_center(ix, iy, cell_size, room_w, room_d):
    x = min((ix + 0.5) * cell_size, room_w)
    y = min((iy + 0.5) * cell_size, room_d)
    return round(x, 2), round(y, 2)


def snap_to_cell_center(x, y, cell_size, room_w, room_d):
    nx, ny = create_grid(room_w, room_d, cell_size)

    ix = int(np.floor(x / cell_size))
    iy = int(np.floor(y / cell_size))

    ix = min(max(ix, 0), nx - 1)
    iy = min(max(iy, 0), ny - 1)

    x_snap, y_snap = idx_to_center(ix, iy, cell_size, room_w, room_d)
    return x_snap, y_snap, ix, iy


def nearest_vent_index(x, y, vents):
    if not vents:
        return None

    best_idx = None
    best_dist = None
    for idx, v in enumerate(vents):
        d = (v["x"] - x) ** 2 + (v["y"] - y) ** 2
        if best_dist is None or d < best_dist:
            best_dist = d
            best_idx = idx
    return best_idx


def vent_exists_at(x, y, vents, tol=1e-9):
    return any(abs(v["x"] - x) < tol and abs(v["y"] - y) < tol for v in vents)


# =========================================================
# Physics / simulation
# =========================================================
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
        G = get_activity_co2_rate(p["activity"])
        delta_ppm = (G / cell_volume) * 1e6 * dt_h
        C[ix, iy] += delta_ppm

    return C


def apply_vents(C, vents, outdoor_co2, cell_size, nx, ny, dt_h, room_h):
    cell_volume = cell_size * cell_size * room_h

    for v in vents:
        ix, iy = pos_to_idx(v["x"], v["y"], cell_size, nx, ny)
        local_exchange = min((v["flow"] / cell_volume) * dt_h, 1.0)

        C[ix, iy] = C[ix, iy] - local_exchange * (C[ix, iy] - outdoor_co2)

        factor = 0.35 if v["type"] == "supply" else 0.45
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
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

    for _ in range(steps):
        C = apply_people_source(C, people, cell_size, nx, ny, dt_h, room_h)
        C = apply_vents(C, vents, outdoor_co2, cell_size, nx, ny, dt_h, room_h)
        C = diffuse(C, diffusion_strength=0.20)

    return C


# =========================================================
# Automatic people placement
# =========================================================
def auto_generate_people(
    room_w, room_d,
    standing_n, sitting_n, lying_n,
    standing_state, sitting_state, lying_state
):
    people = []

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


# =========================================================
# Coordinate conversion
# =========================================================
def pixel_to_room_coords(px, py, img_w, img_h, room_w, room_d):
    # axes fills the whole image, so direct mapping is valid
    x = (px / img_w) * room_w
    y = room_d - (py / img_h) * room_d
    x = min(max(x, 0.0), room_w)
    y = min(max(y, 0.0), room_d)
    return x, y


# =========================================================
# Rendering
# =========================================================
def draw_person_on_ax(ax, person, cell_size):
    if person["type"] == "standing":
        ax.scatter(person["x"], person["y"], marker="o", s=70)
    elif person["type"] == "sitting":
        ax.scatter(person["x"], person["y"], marker="s", s=70)
    else:
        body_w = min(0.70, cell_size * 0.75)
        body_h = min(0.22, cell_size * 0.28)
        ax.add_patch(
            Rectangle(
                (person["x"] - body_w / 2, person["y"] - body_h / 2),
                body_w,
                body_h,
                fill=True,
                alpha=0.85
            )
        )
        ax.add_patch(
            Circle(
                (person["x"] - body_w / 2 - 0.08, person["y"]),
                radius=min(0.08, cell_size * 0.10),
                fill=True,
                alpha=0.85
            )
        )


def render_layout_image(room_w, room_d, cell_size, people, vents):
    max_side = max(room_w, room_d)
    fig_w = 8.0 * (room_w / max_side)
    fig_h = 8.0 * (room_d / max_side)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)

    ax.set_xlim(0, room_w)
    ax.set_ylim(0, room_d)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_position([0, 0, 1, 1])

    x_lines = np.arange(0, room_w + cell_size, cell_size)
    y_lines = np.arange(0, room_d + cell_size, cell_size)

    for x in x_lines:
        ax.plot([x, x], [0, room_d], linewidth=0.6, alpha=0.20)

    for y in y_lines:
        ax.plot([0, room_w], [y, y], linewidth=0.6, alpha=0.20)

    ax.plot([0, room_w, room_w, 0, 0], [0, 0, room_d, room_d, 0], linewidth=2.0)

    nx, ny = create_grid(room_w, room_d, cell_size)
    for i in range(nx):
        for j in range(ny):
            cx, cy = idx_to_center(i, j, cell_size, room_w, room_d)
            ax.scatter(cx, cy, s=8, alpha=0.10)

    for p in people:
        draw_person_on_ax(ax, p, cell_size)

    supply_idx = 1
    exhaust_idx = 1
    for v in vents:
        if v["type"] == "supply":
            ax.scatter(v["x"], v["y"], marker="^", s=160)
            ax.text(v["x"], v["y"] + 0.12, f"S{supply_idx}", ha="center", va="bottom", fontsize=9)
            supply_idx += 1
        else:
            ax.scatter(v["x"], v["y"], marker="v", s=160)
            ax.text(v["x"], v["y"] + 0.12, f"E{exhaust_idx}", ha="center", va="bottom", fontsize=9)
            exhaust_idx += 1

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def plot_heatmap(C, room_w, room_d, cell_size, vents):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)

    im = ax.imshow(
        C.T,
        origin="lower",
        extent=[0, room_w, 0, room_d],
        aspect="auto",
        vmin=400,
        vmax=3000
    )

    x_lines = np.arange(0, room_w + cell_size, cell_size)
    y_lines = np.arange(0, room_d + cell_size, cell_size)

    for x in x_lines:
        ax.plot([x, x], [0, room_d], linewidth=0.4, alpha=0.12)

    for y in y_lines:
        ax.plot([0, room_w], [y, y], linewidth=0.4, alpha=0.12)

    for v in vents:
        if v["type"] == "supply":
            ax.scatter(v["x"], v["y"], marker="^", s=120)
        else:
            ax.scatter(v["x"], v["y"], marker="v", s=120)

    ax.set_title("CO2 heatmap")
    ax.set_xlabel("Width (m)")
    ax.set_ylabel("Depth (m)")
    plt.colorbar(im, ax=ax, label="CO2 (ppm)")
    return fig


# =========================================================
# Session state
# =========================================================
if "vents" not in st.session_state:
    st.session_state.vents = []

if "last_click" not in st.session_state:
    st.session_state.last_click = None


# =========================================================
# UI
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

    st.markdown("### Edit mode")
    edit_mode = st.radio("Mode", ["Add vent", "Delete vent"], horizontal=True)

    st.markdown("### New vent settings")
    pending_vent_type = st.radio("New vent type", ["supply", "exhaust"], horizontal=True)
    pending_vent_flow = st.slider("New vent flow (m³/h)", 50, 3000, 400, 50)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Undo last vent", use_container_width=True):
            if st.session_state.vents:
                st.session_state.vents.pop()
                st.rerun()
    with c2:
        if st.button("Clear vents", use_container_width=True):
            st.session_state.vents = []
            st.rerun()

    st.markdown("### Current vents")
    if st.session_state.vents:
        for i, v in enumerate(st.session_state.vents, start=1):
            st.caption(
                f"{i}. {v['type']} | x={v['x']:.2f} | y={v['y']:.2f} | flow={v['flow']} m³/h"
            )
    else:
        st.caption("No vents yet.")


# =========================================================
# Recompute every rerun
# =========================================================
people = auto_generate_people(
    room_w, room_d,
    standing_n, sitting_n, lying_n,
    standing_state, sitting_state, lying_state
)

C = simulate_grid(
    room_w, room_d, room_h,
    cell_size,
    initial_co2,
    outdoor_co2,
    people,
    st.session_state.vents,
    sim_hours,
    dt_minutes
)

avg_co2 = float(np.mean(C))
max_co2 = float(np.max(C))
risk_ratio = float(np.sum(C > 1000) / C.size * 100.0)


# =========================================================
# Right panel
# =========================================================
with right:
    st.subheader("Results")

    m1, m2, m3 = st.columns(3)
    m1.metric("Average CO2", f"{avg_co2:.0f} ppm")
    m2.metric("Max CO2", f"{max_co2:.0f} ppm")
    m3.metric("Area >1000 ppm", f"{risk_ratio:.1f}%")

    layout_img = render_layout_image(
        room_w, room_d, cell_size, people, st.session_state.vents
    )

    click = streamlit_image_coordinates(layout_img, key="layout_click")

    if click is not None:
        current_click = (click["x"], click["y"], click.get("time"))
        if st.session_state.last_click != current_click:
            st.session_state.last_click = current_click

            x_m, y_m = pixel_to_room_coords(
                click["x"], click["y"],
                layout_img.size[0], layout_img.size[1],
                room_w, room_d
            )

            x_snap, y_snap, ix, iy = snap_to_cell_center(
                x_m, y_m, cell_size, room_w, room_d
            )

            if edit_mode == "Add vent":
                if not vent_exists_at(x_snap, y_snap, st.session_state.vents):
                    st.session_state.vents.append({
                        "type": pending_vent_type,
                        "x": x_snap,
                        "y": y_snap,
                        "flow": int(pending_vent_flow)
                    })
            else:
                idx = nearest_vent_index(x_snap, y_snap, st.session_state.vents)
                if idx is not None:
                    st.session_state.vents.pop(idx)

            st.rerun()

    fig = plot_heatmap(C, room_w, room_d, cell_size, st.session_state.vents)
    st.pyplot(fig, use_container_width=True)

    st.caption("Add vent: layout 그림 클릭 → 가장 가까운 셀 중심에 설치")
    st.caption("Delete vent: 삭제 모드에서 클릭한 셀 기준 가장 가까운 환기구 삭제")
    st.caption("Standing=o, Sitting=square, Lying=block")
    st.caption("Supply=triangle up, Exhaust=triangle down")
