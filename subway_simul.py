import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="CO2 Grid Simulation")

# -------------------------
# Helpers
# -------------------------
def get_activity_co2_rate(activity_type: str) -> float:
    rates = {
        "Stable": 0.018,             # m3/h/person
        "Crowded / Uneasy": 0.024,
        "Moving": 0.032
    }
    return rates.get(activity_type, 0.018)


def create_grid(room_w, room_d, cell_size):
    nx = int(room_w / cell_size)
    ny = int(room_d / cell_size)
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
    for p in people:
        ix, iy = pos_to_idx(p["x"], p["y"], cell_size, nx, ny)
        cell_volume = cell_size * cell_size * room_h
        G = get_activity_co2_rate(p["activity"])  # m3/h
        delta_ppm = (G / cell_volume) * 1e6 * dt_h
        C[ix, iy] += delta_ppm
    return C


def apply_vents(C, vents, outdoor_co2, cell_size, nx, ny, dt_h, room_h):
    for v in vents:
        ix, iy = pos_to_idx(v["x"], v["y"], cell_size, nx, ny)
        cell_volume = cell_size * cell_size * room_h
        air_change_local = (v["flow"] / cell_volume) * dt_h

        # 너무 커지지 않게 제한
        air_change_local = min(air_change_local, 1.0)

        if v["type"] == "supply":
            # outdoor CO2 쪽으로 당김
            C[ix, iy] = C[ix, iy] - air_change_local * (C[ix, iy] - outdoor_co2)

        elif v["type"] == "exhaust":
            # 주변 공기를 빼내는 느낌
            C[ix, iy] = C[ix, iy] - air_change_local * (C[ix, iy] - outdoor_co2)

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
    steps = int(sim_hours / dt_h)

    history = []

    for _ in range(steps):
        C = apply_people_source(C, people, cell_size, nx, ny, dt_h, room_h)
        C = apply_vents(C, vents, outdoor_co2, cell_size, nx, ny, dt_h, room_h)
        C = diffuse(C, diffusion_strength=0.20)
        history.append(C.copy())

    return C, history, nx, ny


def plot_heatmap(C, room_w, room_d, people, vents, vmin=400, vmax=3000):
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        C.T,
        origin="lower",
        extent=[0, room_w, 0, room_d],
        aspect="auto",
        vmin=vmin,
        vmax=vmax
    )

    for p in people:
        marker = "o"
        if p["type"] == "standing":
            marker = "o"
        elif p["type"] == "sitting":
            marker = "s"
        elif p["type"] == "lying":
            marker = "_"

        ax.scatter(p["x"], p["y"], marker=marker, s=100, label=p["type"])

    for v in vents:
        if v["type"] == "supply":
            ax.scatter(v["x"], v["y"], marker="^", s=180)
            ax.text(v["x"], v["y"] + 0.2, "S", ha="center")
        else:
            ax.scatter(v["x"], v["y"], marker="v", s=180)
            ax.text(v["x"], v["y"] + 0.2, "E", ha="center")

    ax.set_title("CO2 Distribution Heatmap")
    ax.set_xlabel("Width (m)")
    ax.set_ylabel("Depth (m)")
    plt.colorbar(im, ax=ax, label="CO2 (ppm)")
    return fig


# -------------------------
# Layout
# -------------------------
st.title("Indoor CO2 Grid Simulation")

left, right = st.columns([1, 2])

with left:
    st.subheader("Settings")

    room_w = st.slider("Room width (m)", 4.0, 30.0, 12.0, 0.5)
    room_d = st.slider("Room depth (m)", 4.0, 30.0, 10.0, 0.5)
    room_h = st.slider("Room height (m)", 2.0, 6.0, 3.0, 0.1)
    cell_size = st.slider("Grid cell size (m)", 0.5, 2.0, 1.0, 0.5)

    initial_co2 = st.number_input("Initial indoor CO2 (ppm)", 300, 3000, 450)
    outdoor_co2 = st.number_input("Outdoor CO2 (ppm)", 300, 600, 420)

    sim_hours = st.slider("Simulation time (h)", 0.5, 8.0, 2.0, 0.5)
    dt_minutes = st.slider("Time step (min)", 1, 20, 5)

    st.markdown("### Vent 1")
    vent1_type = st.selectbox("Vent 1 type", ["supply", "exhaust"])
    vent1_x = st.slider("Vent 1 x", 0.0, room_w, min(2.0, room_w), 0.5)
    vent1_y = st.slider("Vent 1 y", 0.0, room_d, min(2.0, room_d), 0.5)
    vent1_flow = st.slider("Vent 1 flow (m3/h)", 50, 2000, 400, 50)

    st.markdown("### Vent 2")
    vent2_type = st.selectbox("Vent 2 type", ["supply", "exhaust"], index=1)
    vent2_x = st.slider("Vent 2 x ", 0.0, room_w, min(8.0, room_w), 0.5)
    vent2_y = st.slider("Vent 2 y ", 0.0, room_d, min(6.0, room_d), 0.5)
    vent2_flow = st.slider("Vent 2 flow (m3/h)", 50, 2000, 400, 50)

    st.markdown("### Example people")
    standing_n = st.slider("Standing people", 0, 10, 2)
    sitting_n = st.slider("Sitting people", 0, 10, 2)
    lying_n = st.slider("Lying people", 0, 10, 1)

    run = st.button("Run simulation", use_container_width=True)

with right:
    if run:
        people = []

        for i in range(standing_n):
            people.append({
                "type": "standing",
                "x": 1.5 + i,
                "y": 2.0,
                "activity": "Moving"
            })

        for i in range(sitting_n):
            people.append({
                "type": "sitting",
                "x": 1.5 + i,
                "y": 5.0,
                "activity": "Stable"
            })

        for i in range(lying_n):
            people.append({
                "type": "lying",
                "x": 1.5 + i,
                "y": 8.0,
                "activity": "Stable"
            })

        vents = [
            {"type": vent1_type, "x": vent1_x, "y": vent1_y, "flow": vent1_flow},
            {"type": vent2_type, "x": vent2_x, "y": vent2_y, "flow": vent2_flow},
        ]

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

        avg_co2 = np.mean(C)
        max_co2 = np.max(C)
        risk_ratio = np.sum(C > 1000) / C.size * 100

        m1, m2, m3 = st.columns(3)
        m1.metric("Average CO2", f"{avg_co2:.0f} ppm")
        m2.metric("Max CO2", f"{max_co2:.0f} ppm")
        m3.metric("Area >1000 ppm", f"{risk_ratio:.1f}%")

        fig = plot_heatmap(C, room_w, room_d, people, vents)
        st.pyplot(fig, use_container_width=True)

    else:
        st.info("Set vents on the left and run the simulation.")
