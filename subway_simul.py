import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="CO2 Grid Simulation")

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
    nx = max(1, int(round(room_w / cell_size)))
    ny = max(1, int(round(room_d / cell_size)))
    return nx, ny


def pos_to_idx(x, y, cell_size, nx, ny):
    ix = min(max(int(x / cell_size), 0), nx - 1)
    iy = min(max(int(y / cell_size), 0), ny - 1)
    return ix, iy


def idx_to_pos(ix, iy, cell_size):
    x = (ix + 0.5) * cell_size
    y = (iy + 0.5) * cell_size
    return x, y


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

        if v["type"] == "supply":
            C[ix, iy] = C[ix, iy] - local_exchange * (C[ix, iy] - outdoor_co2)

            for dx, dy, factor in [(-1, 0, 0.35), (1, 0, 0.35), (0, -1, 0.35), (0, 1, 0.35)]:
                ni, nj = ix + dx, iy + dy
                if 0 <= ni < nx and 0 <= nj < ny:
                    C[ni, nj] = C[ni, nj] - local_exchange * factor * (C[ni, nj] - outdoor_co2)

        elif v["type"] == "exhaust":
            C[ix, iy] = C[ix, iy] - local_exchange * (C[ix, iy] - outdoor_co2)

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

    for _ in range(steps):
        C = apply_people_source(C, people, cell_size, nx, ny, dt_h, room_h)
        C = apply_vents(C, vents, outdoor_co2, cell_size, nx, ny, dt_h, room_h)
        C = diffuse(C, diffusion_strength=0.20)

    return C, nx, ny


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


def build_layout_figure(room_w, room_d, cell_size, people, vents, C=None):
    nx, ny = create_grid(room_w, room_d, cell_size)

    fig = go.Figure()

    # Heatmap
    if C is not None:
        x_centers = [(i + 0.5) * cell_size for i in range(nx)]
        y_centers = [(j + 0.5) * cell_size for j in range(ny)]

        fig.add_trace(
            go.Heatmap(
                z=C.T,
                x=x_centers,
                y=y_centers,
                colorscale="RdYlBu_r",
                zmin=400,
                zmax=3000,
                colorbar=dict(title="CO2 (ppm)"),
                hovertemplate="x=%{x:.2f} m<br>y=%{y:.2f} m<br>CO2=%{z:.0f} ppm<extra></extra>"
            )
        )

    # Clickable grid points
    grid_x = []
    grid_y = []
    grid_custom = []
    for i in range(nx):
        for j in range(ny):
            x, y = idx_to_pos(i, j, cell_size)
            grid_x.append(x)
            grid_y.append(y)
            grid_custom.append(["grid", round(x, 3), round(y, 3)])

    fig.add_trace(
        go.Scatter(
            x=grid_x,
            y=grid_y,
            mode="markers",
            marker=dict(size=10, opacity=0.18, symbol="circle"),
            customdata=grid_custom,
            name="Grid click points",
            hovertemplate="Add vent here<br>x=%{customdata[1]:.2f} m<br>y=%{customdata[2]:.2f} m<extra></extra>"
        )
    )

    # People
    standing_x = [p["x"] for p in people if p["type"] == "standing"]
    standing_y = [p["y"] for p in people if p["type"] == "standing"]
    sitting_x = [p["x"] for p in people if p["type"] == "sitting"]
    sitting_y = [p["y"] for p in people if p["type"] == "sitting"]
    lying_x = [p["x"] for p in people if p["type"] == "lying"]
    lying_y = [p["y"] for p in people if p["type"] == "lying"]

    if standing_x:
        fig.add_trace(
            go.Scatter(
                x=standing_x, y=standing_y,
                mode="markers",
                marker=dict(symbol="circle", size=10),
                name="Standing",
                hoverinfo="skip"
            )
        )

    if sitting_x:
        fig.add_trace(
            go.Scatter(
                x=sitting_x, y=sitting_y,
                mode="markers",
                marker=dict(symbol="square", size=10),
                name="Sitting",
                hoverinfo="skip"
            )
        )

    if lying_x:
        fig.add_trace(
            go.Scatter(
                x=lying_x, y=lying_y,
                mode="markers",
                marker=dict(symbol="line-ew", size=18),
                name="Lying",
                hoverinfo="skip"
            )
        )

    # Vents: each point gets customdata with index
    supply_x = []
    supply_y = []
    supply_custom = []

    exhaust_x = []
    exhaust_y = []
    exhaust_custom = []

    supply_count = 0
    exhaust_count = 0

    for idx, v in enumerate(vents):
        if v["type"] == "supply":
            supply_count += 1
            supply_x.append(v["x"])
            supply_y.append(v["y"])
            supply_custom.append(["vent", idx])
        else:
            exhaust_count += 1
            exhaust_x.append(v["x"])
            exhaust_y.append(v["y"])
            exhaust_custom.append(["vent", idx])

    if supply_x:
        fig.add_trace(
            go.Scatter(
                x=supply_x, y=supply_y,
                mode="markers+text",
                text=[f"S{i+1}" for i in range(supply_count)],
                textposition="top center",
                marker=dict(symbol="triangle-up", size=15),
                customdata=supply_custom,
                name="Supply",
                hovertemplate="Supply vent<extra></extra>"
            )
        )

    if exhaust_x:
        fig.add_trace(
            go.Scatter(
                x=exhaust_x, y=exhaust_y,
                mode="markers+text",
                text=[f"E{i+1}" for i in range(exhaust_count)],
                textposition="top center",
                marker=dict(symbol="triangle-down", size=15),
                customdata=exhaust_custom,
                name="Exhaust",
                hovertemplate="Exhaust vent<extra></extra>"
            )
        )

    fig.update_layout(
        title="Layout editor / CO2 map",
        xaxis=dict(title="Width (m)", range=[0, room_w], constrain="domain"),
        yaxis=dict(title="Depth (m)", range=[0, room_d], scaleanchor="x", scaleratio=1),
        height=700,
        margin=dict(l=20, r=20, t=60, b=20),
        dragmode="select",
        clickmode="event+select",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

    return fig


def get_selected_payload(event):
    if not event:
        return None

    selection = event.get("selection")
    if not selection:
        return None

    points = selection.get("points", [])
    if not points:
        return None

    p = points[0]
    custom = p.get("customdata")
    if not custom:
        return None

    return custom


def handle_selection(payload, edit_mode, vent_type, vent_flow):
    if payload is None:
        return False

    payload_key = tuple(payload)
    if st.session_state.last_selection_key == payload_key:
        return False

    st.session_state.last_selection_key = payload_key

    kind = payload[0]

    # Add mode: click grid point
    if edit_mode == "Add vent" and kind == "grid":
        x = float(payload[1])
        y = float(payload[2])

        st.session_state.vents.append({
            "type": vent_type,
            "x": round(x, 2),
            "y": round(y, 2),
            "flow": int(vent_flow)
        })
        return True

    # Delete mode: click existing vent
    if edit_mode == "Delete vent" and kind == "vent":
        vent_idx = int(payload[1])
        if 0 <= vent_idx < len(st.session_state.vents):
            st.session_state.vents.pop(vent_idx)
            return True

    return False


# =========================================================
# Session state
# =========================================================
if "vents" not in st.session_state:
    st.session_state.vents = []

if "last_selection_key" not in st.session_state:
    st.session_state.last_selection_key = None


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
                st.session_state.last_selection_key = None
                st.rerun()
    with c2:
        if st.button("Clear vents", use_container_width=True):
            st.session_state.vents = []
            st.session_state.last_selection_key = None
            st.rerun()

    st.markdown("### Current vents")
    if st.session_state.vents:
        for i, v in enumerate(st.session_state.vents, start=1):
            st.caption(
                f"{i}. {v['type']} | x={v['x']:.2f} m | y={v['y']:.2f} m | flow={v['flow']} m³/h"
            )
    else:
        st.caption("No vents yet.")

people = auto_generate_people(
    room_w, room_d,
    standing_n, sitting_n, lying_n,
    standing_state, sitting_state, lying_state
)

C, nx, ny = simulate_grid(
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

with right:
    st.subheader("Results")

    m1, m2, m3 = st.columns(3)
    m1.metric("Average CO2", f"{avg_co2:.0f} ppm")
    m2.metric("Max CO2", f"{max_co2:.0f} ppm")
    m3.metric("Area >1000 ppm", f"{risk_ratio:.1f}%")

    fig = build_layout_figure(
        room_w=room_w,
        room_d=room_d,
        cell_size=cell_size,
        people=people,
        vents=st.session_state.vents,
        C=C
    )

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        key="layout_plot",
        on_select="rerun",
        config={"scrollZoom": False}
    )

    payload = get_selected_payload(event)
    changed = handle_selection(
        payload=payload,
        edit_mode=edit_mode,
        vent_type=pending_vent_type,
        vent_flow=pending_vent_flow
    )

    if changed:
        st.rerun()

    st.caption("Add vent 모드: 희미한 grid 점 클릭")
    st.caption("Delete vent 모드: 기존 삼각형 환기구 클릭")
    st.caption("Standing=o, Sitting=square, Lying=line")
    st.caption("Supply=triangle up, Exhaust=triangle down")
