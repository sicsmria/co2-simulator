import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="CO2 Grid Simulation")

# =========================================================
# Helpers
# =========================================================
def get_activity_co2_rate(activity_type: str) -> float:
    rates = {
        "Stable": 0.018,
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
    x = min((ix + 0.5) * cell_size, 1e9)
    y = min((iy + 0.5) * cell_size, 1e9)
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

        if v["type"] == "supply":
            factor = 0.35
        else:
            factor = 0.45

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
        people.append({"type": "standing", "x": x, "y": y, "activity": standing_state})

    sitting_cols = max(1, int(max(1.0, room_w - 1.0) // 1.1))
    for i in range(sitting_n):
        row = i // sitting_cols
        col = i % sitting_cols
        x = min(0.8 + col * 1.1, room_w - 0.6)
        y = min(room_d * 0.50 + row * 1.0, room_d - 1.0)
        people.append({"type": "sitting", "x": x, "y": y, "activity": sitting_state})

    lying_cols = max(1, int(max(1.0, room_w - 1.0) // 1.3))
    for i in range(lying_n):
        row = i // lying_cols
        col = i % lying_cols
        x = min(1.0 + col * 1.3, room_w - 0.8)
        y = min(room_d * 0.78 + row * 0.9, room_d - 0.8)
        people.append({"type": "lying", "x": x, "y": y, "activity": lying_state})

    return people


# =========================================================
# Plot builders
# =========================================================
def build_editor_figure(room_w, room_d, cell_size, people, vents):
    nx, ny = create_grid(room_w, room_d, cell_size)
    fig = go.Figure()

    # Clickable grid points
    grid_x, grid_y, grid_custom = [], [], []
    for i in range(nx):
        for j in range(ny):
            x, y = idx_to_pos(i, j, cell_size)
            if x <= room_w and y <= room_d:
                grid_x.append(x)
                grid_y.append(y)
                grid_custom.append(["grid", round(x, 3), round(y, 3)])

    fig.add_trace(
        go.Scatter(
            x=grid_x,
            y=grid_y,
            mode="markers",
            marker=dict(size=11, opacity=0.35, symbol="circle"),
            customdata=grid_custom,
            name="Grid",
            hovertemplate="x=%{customdata[1]:.2f} m<br>y=%{customdata[2]:.2f} m<extra></extra>"
        )
    )

    # People
    standing = [p for p in people if p["type"] == "standing"]
    sitting = [p for p in people if p["type"] == "sitting"]
    lying = [p for p in people if p["type"] == "lying"]

    if standing:
        fig.add_trace(go.Scatter(
            x=[p["x"] for p in standing],
            y=[p["y"] for p in standing],
            mode="markers",
            marker=dict(symbol="circle", size=10),
            name="Standing",
            hoverinfo="skip"
        ))
    if sitting:
        fig.add_trace(go.Scatter(
            x=[p["x"] for p in sitting],
            y=[p["y"] for p in sitting],
            mode="markers",
            marker=dict(symbol="square", size=10),
            name="Sitting",
            hoverinfo="skip"
        ))
    if lying:
        fig.add_trace(go.Scatter(
            x=[p["x"] for p in lying],
            y=[p["y"] for p in lying],
            mode="markers",
            marker=dict(symbol="line-ew", size=18),
            name="Lying",
            hoverinfo="skip"
        ))

    # Vents
    supply_pts = [(idx, v) for idx, v in enumerate(vents) if v["type"] == "supply"]
    exhaust_pts = [(idx, v) for idx, v in enumerate(vents) if v["type"] == "exhaust"]

    if supply_pts:
        fig.add_trace(go.Scatter(
            x=[v["x"] for _, v in supply_pts],
            y=[v["y"] for _, v in supply_pts],
            mode="markers+text",
            text=[f"S{i+1}" for i in range(len(supply_pts))],
            textposition="top center",
            marker=dict(symbol="triangle-up", size=16),
            customdata=[["vent", idx] for idx, _ in supply_pts],
            name="Supply",
            hovertemplate="Supply vent<extra></extra>"
        ))

    if exhaust_pts:
        fig.add_trace(go.Scatter(
            x=[v["x"] for _, v in exhaust_pts],
            y=[v["y"] for _, v in exhaust_pts],
            mode="markers+text",
            text=[f"E{i+1}" for i in range(len(exhaust_pts))],
            textposition="top center",
            marker=dict(symbol="triangle-down", size=16),
            customdata=[["vent", idx] for idx, _ in exhaust_pts],
            name="Exhaust",
            hovertemplate="Exhaust vent<extra></extra>"
        ))

    fig.update_layout(
        title="Layout editor",
        xaxis=dict(title="Width (m)", range=[0, room_w]),
        yaxis=dict(title="Depth (m)", range=[0, room_d], scaleanchor="x", scaleratio=1),
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        dragmode="select",
        clickmode="event+select",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return fig


def build_heatmap_figure(room_w, room_d, cell_size, people, vents, C):
    nx, ny = create_grid(room_w, room_d, cell_size)
    x_centers = [(i + 0.5) * cell_size for i in range(nx)]
    y_centers = [(j + 0.5) * cell_size for j in range(ny)]

    fig = go.Figure()

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

    if vents:
        fig.add_trace(go.Scatter(
            x=[v["x"] for v in vents if v["type"] == "supply"],
            y=[v["y"] for v in vents if v["type"] == "supply"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=13),
            name="Supply"
        ))
        fig.add_trace(go.Scatter(
            x=[v["x"] for v in vents if v["type"] == "exhaust"],
            y=[v["y"] for v in vents if v["type"] == "exhaust"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=13),
            name="Exhaust"
        ))

    fig.update_layout(
        title="CO2 heatmap",
        xaxis=dict(title="Width (m)", range=[0, room_w]),
        yaxis=dict(title="Depth (m)", range=[0, room_d], scaleanchor="x", scaleratio=1),
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# =========================================================
# Selection handling
# =========================================================
def extract_payload_from_event(event):
    # event object path
    try:
        points = event.selection.points
        if points:
            custom = points[0].get("customdata")
            if custom:
                return list(custom)
    except Exception:
        pass

    # dict-like path
    try:
        points = event["selection"]["points"]
        if points:
            custom = points[0].get("customdata")
            if custom:
                return list(custom)
    except Exception:
        pass

    return None


def extract_payload_from_session(key_name):
    try:
        state = st.session_state.get(key_name)
        if state and "selection" in state and state["selection"]["points"]:
            custom = state["selection"]["points"][0].get("customdata")
            if custom:
                return list(custom)
    except Exception:
        pass
    return None


def handle_selection(payload, edit_mode, vent_type, vent_flow):
    if payload is None:
        return False

    payload_key = tuple(payload)
    if st.session_state.last_selection_key == payload_key:
        return False

    st.session_state.last_selection_key = payload_key

    kind = payload[0]

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

    a, b = st.columns(2)
    with a:
        if st.button("Undo last vent", use_container_width=True):
            if st.session_state.vents:
                st.session_state.vents.pop()
                st.session_state.last_selection_key = None
                st.rerun()
    with b:
        if st.button("Clear vents", use_container_width=True):
            st.session_state.vents = []
            st.session_state.last_selection_key = None
            st.rerun()

    st.markdown("### Current vents")
    if st.session_state.vents:
        for i, v in enumerate(st.session_state.vents, start=1):
            st.caption(f"{i}. {v['type']} | x={v['x']:.2f} m | y={v['y']:.2f} m | flow={v['flow']} m³/h")
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

    editor_fig = build_editor_figure(room_w, room_d, cell_size, people, st.session_state.vents)

    event = st.plotly_chart(
        editor_fig,
        key="layout_editor",
        on_select="rerun",
        selection_mode="points",
        use_container_width=True,
        config={"scrollZoom": False}
    )

    st.write(event)
    st.write(st.session_state.get("layout_editor"))
    
    payload = extract_payload_from_event(event)
    if payload is None:
        payload = extract_payload_from_session("layout_editor")

    changed = handle_selection(
        payload=payload,
        edit_mode=edit_mode,
        vent_type=pending_vent_type,
        vent_flow=pending_vent_flow
    )

    if changed:
        st.rerun()

    heatmap_fig = build_heatmap_figure(
        room_w, room_d, cell_size, people, st.session_state.vents, C
    )
    st.plotly_chart(heatmap_fig, use_container_width=True, config={"scrollZoom": False})

    st.caption("Add vent: 위쪽 Layout editor의 grid 점 클릭")
    st.caption("Delete vent: 위쪽 Layout editor의 삼각형 환기구 클릭")

