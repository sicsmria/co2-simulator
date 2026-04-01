import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import math

st.set_page_config(page_title="Indoor CO2 Simulation", layout="wide")

# =========================
# Helper functions
# =========================
def get_activity_co2_rate(activity_type: str) -> float:
    """
    CO2 generation rate per person [m3/h]
    Rough assumptions:
    - Stable / seated: 0.018
    - Crowded / uneasy: 0.024
    - Moving: 0.032
    """
    rates = {
        "Stable": 0.018,
        "Crowded / Uneasy": 0.024,
        "Moving": 0.032
    }
    return rates.get(activity_type, 0.018)


def simulate_co2(
    room_w, room_d, room_h,
    outdoor_co2,
    initial_co2,
    ach,
    standing_n, standing_state,
    sitting_n, sitting_state,
    lying_n, lying_state,
    sim_hours,
    dt_minutes
):
    volume = room_w * room_d * room_h  # m3
    Q = ach * volume  # ventilation flow [m3/h]

    gen_standing = standing_n * get_activity_co2_rate(standing_state)
    gen_sitting = sitting_n * get_activity_co2_rate(sitting_state)
    gen_lying = lying_n * get_activity_co2_rate(lying_state)

    G_total = gen_standing + gen_sitting + gen_lying  # m3/h

    dt = dt_minutes / 60.0  # h
    steps = int(sim_hours / dt) + 1
    times = np.linspace(0, sim_hours, steps)

    co2 = np.zeros(steps)
    co2[0] = initial_co2

    # Mass balance:
    # dC/dt = (G/V)*1e6 + ACH*(C_out - C)
    for i in range(1, steps):
        dCdt = (G_total / volume) * 1e6 + ach * (outdoor_co2 - co2[i - 1])
        co2[i] = co2[i - 1] + dCdt * dt

    return times, co2, volume, Q, G_total


def draw_room_layout(room_w, room_d, standing_n, sitting_n, lying_n):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Room boundary
    ax.add_patch(Rectangle((0, 0), room_w, room_d, fill=False, linewidth=2))

    def place_people(count, y_center, color, label, radius=0.25, max_per_row=10):
        if count <= 0:
            return

        x_margin = 0.7
        y_gap = 0.8
        usable_w = max(room_w - 2 * x_margin, 0.5)
        cols = min(max_per_row, max(1, int(usable_w // 0.8)))
        rows = math.ceil(count / cols)

        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= count:
                    return
                x = x_margin + c * 0.8
                y = y_center + r * y_gap
                if x < room_w - 0.3 and y < room_d - 0.3:
                    ax.add_patch(Circle((x, y), radius, alpha=0.7))
                    idx += 1

    # Vertical zones
    zone1 = room_d * 0.15
    zone2 = room_d * 0.45
    zone3 = room_d * 0.75

    # Standing people
    for i in range(standing_n):
        row = i // max(1, int((room_w - 1.4) // 0.8))
        col = i % max(1, int((room_w - 1.4) // 0.8))
        x = 0.7 + col * 0.8
        y = zone1 + row * 0.8
        if x < room_w - 0.3 and y < zone1 + 2.0:
            ax.add_patch(Circle((x, y), 0.18, alpha=0.8))
            ax.plot([x, x], [y - 0.5, y - 0.1], linewidth=2)

    # Sitting people
    max_cols = max(1, int((room_w - 1.4) // 1.0))
    for i in range(sitting_n):
        row = i // max_cols
        col = i % max_cols
        x = 0.7 + col * 1.0
        y = zone2 + row * 0.9
        if x < room_w - 0.3 and y < zone2 + 2.0:
            ax.add_patch(Circle((x, y + 0.2), 0.16, alpha=0.8))
            ax.plot([x - 0.2, x + 0.2], [y, y], linewidth=2)
            ax.plot([x - 0.2, x - 0.2], [y, y - 0.25], linewidth=2)
            ax.plot([x + 0.2, x + 0.2], [y, y - 0.25], linewidth=2)

    # Lying people
    for i in range(lying_n):
        row = i // max_cols
        col = i % max_cols
        x = 0.8 + col * 1.2
        y = zone3 + row * 0.8
        if x < room_w - 0.8 and y < room_d - 0.4:
            ax.add_patch(Rectangle((x - 0.35, y - 0.12), 0.7, 0.24, alpha=0.7))
            ax.add_patch(Circle((x - 0.45, y), 0.10, alpha=0.8))

    ax.set_xlim(-0.5, room_w + 0.5)
    ax.set_ylim(-0.5, room_d + 0.5)
    ax.set_aspect("equal")
    ax.set_title("Room Layout with Occupants")
    ax.set_xlabel("Width (m)")
    ax.set_ylabel("Depth (m)")
    ax.grid(True, alpha=0.3)
    return fig


def plot_co2_curve(times, co2):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times, co2, linewidth=2)

    # Risk zones
    ax.axhspan(1000, 2000, alpha=0.15)
    ax.axhspan(2000, 5000, alpha=0.22)
    ax.axhline(1000, linestyle="--", linewidth=1.2)
    ax.axhline(2000, linestyle="--", linewidth=1.2)

    ax.set_title("CO2 Concentration Over Time")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("CO2 (ppm)")
    ax.grid(True, alpha=0.3)
    return fig


def summarize_result(co2):
    peak = float(np.max(co2))
    avg = float(np.mean(co2))
    over_1000 = np.sum(co2 > 1000)
    over_2000 = np.sum(co2 > 2000)

    if peak < 1000:
        status = "Good"
    elif peak < 2000:
        status = "Moderate Risk"
    else:
        status = "High Risk"

    return peak, avg, over_1000, over_2000, status


# =========================
# Left-right layout
# =========================
st.title("Indoor CO2 Simulation Dashboard")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Settings")

    st.markdown("### Room Geometry")
    room_w = st.slider("Room Width (m)", 3.0, 50.0, 12.0, 0.5)
    room_d = st.slider("Room Depth (m)", 3.0, 50.0, 10.0, 0.5)
    room_h = st.slider("Room Height (m)", 2.0, 10.0, 3.0, 0.1)

    st.markdown("### Air / Ventilation")
    outdoor_co2 = st.number_input("Outdoor CO2 (ppm)", 300, 600, 420)
    initial_co2 = st.number_input("Initial Indoor CO2 (ppm)", 300, 3000, 450)
    ach = st.slider("Ventilation Rate ACH (1/h)", 0.0, 20.0, 2.0, 0.1)

    st.markdown("### Standing Occupants")
    standing_n = st.number_input("Standing Count", 0, 300, 10)
    standing_state = st.selectbox("Standing Activity", ["Stable", "Crowded / Uneasy", "Moving"])

    st.markdown("### Sitting Occupants")
    sitting_n = st.number_input("Sitting Count", 0, 300, 10)
    sitting_state = st.selectbox("Sitting Activity", ["Stable", "Crowded / Uneasy", "Moving"])

    st.markdown("### Lying Occupants")
    lying_n = st.number_input("Lying Count", 0, 300, 0)
    lying_state = st.selectbox("Lying Activity", ["Stable", "Crowded / Uneasy", "Moving"])

    st.markdown("### Simulation")
    sim_hours = st.slider("Simulation Duration (hours)", 0.5, 24.0, 4.0, 0.5)
    dt_minutes = st.slider("Time Step (minutes)", 1, 30, 5)

    run_button = st.button("Run Simulation", use_container_width=True)

with col_right:
    st.subheader("Results")

    if run_button:
        times, co2, volume, Q, G_total = simulate_co2(
            room_w, room_d, room_h,
            outdoor_co2,
            initial_co2,
            ach,
            standing_n, standing_state,
            sitting_n, sitting_state,
            lying_n, lying_state,
            sim_hours,
            dt_minutes
        )

        peak, avg, over_1000, over_2000, status = summarize_result(co2)

        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("Room Volume", f"{volume:.1f} m³")
        metric2.metric("Ventilation Flow", f"{Q:.1f} m³/h")
        metric3.metric("CO2 Generation", f"{G_total:.3f} m³/h")
        metric4.metric("Risk Status", status)

        st.markdown("### CO2 Curve")
        fig1 = plot_co2_curve(times, co2)
        st.pyplot(fig1, use_container_width=True)

        st.markdown("### Occupant Layout")
        fig2 = draw_room_layout(room_w, room_d, standing_n, sitting_n, lying_n)
        st.pyplot(fig2, use_container_width=True)

        st.markdown("### Interpretation")
        st.write(f"- Peak CO2: **{peak:.0f} ppm**")
        st.write(f"- Average CO2: **{avg:.0f} ppm**")
        st.write(f"- Time steps above 1000 ppm: **{over_1000}**")
        st.write(f"- Time steps above 2000 ppm: **{over_2000}**")

        if peak < 1000:
            st.success("Air quality remains in a generally acceptable range.")
        elif peak < 2000:
            st.warning("Ventilation is somewhat insufficient. Improvement is recommended.")
        else:
            st.error("High CO2 buildup is expected. Stronger ventilation or lower occupancy is needed.")
    else:
        st.info("Adjust the settings on the left, then click **Run Simulation**.")
