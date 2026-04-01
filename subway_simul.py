import math
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Subway Shelter CO2 Simulation", layout="wide")

# -----------------------------
# ISO 7250-based simplified occupant geometry
# -----------------------------
POSTURE_SPECS = {
    "standing": {
        "label": "Standing",
        "width_m": 0.50,
        "depth_m": 0.30,
        "height_m": 1.70,
    },
    "seated": {
        "label": "Seated",
        "width_m": 0.50,
        "depth_m": 0.80,
        "height_m": 0.90,
    },
    "lying": {
        "label": "Lying",
        "width_m": 0.60,
        "depth_m": 1.80,
        "height_m": 0.25,
    },
}

ACTIVITY_LEVELS = {
    "stable": {
        "label": "Stable",
        "co2_gen_lps_per_person": 0.0040,
    },
    "crowded": {
        "label": "Crowded / Unstable",
        "co2_gen_lps_per_person": 0.0055,
    },
    "moving": {
        "label": "Moving",
        "co2_gen_lps_per_person": 0.0070,
    },
}


# -----------------------------
# Helper functions
# -----------------------------
def get_posture_area(posture: str) -> float:
    spec = POSTURE_SPECS[posture]
    return spec["width_m"] * spec["depth_m"]


def get_posture_volume(posture: str) -> float:
    spec = POSTURE_SPECS[posture]
    return spec["width_m"] * spec["depth_m"] * spec["height_m"]


def estimate_capacity_by_posture(length_m: float, width_m: float, posture: str, packing_factor: float = 1.3) -> int:
    floor_area_m2 = length_m * width_m
    area_per_person = get_posture_area(posture) * packing_factor
    if area_per_person <= 0:
        return 0
    return int(floor_area_m2 // area_per_person)


def simulate_co2(
    volume_m3: float,
    ventilation_m3_per_h: float,
    occupants: int,
    activity: str,
    outdoor_co2_ppm: float = 420.0,
    initial_co2_ppm: float = 420.0,
    duration_min: int = 180,
    dt_sec: int = 60,
):
    co2_ppm = initial_co2_ppm
    results = []

    Q = ventilation_m3_per_h / 3600.0
    co2_gen_per_person_m3s = ACTIVITY_LEVELS[activity]["co2_gen_lps_per_person"] / 1000.0
    G = occupants * co2_gen_per_person_m3s

    steps = int(duration_min * 60 / dt_sec)

    for step in range(steps + 1):
        time_min = step * dt_sec / 60.0
        results.append({"time_min": time_min, "co2_ppm": co2_ppm})

        c_in = co2_ppm / 1_000_000.0
        c_out = outdoor_co2_ppm / 1_000_000.0

        dc_dt = (G / volume_m3) - (Q / volume_m3) * (c_in - c_out)
        c_next = c_in + dc_dt * dt_sec
        co2_ppm = max(c_next * 1_000_000.0, outdoor_co2_ppm)

    return results


def run_shelter_simulation(
    length_m: float,
    width_m: float,
    height_m: float,
    posture: str,
    activity: str,
    occupants: int,
    ventilation_m3_per_h: float,
    outdoor_co2_ppm: float = 420.0,
    initial_co2_ppm: float = 420.0,
    duration_min: int = 180,
    packing_factor: float = 1.3,
):
    floor_area_m2 = length_m * width_m
    volume_m3 = floor_area_m2 * height_m

    capacity = estimate_capacity_by_posture(
        length_m=length_m,
        width_m=width_m,
        posture=posture,
        packing_factor=packing_factor,
    )

    overcrowded = occupants > capacity

    sim_results = simulate_co2(
        volume_m3=volume_m3,
        ventilation_m3_per_h=ventilation_m3_per_h,
        occupants=occupants,
        activity=activity,
        outdoor_co2_ppm=outdoor_co2_ppm,
        initial_co2_ppm=initial_co2_ppm,
        duration_min=duration_min,
    )

    return {
        "floor_area_m2": floor_area_m2,
        "volume_m3": volume_m3,
        "capacity": capacity,
        "overcrowded": overcrowded,
        "posture_area_m2_per_person": get_posture_area(posture),
        "posture_volume_m3_per_person": get_posture_volume(posture),
        "results": sim_results,
    }


def get_threshold_crossing_time(df: pd.DataFrame, threshold: float):
    crossed = df[df["co2_ppm"] >= threshold]
    if crossed.empty:
        return None
    return float(crossed.iloc[0]["time_min"])


def compute_layout(length_m: float, width_m: float, posture: str, packing_factor: float):
    """
    Returns layout info for drawing occupants in top view.
    """
    spec = POSTURE_SPECS[posture]
    occ_w = spec["width_m"] * math.sqrt(packing_factor)
    occ_d = spec["depth_m"] * math.sqrt(packing_factor)

    n_cols = max(1, int(length_m // occ_d))
    n_rows = max(1, int(width_m // occ_w))
    capacity = n_cols * n_rows

    return {
        "occ_w": occ_w,
        "occ_d": occ_d,
        "n_cols": n_cols,
        "n_rows": n_rows,
        "capacity": capacity,
    }


def draw_top_view(length_m: float, width_m: float, posture: str, occupants: int, packing_factor: float):
    spec = POSTURE_SPECS[posture]
    layout = compute_layout(length_m, width_m, posture, packing_factor)

    occ_w = layout["occ_w"]
    occ_d = layout["occ_d"]
    n_cols = layout["n_cols"]
    n_rows = layout["n_rows"]
    capacity = layout["capacity"]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Room boundary
    ax.add_patch(Rectangle((0, 0), length_m, width_m, fill=False, linewidth=2))

    # Draw occupants
    max_draw = min(occupants, capacity)
    count = 0

    x_margin = max(0.0, (length_m - n_cols * occ_d) / 2)
    y_margin = max(0.0, (width_m - n_rows * occ_w) / 2)

    for row in range(n_rows):
        for col in range(n_cols):
            if count >= max_draw:
                break
            x = x_margin + col * occ_d
            y = y_margin + row * occ_w

            ax.add_patch(
                Rectangle(
                    (x, y),
                    spec["depth_m"],
                    spec["width_m"],
                    alpha=0.6,
                )
            )
            count += 1
        if count >= max_draw:
            break

    ax.set_xlim(0, length_m)
    ax.set_ylim(0, width_m)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Top View Layout")
    ax.set_xlabel("Length (m)")
    ax.set_ylabel("Width (m)")
    ax.grid(True, alpha=0.3)

    if occupants > capacity:
        ax.text(
            length_m * 0.5,
            width_m * 0.5,
            f"Over capacity\nShown: {capacity}/{occupants}",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.8),
        )

    return fig


def draw_side_view(length_m: float, height_m: float, posture: str, occupants: int, packing_factor: float):
    spec = POSTURE_SPECS[posture]
    layout = compute_layout(length_m, 1.0, posture, packing_factor)  # only use along length
    occ_d = layout["occ_d"]
    n_cols = max(1, int(length_m // occ_d))

    fig, ax = plt.subplots(figsize=(8, 4))

    # Room boundary
    ax.add_patch(Rectangle((0, 0), length_m, height_m, fill=False, linewidth=2))

    # Draw simplified occupants along length
    shown_people = min(occupants, n_cols, 20)
    x_margin = max(0.0, (length_m - shown_people * occ_d) / 2)

    for i in range(shown_people):
        x = x_margin + i * occ_d
        ax.add_patch(
            Rectangle(
                (x, 0),
                spec["depth_m"],
                spec["height_m"],
                alpha=0.6,
            )
        )

    ax.set_xlim(0, length_m)
    ax.set_ylim(0, max(height_m * 1.05, spec["height_m"] * 1.2))
    ax.set_aspect("auto")
    ax.set_title("Side View Section")
    ax.set_xlabel("Length (m)")
    ax.set_ylabel("Height (m)")
    ax.grid(True, alpha=0.3)

    return fig


# -----------------------------
# UI
# -----------------------------
st.title("Subway Shelter CO₂ Simulation")
st.caption("Automatic update + geometric room input + occupant layout view")

left, right = st.columns([1, 1])

with left:
    st.subheader("1. Space geometry")
    length_m = st.number_input("Length (m)", min_value=1.0, value=20.0, step=1.0)
    width_m = st.number_input("Width (m)", min_value=1.0, value=10.0, step=1.0)
    height_m = st.number_input("Height (m)", min_value=1.0, value=3.5, step=0.1)

    st.subheader("2. Occupant settings")
    posture = st.selectbox(
        "Occupant posture",
        options=["standing", "seated", "lying"],
        format_func=lambda x: POSTURE_SPECS[x]["label"],
    )
    activity = st.selectbox(
        "Occupant activity state",
        options=["stable", "crowded", "moving"],
        format_func=lambda x: ACTIVITY_LEVELS[x]["label"],
    )
    occupants = st.number_input("Number of occupants", min_value=1, value=100, step=1)
    packing_factor = st.slider("Packing factor", min_value=1.0, max_value=2.0, value=1.3, step=0.05)

with right:
    st.subheader("3. Ventilation / CO₂ settings")
    ventilation_m3_per_h = st.number_input("Ventilation rate (m³/h)", min_value=0.0, value=5000.0, step=100.0)
    outdoor_co2_ppm = st.number_input("Outdoor CO₂ (ppm)", min_value=300.0, value=420.0, step=10.0)
    initial_co2_ppm = st.number_input("Initial indoor CO₂ (ppm)", min_value=300.0, value=420.0, step=10.0)
    duration_min = st.slider("Simulation duration (min)", min_value=10, max_value=720, value=180, step=10)

# -----------------------------
# Run automatically
# -----------------------------
output = run_shelter_simulation(
    length_m=length_m,
    width_m=width_m,
    height_m=height_m,
    posture=posture,
    activity=activity,
    occupants=occupants,
    ventilation_m3_per_h=ventilation_m3_per_h,
    outdoor_co2_ppm=outdoor_co2_ppm,
    initial_co2_ppm=initial_co2_ppm,
    duration_min=duration_min,
    packing_factor=packing_factor,
)

df = pd.DataFrame(output["results"])

t_1000 = get_threshold_crossing_time(df, 1000)
t_2000 = get_threshold_crossing_time(df, 2000)
t_3000 = get_threshold_crossing_time(df, 3000)

# -----------------------------
# Summary
# -----------------------------
st.subheader("Simulation Summary")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Floor area", f"{output['floor_area_m2']:.1f} m²")
with c2:
    st.metric("Volume", f"{output['volume_m3']:.1f} m³")
with c3:
    st.metric("Area / person", f"{output['posture_area_m2_per_person']:.2f} m²")
with c4:
    st.metric("Estimated capacity", f"{output['capacity']} persons")
with c5:
    st.metric("Density", f"{occupants / output['floor_area_m2']:.2f} persons/m²")

if output["overcrowded"]:
    st.error("Occupancy exceeds estimated posture-based capacity.")
else:
    st.success("Occupancy is within estimated posture-based capacity.")

# -----------------------------
# Layout visualization
# -----------------------------
st.subheader("Space and Occupant Visualization")

viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    top_fig = draw_top_view(
        length_m=length_m,
        width_m=width_m,
        posture=posture,
        occupants=occupants,
        packing_factor=packing_factor,
    )
    st.pyplot(top_fig)

with viz_col2:
    side_fig = draw_side_view(
        length_m=length_m,
        height_m=height_m,
        posture=posture,
        occupants=occupants,
        packing_factor=packing_factor,
    )
    st.pyplot(side_fig)

# -----------------------------
# Threshold info
# -----------------------------
st.subheader("Threshold Crossing Time")

tcol1, tcol2, tcol3 = st.columns(3)
with tcol1:
    st.metric("1000 ppm", "Not reached" if t_1000 is None else f"{t_1000:.1f} min")
with tcol2:
    st.metric("2000 ppm", "Not reached" if t_2000 is None else f"{t_2000:.1f} min")
with tcol3:
    st.metric("3000 ppm", "Not reached" if t_3000 is None else f"{t_3000:.1f} min")

# -----------------------------
# CO2 plot
# -----------------------------
st.subheader("CO₂ Concentration Over Time")

fig, ax = plt.subplots(figsize=(12, 6))
x = df["time_min"]
y = df["co2_ppm"]

ax.axhspan(0, 1000, alpha=0.10)
ax.axhspan(1000, 2000, alpha=0.15)
ax.axhspan(2000, 3000, alpha=0.20)
ax.axhspan(3000, max(4000, y.max() * 1.05), alpha=0.25)

ax.plot(x, y, linewidth=2)
ax.axhline(1000, linestyle="--", linewidth=1)
ax.axhline(2000, linestyle="--", linewidth=1)
ax.axhline(3000, linestyle="--", linewidth=1)

ax.text(x.max() * 0.99, 1000, " 1000 ppm", va="bottom", ha="right")
ax.text(x.max() * 0.99, 2000, " 2000 ppm", va="bottom", ha="right")
ax.text(x.max() * 0.99, 3000, " 3000 ppm", va="bottom", ha="right")

for threshold, crossing_time in [(1000, t_1000), (2000, t_2000), (3000, t_3000)]:
    if crossing_time is not None:
        crossing_row = df[df["time_min"] >= crossing_time].iloc[0]
        ax.scatter(crossing_row["time_min"], crossing_row["co2_ppm"], s=40)
        ax.text(
            crossing_row["time_min"],
            crossing_row["co2_ppm"],
            f"  {threshold} ppm @ {crossing_time:.1f} min",
            va="bottom",
            ha="left",
            fontsize=9,
        )

ax.set_xlabel("Time (min)")
ax.set_ylabel("Indoor CO₂ (ppm)")
ax.set_title("CO₂ Concentration with Risk Zones")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, duration_min)
ax.set_ylim(0, max(4000, y.max() * 1.10))

st.pyplot(fig)

# -----------------------------
# Data table
# -----------------------------
with st.expander("Show simulation data table"):
    st.dataframe(df, use_container_width=True)
