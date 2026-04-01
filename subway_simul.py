import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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

# -----------------------------
# Activity-based CO2 generation
# -----------------------------
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


def estimate_capacity_by_posture(
    floor_area_m2: float,
    posture: str,
    packing_factor: float = 1.3,
) -> int:
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
    """
    Single-zone CO2 mass balance model:
    dC/dt = G/V - Q/V * (C - Cout)
    """
    co2_ppm = initial_co2_ppm
    results = []

    # m3/h -> m3/s
    Q = ventilation_m3_per_h / 3600.0

    # L/s/person -> m3/s/person
    co2_gen_per_person_m3s = ACTIVITY_LEVELS[activity]["co2_gen_lps_per_person"] / 1000.0

    # total CO2 generation
    G = occupants * co2_gen_per_person_m3s

    steps = int(duration_min * 60 / dt_sec)

    for step in range(steps + 1):
        time_min = step * dt_sec / 60.0
        results.append(
            {
                "time_min": time_min,
                "co2_ppm": co2_ppm,
            }
        )

        c_in = co2_ppm / 1_000_000.0
        c_out = outdoor_co2_ppm / 1_000_000.0

        dc_dt = (G / volume_m3) - (Q / volume_m3) * (c_in - c_out)
        c_next = c_in + dc_dt * dt_sec

        co2_ppm = max(c_next * 1_000_000.0, outdoor_co2_ppm)

    return results


def run_shelter_simulation(
    floor_area_m2: float,
    ceiling_height_m: float,
    posture: str,
    activity: str,
    occupants: int,
    ventilation_m3_per_h: float,
    outdoor_co2_ppm: float = 420.0,
    initial_co2_ppm: float = 420.0,
    duration_min: int = 180,
    packing_factor: float = 1.3,
):
    volume_m3 = floor_area_m2 * ceiling_height_m

    capacity = estimate_capacity_by_posture(
        floor_area_m2=floor_area_m2,
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


# -----------------------------
# UI
# -----------------------------
st.title("Subway Shelter CO₂ Simulation")
st.caption("Automatic update version with posture-based capacity and risk-zone visualization")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Space settings")
    floor_area_m2 = st.number_input(
        "Usable floor area (m²)",
        min_value=10.0,
        value=500.0,
        step=10.0,
    )
    ceiling_height_m = st.number_input(
        "Ceiling height (m)",
        min_value=2.0,
        value=3.5,
        step=0.1,
    )
    ventilation_m3_per_h = st.number_input(
        "Ventilation rate (m³/h)",
        min_value=0.0,
        value=10000.0,
        step=100.0,
    )

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

    occupants = st.number_input(
        "Number of occupants",
        min_value=1,
        value=300,
        step=1,
    )

with col2:
    st.subheader("3. CO₂ settings")
    outdoor_co2_ppm = st.number_input(
        "Outdoor CO₂ (ppm)",
        min_value=300.0,
        value=420.0,
        step=10.0,
    )
    initial_co2_ppm = st.number_input(
        "Initial indoor CO₂ (ppm)",
        min_value=300.0,
        value=420.0,
        step=10.0,
    )

    duration_min = st.slider(
        "Simulation duration (min)",
        min_value=10,
        max_value=720,
        value=180,
        step=10,
    )

    packing_factor = st.slider(
        "Packing factor",
        min_value=1.0,
        max_value=2.0,
        value=1.3,
        step=0.05,
    )

# -----------------------------
# Run automatically
# -----------------------------
output = run_shelter_simulation(
    floor_area_m2=floor_area_m2,
    ceiling_height_m=ceiling_height_m,
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

# Threshold crossing times
t_1000 = get_threshold_crossing_time(df, 1000)
t_2000 = get_threshold_crossing_time(df, 2000)
t_3000 = get_threshold_crossing_time(df, 3000)

# -----------------------------
# Summary
# -----------------------------
st.subheader("Simulation Summary")

sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)

with sum_col1:
    st.metric("Space volume", f"{output['volume_m3']:.1f} m³")
with sum_col2:
    st.metric("Area per person", f"{output['posture_area_m2_per_person']:.2f} m²/person")
with sum_col3:
    st.metric("Estimated capacity", f"{output['capacity']} persons")
with sum_col4:
    density = occupants / floor_area_m2
    st.metric("Occupant density", f"{density:.2f} persons/m²")

if output["overcrowded"]:
    st.error("Occupancy exceeds estimated posture-based capacity.")
else:
    st.success("Occupancy is within estimated posture-based capacity.")

# -----------------------------
# Threshold info
# -----------------------------
st.subheader("Threshold Crossing Time")

thr_col1, thr_col2, thr_col3 = st.columns(3)

with thr_col1:
    st.metric("1000 ppm", "Not reached" if t_1000 is None else f"{t_1000:.1f} min")
with thr_col2:
    st.metric("2000 ppm", "Not reached" if t_2000 is None else f"{t_2000:.1f} min")
with thr_col3:
    st.metric("3000 ppm", "Not reached" if t_3000 is None else f"{t_3000:.1f} min")

# -----------------------------
# Plot
# -----------------------------
st.subheader("CO₂ Concentration Over Time")

fig, ax = plt.subplots(figsize=(12, 6))

x = df["time_min"]
y = df["co2_ppm"]

# Risk zones
ax.axhspan(0, 1000, alpha=0.10, label="Acceptable zone")
ax.axhspan(1000, 2000, alpha=0.15, label="Elevated zone")
ax.axhspan(2000, 3000, alpha=0.20, label="High-risk zone")
ax.axhspan(3000, max(4000, y.max() * 1.05), alpha=0.25, label="Severe-risk zone")

# CO2 curve
ax.plot(x, y, linewidth=2)

# Threshold lines
ax.axhline(1000, linestyle="--", linewidth=1)
ax.axhline(2000, linestyle="--", linewidth=1)
ax.axhline(3000, linestyle="--", linewidth=1)

# Annotate threshold labels
ax.text(x.max() * 0.99, 1000, " 1000 ppm", va="bottom", ha="right")
ax.text(x.max() * 0.99, 2000, " 2000 ppm", va="bottom", ha="right")
ax.text(x.max() * 0.99, 3000, " 3000 ppm", va="bottom", ha="right")

# Mark crossing points
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
# Table
# -----------------------------
with st.expander("Show simulation data table"):
    st.dataframe(df, use_container_width=True)
