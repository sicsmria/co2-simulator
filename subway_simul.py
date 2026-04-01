import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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


def get_posture_area(posture: str) -> float:
    spec = POSTURE_SPECS[posture]
    return spec["width_m"] * spec["depth_m"]


def get_posture_volume(posture: str) -> float:
    spec = POSTURE_SPECS[posture]
    return spec["width_m"] * spec["depth_m"] * spec["height_m"]


def estimate_capacity_by_posture(floor_area_m2: float, posture: str, packing_factor: float = 1.3) -> int:
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
        results.append({
            "time_min": time_min,
            "co2_ppm": co2_ppm
        })

        c_in = co2_ppm / 1_000_000.0
        c_out = outdoor_co2_ppm / 1_000_000.0

        dc_dt = (G / volume_m3) - (Q / volume_m3) * (c_in - c_out)
        c_in_next = c_in + dc_dt * dt_sec

        co2_ppm = max(c_in_next * 1_000_000.0, outdoor_co2_ppm)

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
        packing_factor=packing_factor
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


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Subway Shelter CO2 Simulation")

st.subheader("1. Space settings")
floor_area_m2 = st.number_input("Usable floor area (m²)", min_value=10.0, value=500.0, step=10.0)
ceiling_height_m = st.number_input("Ceiling height (m)", min_value=2.0, value=3.5, step=0.1)
ventilation_m3_per_h = st.number_input("Ventilation rate (m³/h)", min_value=0.0, value=10000.0, step=100.0)

st.subheader("2. Occupant settings")
posture = st.selectbox(
    "Occupant posture",
    options=["standing", "seated", "lying"],
    format_func=lambda x: POSTURE_SPECS[x]["label"]
)

activity = st.selectbox(
    "Occupant activity state",
    options=["stable", "crowded", "moving"],
    format_func=lambda x: ACTIVITY_LEVELS[x]["label"]
)

occupants = st.number_input("Number of occupants", min_value=1, value=300, step=1)
duration_min = st.number_input("Simulation duration (min)", min_value=1, value=180, step=10)

st.subheader("3. CO2 settings")
outdoor_co2_ppm = st.number_input("Outdoor CO2 (ppm)", min_value=300.0, value=420.0, step=10.0)
initial_co2_ppm = st.number_input("Initial indoor CO2 (ppm)", min_value=300.0, value=420.0, step=10.0)
packing_factor = st.slider("Packing factor", min_value=1.0, max_value=2.0, value=1.3, step=0.05)

if st.button("Run Simulation"):
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

    st.subheader("Simulation Summary")
    st.write(f"Space volume: **{output['volume_m3']:.1f} m³**")
    st.write(f"Posture area per person: **{output['posture_area_m2_per_person']:.2f} m²/person**")
    st.write(f"Posture volume per person: **{output['posture_volume_m3_per_person']:.2f} m³/person**")
    st.write(f"Estimated max capacity: **{output['capacity']} persons**")

    if output["overcrowded"]:
        st.error("Occupancy exceeds estimated posture-based capacity.")
    else:
        st.success("Occupancy is within estimated posture-based capacity.")

    df = pd.DataFrame(output["results"])
    st.dataframe(df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["time_min"], df["co2_ppm"])
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Indoor CO2 (ppm)")
    ax.set_title("CO2 Concentration Over Time")
    ax.grid(True)
    st.pyplot(fig)
