import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import streamlit as st

# -----------------------------
# CONFIG
# -----------------------------
OUTDOOR_CO2 = 420
INITIAL_CO2 = 450

# Base CO2 emission by posture [m3/h/person]
POSTURE_CO2 = {
    "standing": 0.022,
    "sitting": 0.018,
    "lying": 0.015,
}

# Activity multipliers
ACTIVITY_FACTOR = {
    "stable": 1.0,
    "anxious_crowded": 1.3,
    "moving": 1.6,
}

PERSON_AREA = {
    "standing": 0.30,
    "sitting": 0.50,
    "lying": 1.80,
}

PERSON_SHAPE = {
    "standing": (0.5, 0.6),
    "sitting": (0.7, 0.7),
    "lying": (1.8, 0.8),
}

st.set_page_config(page_title="CO2 + Occupancy Simulator", layout="wide")
st.title("Indoor CO₂ & Spatial Occupancy Simulator")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Settings")

width = st.sidebar.slider("Room Width (m)", 5, 3000, 12)
length = st.sidebar.slider("Room Length (m)", 5, 100, 10)
height = st.sidebar.slider("Room Height (m)", 2, 6, 3)

standing = st.sidebar.slider("Standing People", 0, 3000, 15)
sitting = st.sidebar.slider("Sitting People", 0, 3000, 30)
lying = st.sidebar.slider("Lying People", 0, 3000, 5)

activity_state = st.sidebar.selectbox(
    "Activity State",
    ["stable", "anxious_crowded", "moving"],
    index=0
)

airflow = st.sidebar.slider("HVAC Airflow (m³/h)", 0, 200000, 350)
time_total = st.sidebar.slider("Simulation Time (min)", 30, 720, 180)
seed = st.sidebar.number_input("Random Seed", 0, 99999, 42)

# -----------------------------
# FUNCTIONS
# -----------------------------
def room_volume():
    return width * length * height

def room_area():
    return width * length

def total_people():
    return standing + sitting + lying

def total_used_area():
    return (
        standing * PERSON_AREA["standing"] +
        sitting * PERSON_AREA["sitting"] +
        lying * PERSON_AREA["lying"]
    )

def occupancy_ratio():
    if room_area() == 0:
        return 0
    return total_used_area() / room_area()

def air_quality_label(co2):
    if co2 < 800:
        return "Good"
    elif co2 < 1000:
        return "Moderate"
    elif co2 < 1500:
        return "Warning"
    elif co2 < 2000:
        return "Poor"
    return "Very Poor"

def crowd_label(ratio):
    if ratio < 0.2:
        return "Spacious"
    elif ratio < 0.4:
        return "Normal"
    elif ratio < 0.6:
        return "Crowded"
    elif ratio < 0.8:
        return "Very Crowded"
    return "Overcrowded"

def per_person_emission(posture, state):
    return POSTURE_CO2[posture] * ACTIVITY_FACTOR[state]

def total_generation_per_hour(state):
    return (
        standing * per_person_emission("standing", state) +
        sitting * per_person_emission("sitting", state) +
        lying * per_person_emission("lying", state)
    )

def simulate_co2(state):
    V = room_volume()
    co2 = INITIAL_CO2
    rows = []

    gen_per_hour = total_generation_per_hour(state)

    for t in range(time_total + 1):
        rows.append({
            "time_min": t,
            "time_hour": t / 60,
            "co2": co2,
            "air_quality": air_quality_label(co2),
        })

        if t == time_total:
            break

        gen = gen_per_hour / 60.0
        vent = airflow / 60.0

        indoor_frac = co2 / 1_000_000.0
        outdoor_frac = OUTDOOR_CO2 / 1_000_000.0

        co2_volume = indoor_frac * V
        removed = indoor_frac * vent
        incoming = outdoor_frac * vent

        co2_volume = co2_volume + gen - removed + incoming
        co2 = max((co2_volume / V) * 1_000_000.0, 0)

    return pd.DataFrame(rows)

# -----------------------------
# LAYOUT / PLACEMENT
# -----------------------------
def intersects(a, b):
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    return not (
        ax + aw <= bx or
        bx + bw <= ax or
        ay + ah <= by or
        by + bh <= ay
    )

def can_place(rect, placed):
    x, y, w, h = rect[:4]

    if x < 0 or y < 0 or x + w > width or y + h > length:
        return False

    for other in placed:
        if intersects(rect, other):
            return False
    return True

def place_people():
    rng = random.Random(seed)

    people = (
        [("lying", *PERSON_SHAPE["lying"])] * lying +
        [("sitting", *PERSON_SHAPE["sitting"])] * sitting +
        [("standing", *PERSON_SHAPE["standing"])] * standing
    )

    placed = []
    failed = 0

    for posture, base_w, base_h in people:
        success = False

        orientations = [(base_w, base_h)]
        if posture in ("lying", "sitting") and base_w != base_h:
            orientations.append((base_h, base_w))

        for _ in range(500):
            w, h = rng.choice(orientations)
            x = rng.uniform(0, max(width - w, 0))
            y = rng.uniform(0, max(length - h, 0))
            rect = (x, y, w, h)

            if can_place(rect, placed):
                placed.append((x, y, w, h, posture))
                success = True
                break

        if not success:
            failed += 1

    return placed, failed

# -----------------------------
# RUN
# -----------------------------
df = simulate_co2(activity_state)
placed_people, failed_count = place_people()

max_co2 = df["co2"].max()
final_co2 = df["co2"].iloc[-1]

# -----------------------------
# SUMMARY
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Room Area", f"{room_area():.1f} m²")
c2.metric("Room Volume", f"{room_volume():.1f} m³")
c3.metric("People", total_people())
c4.metric("Occupancy", f"{occupancy_ratio()*100:.1f}%")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Used Area", f"{total_used_area():.1f} m²")
c6.metric("Free Area", f"{max(room_area() - total_used_area(), 0):.1f} m²")
c7.metric("Max CO₂", f"{max_co2:.0f} ppm")
c8.metric("Final Quality", air_quality_label(final_co2))

st.info(
    f"Activity State: **{activity_state}** | "
    f"Total CO₂ generation: **{total_generation_per_hour(activity_state):.3f} m³/h** | "
    f"Crowding: **{crowd_label(occupancy_ratio())}**"
)

# -----------------------------
# MAIN PANELS
# -----------------------------
left, right = st.columns([1.05, 1])

with left:
    st.subheader("CO₂ Over Time")

    fig1 = plt.figure(figsize=(8, 5))
    plt.plot(df["time_hour"], df["co2"], linewidth=2)
    plt.axhline(800, linestyle="--", linewidth=1)
    plt.axhline(1000, linestyle="--", linewidth=1)
    plt.axhline(1500, linestyle="--", linewidth=1)
    plt.axhline(2000, linestyle="--", linewidth=1)
    plt.xlabel("Time (hours)")
    plt.ylabel("CO₂ (ppm)")
    plt.title("Indoor CO₂ Concentration")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)

    st.subheader("Emission Assumptions")

    posture_table = pd.DataFrame({
        "Posture": ["standing", "sitting", "lying"],
        "Base Emission (m³/h/person)": [
            POSTURE_CO2["standing"],
            POSTURE_CO2["sitting"],
            POSTURE_CO2["lying"],
        ],
        f"Emission in '{activity_state}' state": [
            per_person_emission("standing", activity_state),
            per_person_emission("sitting", activity_state),
            per_person_emission("lying", activity_state),
        ],
        "Area per Person (m²)": [
            PERSON_AREA["standing"],
            PERSON_AREA["sitting"],
            PERSON_AREA["lying"],
        ]
    })
    st.dataframe(posture_table, use_container_width=True)

with right:
    st.subheader("Spatial Layout")

    fig2 = plt.figure(figsize=(7, 7))
    ax = plt.gca()

    ax.add_patch(Rectangle((0, 0), width, length, fill=False, linewidth=2))

    colors = {
        "standing": "blue",
        "sitting": "orange",
        "lying": "green",
    }

    used_label = {"standing": False, "sitting": False, "lying": False}

    for x, y, w, h, posture in placed_people:
        patch = Rectangle(
            (x, y), w, h,
            facecolor=colors[posture],
            edgecolor="black",
            alpha=0.7,
            label=posture if not used_label[posture] else None
        )
        used_label[posture] = True
        ax.add_patch(patch)

    ax.set_xlim(0, width)
    ax.set_ylim(0, length)
    ax.set_aspect("equal")
    ax.set_xlabel("Width (m)")
    ax.set_ylabel("Length (m)")
    ax.set_title("People Placement by Posture")
    ax.grid(True, alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    st.pyplot(fig2)

    if failed_count > 0:
        st.warning(f"{failed_count} people could not be placed in the room layout.")
    else:
        st.success("All people were placed successfully.")

# -----------------------------
# INTERPRETATION
# -----------------------------
st.subheader("Interpretation")

messages = []

if max_co2 >= 2000:
    messages.append("- CO₂ reaches a very poor air-quality range.")
elif max_co2 >= 1000:
    messages.append("- CO₂ exceeds 1000 ppm, indicating insufficient ventilation for long stays.")
else:
    messages.append("- CO₂ remains relatively controlled under the current settings.")

if activity_state == "stable":
    messages.append("- This assumes calm occupants with baseline breathing rates.")
elif activity_state == "anxious_crowded":
    messages.append("- This assumes increased breathing due to stress or crowding.")
elif activity_state == "moving":
    messages.append("- This assumes active movement, producing the highest CO₂ generation.")

if occupancy_ratio() >= 0.8:
    messages.append("- Spatial occupancy is extremely high, making movement and response difficult.")
elif occupancy_ratio() >= 0.5:
    messages.append("- Spatial density is high and may reduce comfort and safety.")
else:
    messages.append("- Spatial density is still within a relatively manageable range.")

st.markdown("\n".join(messages))
