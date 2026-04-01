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

CO2_EMISSION = {
    "standing": 0.022,
    "sitting": 0.018,
    "lying": 0.015,
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

width = st.sidebar.slider("Room Width (m)", 5, 500, 12)
length = st.sidebar.slider("Room Length (m)", 5, 500, 10)
height = st.sidebar.slider("Room Height (m)", 2, 6, 3)

standing = st.sidebar.slider("Standing People", 0, 3000, 15)
sitting = st.sidebar.slider("Sitting People", 0, 3000, 30)
lying = st.sidebar.slider("Lying People", 0, 3000, 5)

airflow = st.sidebar.slider("HVAC Airflow (m³/h)", 0, 300000, 350)
time_total = st.sidebar.slider("Simulation Time (min)", 30, 720, 180)

seed = st.sidebar.number_input("Random Seed", 0, 99999, 42)

# -----------------------------
# FUNCTIONS
# -----------------------------
def volume():
    return width * length * height

def area():
    return width * length

def total_people():
    return standing + sitting + lying

def used_area():
    return (
        standing * PERSON_AREA["standing"] +
        sitting * PERSON_AREA["sitting"] +
        lying * PERSON_AREA["lying"]
    )

def occupancy_ratio():
    return used_area() / area()

def air_quality_label(co2):
    if co2 < 800:
        return "Good"
    elif co2 < 1000:
        return "Moderate"
    elif co2 < 1500:
        return "Warning"
    elif co2 < 2000:
        return "Poor"
    else:
        return "Very Poor"

def simulate():
    V = volume()
    co2 = INITIAL_CO2
    rows = []

    gen_per_hour = (
        standing * CO2_EMISSION["standing"] +
        sitting * CO2_EMISSION["sitting"] +
        lying * CO2_EMISSION["lying"]
    )

    for t in range(time_total + 1):
        rows.append({"time": t/60, "co2": co2})

        if t == time_total:
            break

        gen = gen_per_hour / 60
        vent = airflow / 60

        indoor = co2 / 1_000_000
        outdoor = OUTDOOR_CO2 / 1_000_000

        co2_vol = indoor * V
        co2_vol += gen
        co2_vol -= indoor * vent
        co2_vol += outdoor * vent

        co2 = (co2_vol / V) * 1_000_000

    return pd.DataFrame(rows)

# -----------------------------
# COLLISION FIXED
# -----------------------------
def intersects(a, b):
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    return not (
        ax + aw <= bx or bx + bw <= ax or
        ay + ah <= by or by + bh <= ay
    )

def can_place(rect, placed):
    for p in placed:
        if intersects(rect, p):
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

    for posture, w, h in people:
        success = False

        for _ in range(300):
            x = rng.uniform(0, width - w)
            y = rng.uniform(0, length - h)

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
df = simulate()
placed, failed = place_people()

# -----------------------------
# SUMMARY
# -----------------------------
c1, c2, c3, c4 = st.columns(4)

c1.metric("Room Area", f"{area():.1f} m²")
c2.metric("Room Volume", f"{volume():.1f} m³")
c3.metric("People", total_people())
c4.metric("Occupancy", f"{occupancy_ratio()*100:.1f}%")

c5, c6, c7, c8 = st.columns(4)

c5.metric("Used Area", f"{used_area():.1f} m²")
c6.metric("Free Area", f"{area()-used_area():.1f} m²")
c7.metric("Max CO₂", f"{df['co2'].max():.0f} ppm")
c8.metric("Final Quality", air_quality_label(df["co2"].iloc[-1]))

# -----------------------------
# LAYOUT
# -----------------------------
left, right = st.columns(2)

# CO2 GRAPH
with left:
    st.subheader("CO₂ Over Time")

    fig = plt.figure()
    plt.plot(df["time"], df["co2"])

    plt.axhline(800, linestyle="--")
    plt.axhline(1000, linestyle="--")
    plt.axhline(1500, linestyle="--")
    plt.axhline(2000, linestyle="--")

    plt.xlabel("Time (hours)")
    plt.ylabel("CO₂ (ppm)")
    plt.grid(True)

    st.pyplot(fig)

# ROOM GRAPH
with right:
    st.subheader("Spatial Layout")

    fig2 = plt.figure()
    ax = plt.gca()

    ax.add_patch(Rectangle((0, 0), width, length, fill=False))

    colors = {
        "standing": "blue",
        "sitting": "orange",
        "lying": "green"
    }

    for x, y, w, h, p in placed:
        ax.add_patch(Rectangle((x, y), w, h, color=colors[p], alpha=0.7))

    ax.set_xlim(0, width)
    ax.set_ylim(0, length)
    ax.set_aspect("equal")

    st.pyplot(fig2)

if failed > 0:
    st.warning(f"{failed} people could not be placed (overcrowded).")