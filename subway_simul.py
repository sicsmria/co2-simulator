import math
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch

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

POSTURE_ORDER = ["standing", "seated", "lying"]

# -----------------------------
# Helper functions
# -----------------------------
def get_posture_area(posture: str) -> float:
    spec = POSTURE_SPECS[posture]
    return spec["width_m"] * spec["depth_m"]


def get_posture_volume(posture: str) -> float:
    spec = POSTURE_SPECS[posture]
    return spec["width_m"] * spec["depth_m"] * spec["height_m"]


def posture_total_people(groups: dict) -> int:
    return sum(groups[p]["count"] for p in POSTURE_ORDER)


def total_co2_generation_m3s(groups: dict) -> float:
    total = 0.0
    for posture in POSTURE_ORDER:
        count = groups[posture]["count"]
        activity = groups[posture]["activity"]
        lps = ACTIVITY_LEVELS[activity]["co2_gen_lps_per_person"]
        total += count * (lps / 1000.0)  # L/s -> m3/s
    return total


def estimate_mixed_capacity(length_m: float, width_m: float, groups: dict, packing_factor: float = 1.3) -> int:
    """
    Approximate capacity by using weighted average area of actual occupant mix.
    """
    total_people = posture_total_people(groups)
    floor_area = length_m * width_m
    if total_people == 0:
        return 0

    weighted_area = 0.0
    for posture in POSTURE_ORDER:
        weighted_area += groups[posture]["count"] * get_posture_area(posture)

    avg_area = weighted_area / total_people
    area_per_person = avg_area * packing_factor
    if area_per_person <= 0:
        return 0

    return int(floor_area // area_per_person)


def simulate_co2(
    volume_m3: float,
    ventilation_m3_per_h: float,
    groups: dict,
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

    Q = ventilation_m3_per_h / 3600.0  # m3/s
    G = total_co2_generation_m3s(groups)

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
    length_m: float,
    width_m: float,
    height_m: float,
    groups: dict,
    ventilation_m3_per_h: float,
    outdoor_co2_ppm: float = 420.0,
    initial_co2_ppm: float = 420.0,
    duration_min: int = 180,
    packing_factor: float = 1.3,
):
    floor_area_m2 = length_m * width_m
    volume_m3 = floor_area_m2 * height_m
    total_people = posture_total_people(groups)

    capacity = estimate_mixed_capacity(
        length_m=length_m,
        width_m=width_m,
        groups=groups,
        packing_factor=packing_factor,
    )

    overcrowded = total_people > capacity if capacity > 0 else total_people > 0

    sim_results = simulate_co2(
        volume_m3=volume_m3,
        ventilation_m3_per_h=ventilation_m3_per_h,
        groups=groups,
        outdoor_co2_ppm=outdoor_co2_ppm,
        initial_co2_ppm=initial_co2_ppm,
        duration_min=duration_min,
    )

    return {
        "floor_area_m2": floor_area_m2,
        "volume_m3": volume_m3,
        "capacity": capacity,
        "overcrowded": overcrowded,
        "total_people": total_people,
        "results": sim_results,
    }


def get_threshold_crossing_time(df: pd.DataFrame, threshold: float):
    crossed = df[df["co2_ppm"] >= threshold]
    if crossed.empty:
        return None
    return float(crossed.iloc[0]["time_min"])


def expand_dims_with_packing(posture: str, packing_factor: float):
    spec = POSTURE_SPECS[posture]
    scale = math.sqrt(packing_factor)
    return {
        "width": spec["width_m"] * scale,
        "depth": spec["depth_m"] * scale,
        "height": spec["height_m"],
    }


# -----------------------------
# Layout generation
# -----------------------------
def generate_occupant_positions(length_m: float, width_m: float, groups: dict, packing_factor: float):
    """
    Simple row packing from top to bottom by posture type.
    Returns placed occupants and overflow count.
    """
    placed = []
    overflow = 0

    current_y = width_m
    gap_y = 0.15
    gap_x = 0.10

    for posture in POSTURE_ORDER:
        count = groups[posture]["count"]
        if count <= 0:
            continue

        dims = expand_dims_with_packing(posture, packing_factor)
        occ_w = dims["width"]
        occ_d = dims["depth"]

        band_h = occ_w + gap_y
        current_y -= band_h
        if current_y < 0:
            overflow += count
            continue

        usable_length = max(length_m - 0.2, 0.0)
        per_row = max(1, int((usable_length + gap_x) // (occ_d + gap_x)))

        row_count = 0
        col_count = 0
        placed_for_posture = 0

        while placed_for_posture < count:
            y = current_y - row_count * band_h
            if y < 0:
                overflow += (count - placed_for_posture)
                break

            x = 0.1 + col_count * (occ_d + gap_x)

            if x + occ_d > length_m - 0.1:
                row_count += 1
                col_count = 0
                continue

            placed.append(
                {
                    "posture": posture,
                    "x": x,
                    "y": y,
                    "draw_depth": POSTURE_SPECS[posture]["depth_m"],
                    "draw_width": POSTURE_SPECS[posture]["width_m"],
                    "height": POSTURE_SPECS[posture]["height_m"],
                }
            )

            placed_for_posture += 1
            col_count += 1

        current_y -= max(0, row_count) * band_h + 0.10

    return placed, overflow


# -----------------------------
# Drawing helpers
# -----------------------------
def draw_person_top(ax, posture: str, x: float, y: float, depth_m: float, width_m: float):
    """
    Pretty top-view glyph while respecting outer footprint.
    """
    body = FancyBboxPatch(
        (x, y),
        depth_m,
        width_m,
        boxstyle="round,pad=0.01,rounding_size=0.05",
        linewidth=1.0,
        fill=False,
    )
    ax.add_patch(body)

    if posture == "standing":
        r = min(depth_m, width_m) * 0.18
        cx = x + depth_m * 0.5
        cy = y + width_m * 0.5
        ax.add_patch(Circle((cx, cy), r, fill=False, linewidth=1.0))
        ax.plot([cx, cx], [cy - r, y + width_m * 0.20], linewidth=1.0)
        ax.plot([cx, x + depth_m * 0.25], [cy - r * 0.2, y + width_m * 0.40], linewidth=1.0)
        ax.plot([cx, x + depth_m * 0.75], [cy - r * 0.2, y + width_m * 0.40], linewidth=1.0)

    elif posture == "seated":
        head_r = min(depth_m, width_m) * 0.14
        ax.add_patch(Circle((x + depth_m * 0.28, y + width_m * 0.72), head_r, fill=False, linewidth=1.0))
        ax.plot([x + depth_m * 0.28, x + depth_m * 0.42], [y + width_m * 0.60, y + width_m * 0.38], linewidth=1.0)
        ax.plot([x + depth_m * 0.42, x + depth_m * 0.70], [y + width_m * 0.38, y + width_m * 0.38], linewidth=1.0)
        ax.plot([x + depth_m * 0.68, x + depth_m * 0.68], [y + width_m * 0.38, y + width_m * 0.70], linewidth=1.0)

    elif posture == "lying":
        head_r = min(depth_m, width_m) * 0.18
        ax.add_patch(Circle((x + depth_m * 0.15, y + width_m * 0.50), head_r, fill=False, linewidth=1.0))
        ax.plot([x + depth_m * 0.28, x + depth_m * 0.85], [y + width_m * 0.50, y + width_m * 0.50], linewidth=1.0)
        ax.plot([x + depth_m * 0.55, x + depth_m * 0.78], [y + width_m * 0.50, y + width_m * 0.72], linewidth=1.0)
        ax.plot([x + depth_m * 0.55, x + depth_m * 0.78], [y + width_m * 0.50, y + width_m * 0.28], linewidth=1.0)


def draw_person_side(ax, posture: str, x: float, baseline: float, depth_m: float, height_m: float):
    """
    Pretty side-view glyph while respecting outer depth/height.
    """
    frame = FancyBboxPatch(
        (x, baseline),
        depth_m,
        height_m,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        linewidth=0.8,
        fill=False,
        alpha=0.7,
    )
    ax.add_patch(frame)

    if posture == "standing":
        head_r = min(depth_m, height_m) * 0.10
        cx = x + depth_m * 0.5
        cy = baseline + height_m * 0.85
        ax.add_patch(Circle((cx, cy), head_r, fill=False, linewidth=1.0))
        ax.plot([cx, cx], [baseline + height_m * 0.28, cy - head_r], linewidth=1.0)
        ax.plot([cx, x + depth_m * 0.25], [baseline + height_m * 0.62, baseline + height_m * 0.48], linewidth=1.0)
        ax.plot([cx, x + depth_m * 0.75], [baseline + height_m * 0.62, baseline + height_m * 0.48], linewidth=1.0)
        ax.plot([cx, x + depth_m * 0.30], [baseline + height_m * 0.28, baseline], linewidth=1.0)
        ax.plot([cx, x + depth_m * 0.70], [baseline + height_m * 0.28, baseline], linewidth=1.0)

    elif posture == "seated":
        head_r = min(depth_m, height_m) * 0.10
        ax.add_patch(Circle((x + depth_m * 0.30, baseline + height_m * 0.80), head_r, fill=False, linewidth=1.0))
        ax.plot([x + depth_m * 0.30, x + depth_m * 0.45], [baseline + height_m * 0.68, baseline + height_m * 0.45], linewidth=1.0)
        ax.plot([x + depth_m * 0.45, x + depth_m * 0.75], [baseline + height_m * 0.45, baseline + height_m * 0.45], linewidth=1.0)
        ax.plot([x + depth_m * 0.72, x + depth_m * 0.72], [baseline + height_m * 0.45, baseline + height_m * 0.15], linewidth=1.0)
        ax.plot([x + depth_m * 0.45, x + depth_m * 0.60], [baseline + height_m * 0.45, baseline + height_m * 0.10], linewidth=1.0)
        ax.plot([x + depth_m * 0.60, x + depth_m * 0.80], [baseline + height_m * 0.10, baseline + height_m * 0.10], linewidth=1.0)

    elif posture == "lying":
        head_r = min(depth_m, height_m) * 0.18
        ax.add_patch(Circle((x + depth_m * 0.12, baseline + height_m * 0.55), head_r, fill=False, linewidth=1.0))
        ax.plot([x + depth_m * 0.22, x + depth_m * 0.82], [baseline + height_m * 0.50, baseline + height_m * 0.50], linewidth=1.0)
        ax.plot([x + depth_m * 0.52, x + depth_m * 0.74], [baseline + height_m * 0.50, baseline + height_m * 0.66], linewidth=1.0)
        ax.plot([x + depth_m * 0.52, x + depth_m * 0.74], [baseline + height_m * 0.50, baseline + height_m * 0.34], linewidth=1.0)


def draw_top_view(length_m: float, width_m: float, groups: dict, packing_factor: float):
    placed, overflow = generate_occupant_positions(length_m, width_m, groups, packing_factor)

    fig, ax = plt.subplots(figsize=(9, 6))
    room = FancyBboxPatch(
        (0, 0),
        length_m,
        width_m,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        linewidth=2.0,
        fill=False,
    )
    ax.add_patch(room)

    for occ in placed:
        draw_person_top(
            ax=ax,
            posture=occ["posture"],
            x=occ["x"],
            y=occ["y"],
            depth_m=occ["draw_depth"],
            width_m=occ["draw_width"],
        )

    # Labels
    legend_y = width_m + max(width_m * 0.03, 0.2)
    legend_text = (
        f"Standing: {groups['standing']['count']}   "
        f"Seated: {groups['seated']['count']}   "
        f"Lying: {groups['lying']['count']}"
    )
    ax.text(length_m * 0.02, legend_y, legend_text, fontsize=10)

    if overflow > 0:
        ax.text(
            length_m * 0.5,
            width_m * 0.5,
            f"Overflow: {overflow} person(s)",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.85),
        )

    ax.set_xlim(-0.2, length_m + 0.2)
    ax.set_ylim(-0.2, width_m + max(width_m * 0.10, 0.5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Top View Layout")
    ax.set_xlabel("Length (m)")
    ax.set_ylabel("Width (m)")
    ax.grid(True, alpha=0.25)

    return fig


def draw_side_view(length_m: float, height_m: float, groups: dict, packing_factor: float):
    fig, ax = plt.subplots(figsize=(9, 4.8))

    room = FancyBboxPatch(
        (0, 0),
        length_m,
        height_m,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=2.0,
        fill=False,
    )
    ax.add_patch(room)

    total = posture_total_people(groups)
    if total == 0:
        total = 1

    visible_limit = 24
    visible_people = min(posture_total_people(groups), visible_limit)

    if visible_people > 0:
        x_cursor = 0.3
        x_gap = max(length_m / max(visible_people + 2, 6), 0.45)

        sequence = []
        for posture in POSTURE_ORDER:
            sequence.extend([posture] * groups[posture]["count"])
        sequence = sequence[:visible_people]

        for posture in sequence:
            spec = POSTURE_SPECS[posture]
            depth = min(spec["depth_m"], x_gap * 0.8)
            x = x_cursor
            draw_person_side(
                ax=ax,
                posture=posture,
                x=x,
                baseline=0.0,
                depth_m=depth,
                height_m=spec["height_m"],
            )
            x_cursor += x_gap
            if x_cursor > length_m - 0.4:
                break

    if posture_total_people(groups) > visible_limit:
        ax.text(
            length_m * 0.98,
            height_m * 0.95,
            f"Showing first {visible_limit} people",
            ha="right",
            va="top",
            fontsize=9,
        )

    ax.set_xlim(-0.2, length_m + 0.2)
    ax.set_ylim(0, max(height_m * 1.08, 2.2))
    ax.set_title("Side View Section")
    ax.set_xlabel("Length (m)")
    ax.set_ylabel("Height (m)")
    ax.grid(True, alpha=0.25)

    return fig


# -----------------------------
# UI
# -----------------------------
st.title("Subway Shelter CO₂ Simulation")
st.caption("ISO 7250-based simplified geometry with mixed postures, automatic update, and visual layout")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1. Space geometry")
    length_m = st.number_input("Length (m)", min_value=1.0, value=20.0, step=1.0)
    width_m = st.number_input("Width (m)", min_value=1.0, value=10.0, step=1.0)
    height_m = st.number_input("Height (m)", min_value=1.0, value=3.5, step=0.1)

    st.subheader("2. Occupants")

    standing_count = st.number_input("Standing people", min_value=0, value=40, step=1)
    standing_activity = st.selectbox(
        "Standing activity",
        options=["stable", "crowded", "moving"],
        format_func=lambda x: ACTIVITY_LEVELS[x]["label"],
        key="standing_activity",
    )

    seated_count = st.number_input("Seated people", min_value=0, value=20, step=1)
    seated_activity = st.selectbox(
        "Seated activity",
        options=["stable", "crowded", "moving"],
        format_func=lambda x: ACTIVITY_LEVELS[x]["label"],
        key="seated_activity",
    )

    lying_count = st.number_input("Lying people", min_value=0, value=10, step=1)
    lying_activity = st.selectbox(
        "Lying activity",
        options=["stable", "crowded", "moving"],
        format_func=lambda x: ACTIVITY_LEVELS[x]["label"],
        key="lying_activity",
    )

with col_right:
    st.subheader("3. Ventilation / CO₂ settings")
    ventilation_m3_per_h = st.number_input("Ventilation rate (m³/h)", min_value=0.0, value=5000.0, step=100.0)
    outdoor_co2_ppm = st.number_input("Outdoor CO₂ (ppm)", min_value=300.0, value=420.0, step=10.0)
    initial_co2_ppm = st.number_input("Initial indoor CO₂ (ppm)", min_value=300.0, value=420.0, step=10.0)
    duration_min = st.slider("Simulation duration (min)", min_value=10, max_value=720, value=180, step=10)
    packing_factor = st.slider("Packing factor", min_value=1.0, max_value=2.0, value=1.3, step=0.05)

groups = {
    "standing": {"count": int(standing_count), "activity": standing_activity},
    "seated": {"count": int(seated_count), "activity": seated_activity},
    "lying": {"count": int(lying_count), "activity": lying_activity},
}

# -----------------------------
# Run automatically
# -----------------------------
output = run_shelter_simulation(
    length_m=length_m,
    width_m=width_m,
    height_m=height_m,
    groups=groups,
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

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Floor area", f"{output['floor_area_m2']:.1f} m²")
with m2:
    st.metric("Volume", f"{output['volume_m3']:.1f} m³")
with m3:
    st.metric("Total people", f"{output['total_people']}")
with m4:
    st.metric("Estimated capacity", f"{output['capacity']}")
with m5:
    density = output["total_people"] / output["floor_area_m2"] if output["floor_area_m2"] > 0 else 0
    st.metric("Density", f"{density:.2f} persons/m²")

if output["overcrowded"]:
    st.error("Occupancy exceeds estimated mixed-posture capacity.")
else:
    st.success("Occupancy is within estimated mixed-posture capacity.")

# -----------------------------
# Per-posture info
# -----------------------------
st.subheader("Per-Posture Settings Summary")

info_cols = st.columns(3)
for idx, posture in enumerate(POSTURE_ORDER):
    spec = POSTURE_SPECS[posture]
    with info_cols[idx]:
        st.markdown(f"**{spec['label']}**")
        st.write(f"Count: {groups[posture]['count']}")
        st.write(f"Activity: {ACTIVITY_LEVELS[groups[posture]['activity']]['label']}")
        st.write(f"Footprint: {spec['depth_m']:.2f} m × {spec['width_m']:.2f} m")
        st.write(f"Height: {spec['height_m']:.2f} m")

# -----------------------------
# Visualization
# -----------------------------
st.subheader("Space and Occupant Visualization")

v1, v2 = st.columns(2)

with v1:
    top_fig = draw_top_view(
        length_m=length_m,
        width_m=width_m,
        groups=groups,
        packing_factor=packing_factor,
    )
    st.pyplot(top_fig)

with v2:
    side_fig = draw_side_view(
        length_m=length_m,
        height_m=height_m,
        groups=groups,
        packing_factor=packing_factor,
    )
    st.pyplot(side_fig)

# -----------------------------
# Threshold info
# -----------------------------
st.subheader("Threshold Crossing Time")

t1, t2, t3 = st.columns(3)
with t1:
    st.metric("1000 ppm", "Not reached" if t_1000 is None else f"{t_1000:.1f} min")
with t2:
    st.metric("2000 ppm", "Not reached" if t_2000 is None else f"{t_2000:.1f} min")
with t3:
    st.metric("3000 ppm", "Not reached" if t_3000 is None else f"{t_3000:.1f} min")

# -----------------------------
# CO2 plot
# -----------------------------
st.subheader("CO₂ Concentration Over Time")

fig, ax = plt.subplots(figsize=(12, 6))
x = df["time_min"]
y = df["co2_ppm"]

ax.axhspan(0, 1000, alpha=0.10)
ax.axhspan(1000, 2000, alpha=0.14)
ax.axhspan(2000, 3000, alpha=0.18)
ax.axhspan(3000, max(4000, y.max() * 1.05), alpha=0.24)

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
