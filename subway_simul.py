import math
from typing import Dict, List
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Ultimate Underground Simulator", layout="wide")

# =========================================================
# Constants
# =========================================================
OUTDOOR_CO2_PPM = 420.0
OUTDOOR_AH_G_M3 = 8.0 
OUTDOOR_TEMP_C = 20.0 # 지하 기본 온도
OUTDOOR_O2_PCT = 21.0

MAX_HEATMAP_CELLS = 12000

# 1인당 배출량 (CO2: m³/h, 잠열(수증기): g/h, 현열(체온): W)
CO2_GEN_M3_PER_H = {"standing": 0.021, "sitting": 0.018, "lying": 0.015}
MOISTURE_GEN_G_PER_H = {"standing": 75.0, "sitting": 50.0, "lying": 40.0}
HEAT_GEN_W = {"standing": 75.0, "sitting": 60.0, "lying": 45.0}

EQUIPMENT_TYPES = {
    "Supply": {"color": "rgba(0,140,255,0.95)", "symbol": "S", "radius_m": 8.0, "ppm_red": 120.0, "ah_red": 3.0, "temp_red": 3.0, "label": "🟦"},
    "Exhaust": {"color": "rgba(255,90,90,0.95)", "symbol": "E", "radius_m": 10.0, "ppm_red": 150.0, "ah_red": 4.0, "temp_red": 2.0, "label": "🟥"},
    "Purifier": {"color": "rgba(0,180,120,0.95)", "symbol": "P", "radius_m": 6.0, "ppm_red": 90.0, "ah_red": 0.5, "temp_red": -0.5, "label": "🟩"}, # 모터 발열로 온도 상승(-)
}

if "equipment_map" not in st.session_state:
    st.session_state.equipment_map = {}

# =========================================================
# Physics Engine
# =========================================================
def compute_transient_baseline(room_w, room_d, room_h, ach, n_st, n_si, n_ly, t, panic_mode, blackout_t):
    vol = room_w * room_d * room_h
    
    # 패닉 모드: 호흡, 발한, 발열 3배 증가
    multiplier = 3.0 if panic_mode else 1.0
    
    g_co2 = (n_st*0.021 + n_si*0.018 + n_ly*0.015) * multiplier
    g_ah = (n_st*75.0 + n_si*50.0 + n_ly*40.0) * multiplier
    g_heat_w = (n_st*75.0 + n_si*60.0 + n_ly*45.0) * multiplier
    g_temp = g_heat_w * 3.0 # W -> °C*m³/h 변환 계수 (공기 비열 반영)

    def get_deltas(current_t, current_ach):
        q_vent = max(current_ach * vol, 0.001)
        if current_ach > 0:
            decay = (1.0 - math.exp(-current_ach * current_t))
            return (1e6 * g_co2 / q_vent) * decay, (g_ah / q_vent) * decay, (g_temp / q_vent) * decay
        else:
            return (1e6 * g_co2 * current_t)/vol, (g_ah * current_t)/vol, (g_temp * current_t)/vol

    # 전력 고장(Blackout) 시점 전후로 나누어 누적 계산
    if t <= blackout_t:
        d_c, d_a, d_t = get_deltas(t, ach)
    else:
        fail_c, fail_a, fail_t = get_deltas(blackout_t, ach) # 전력 끊긴 시점의 농도
        post_c, post_a, post_t = get_deltas(t - blackout_t, 0.0) # 그 이후 밀폐 누적(ACH=0)
        d_c, d_a, d_t = fail_c + post_c, fail_a + post_a, fail_t + post_t

    base_c = OUTDOOR_CO2_PPM + d_c
    base_a = OUTDOOR_AH_G_M3 + d_a
    base_t = OUTDOOR_TEMP_C + d_t
    base_o2 = OUTDOOR_O2_PCT - (d_c * 1.2 / 10000.0) # CO2 증가량에 비례해 산소 감소

    return vol, base_c, base_a, base_t, base_o2

@st.cache_data(show_spinner=False)
def get_grid(room_w, room_d):
    step = max(1.0, math.sqrt(room_w * room_d / MAX_HEATMAP_CELLS))
    nx, ny = int(room_w/step)+1, int(room_d/step)+1
    return np.meshgrid(np.linspace(0, room_w, nx), np.linspace(0, room_d, ny)), step

def make_trend_charts(room_w, room_d, room_h, ach, n_st, n_si, n_ly, current_t, panic_mode, blackout_t):
    times = np.linspace(0, 24, 49)
    co2_v, o2_v, ah_v, temp_v = [], [], [], []
    for t in times:
        _, c, a, temp, o2 = compute_transient_baseline(room_w, room_d, room_h, ach, n_st, n_si, n_ly, t, panic_mode, blackout_t)
        co2_v.append(c); o2_v.append(o2); ah_v.append(a); temp_v.append(temp)
    
    # Chart 1: CO2 & O2
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=times, y=co2_v, name="CO2 (ppm)", yaxis="y1", line=dict(color="orangered", width=3)))
    fig1.add_trace(go.Scatter(x=times, y=o2_v, name="O2 (%)", yaxis="y2", line=dict(color="green", width=3, dash='dot')))
    fig1.add_vline(x=current_t, line_width=2, line_dash="dash"); fig1.add_vline(x=blackout_t, line_width=2, line_color="red")
    fig1.update_layout(title="Air Quality (CO2 & Oxygen)", height=250, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified",
                       yaxis=dict(title=dict(text="CO2 (ppm)", font=dict(color="orangered")), tickfont=dict(color="orangered")),
                       yaxis2=dict(title=dict(text="O2 (%)", font=dict(color="green")), tickfont=dict(color="green"), overlaying="y", side="right"))
    
    # Chart 2: Temp & Humidity
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=times, y=temp_v, name="Temp (°C)", yaxis="y1", line=dict(color="firebrick", width=3)))
    fig2.add_trace(go.Scatter(x=times, y=ah_v, name="Hum (g/m³)", yaxis="y2", line=dict(color="royalblue", width=3, dash='dot')))
    fig2.add_vline(x=current_t, line_width=2, line_dash="dash"); fig2.add_vline(x=blackout_t, line_width=2, line_color="red")
    fig2.update_layout(title="Thermal Comfort (Temp & Humidity)", height=250, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified",
                       yaxis=dict(title=dict(text="Temp (°C)", font=dict(color="firebrick")), tickfont=dict(color="firebrick")),
                       yaxis2=dict(title=dict(text="Abs. Hum (g/m³)", font=dict(color="royalblue")), tickfont=dict(color="royalblue"), overlaying="y", side="right"))
    return fig1, fig2

def equipment_list_from_map(equipment_map, room_w, room_d):
    eqs = []
    dx, dy = room_w / 10, room_d / 10
    for (ix, iy), eq_type in equipment_map.items():
        if eq_type in EQUIPMENT_TYPES:
            eq = EQUIPMENT_TYPES[eq_type]
            eqs.append({"ix": ix, "iy": iy, "x": (ix*dx)-0.5, "y": (iy*dy)-0.5, "w": 1.0, "d": 1.0, "type": eq_type, **eq})
    return eqs

def get_non_overlapping_points(room_w, room_d, n_st, n_si, n_ly, seed=42):
    total_n = n_st + n_si + n_ly
    if total_n == 0: return [], [], [], [], [], []
    nx = max(1, int(math.ceil(math.sqrt(total_n * (room_w / room_d)))))
    ny = max(1, int(math.ceil(total_n / nx)))
    while nx * ny < total_n:
        if nx / room_w < ny / room_d: nx += 1
        else: ny += 1
    dx, dy = room_w / nx, room_d / ny
    points = [(ix * dx + dx / 2, iy * dy + dy / 2) for ix in range(nx) for iy in range(ny)]
    rng = np.random.default_rng(seed)
    rng.shuffle(points)
    pts = points[:total_n]
    fx = [p[0] + jx for p, jx in zip(pts, rng.uniform(-dx/3, dx/3, total_n))]
    fy = [p[1] + jy for p, jy in zip(pts, rng.uniform(-dy/3, dy/3, total_n))]
    return fx[:n_st], fy[:n_st], fx[n_st:n_st+n_si], fy[n_st:n_st+n_si], fx[n_st+n_si:], fy[n_st+n_si:]

def make_heatmap_fig(room_w, room_d, X, Y, Z, eqs, t, n_st, n_si, n_ly, title, scale, zmin, zmax, cbtitle):
    fig = go.Figure(go.Heatmap(x=X[0], y=Y[:,0], z=Z, colorscale=scale, zmin=zmin, zmax=zmax, colorbar=dict(title=cbtitle), zsmooth="best"))
    for eq in eqs:
        fig.add_shape(type="rect", x0=eq["x"]+0.15, y0=eq["y"]+0.15, x1=eq["x"]+0.85, y1=eq["y"]+0.85, line=dict(color="black", width=1), fillcolor=eq["color"])
        fig.add_trace(go.Scatter(x=[eq["x"]+0.5], y=[eq["y"]+0.5], mode="text", text=[eq["symbol"]], textfont=dict(color="white"), showlegend=False))
    st_x, st_y, si_x, si_y, ly_x, ly_y = get_non_overlapping_points(room_w, room_d, n_st, n_si, n_ly, seed=99)
    if n_st > 0: fig.add_trace(go.Scatter(x=st_x, y=st_y, mode="markers", name=f"Standing ({n_st})", marker=dict(size=6, color="rgba(20,20,20,0.9)", symbol="circle", line=dict(width=0.5, color="white")), hoverinfo="name"))
    if n_si > 0: fig.add_trace(go.Scatter(x=si_x, y=si_y, mode="markers", name=f"Sitting ({n_si})", marker=dict(size=8, color="rgba(40,90,255,0.9)", symbol="square", line=dict(width=0.5, color="white")), hoverinfo="name"))
    if n_ly > 0: fig.add_trace(go.Scatter(x=ly_x, y=ly_y, mode="markers", name=f"Lying ({n_ly})", marker=dict(size=11, color="rgba(0,150,90,0.9)", symbol="diamond-wide", line=dict(width=0.5, color="white")), hoverinfo="name"))
    fig.update_layout(width=900, height=500, title=f"{title} (T={t:.1f}h)", margin=dict(l=20, r=20, t=40, b=20))
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    return fig

# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("🏢 Room Physics")
room_w = st.sidebar.slider("Width (m)", 5.0, 500.0, 100.0)
room_d = st.sidebar.slider("Depth (m)", 5.0, 500.0, 50.0)
room_h = st.sidebar.slider("Height (m)", 2.0, 6.0, 3.0)
ach = st.sidebar.slider("Ventilation (ACH)", 0.0, 15.0, 2.0)

st.sidebar.header("👥 Population")
n_st = st.sidebar.number_input("Standing", 0, 5000, 100)
n_si = st.sidebar.number_input("Sitting", 0, 5000, 50)
n_ly = st.sidebar.number_input("Lying", 0, 5000, 10)

st.sidebar.header("🚨 Emergency Scenario")
panic_mode = st.sidebar.checkbox("Panic Mode (Emissions x3)")
blackout_t = st.sidebar.slider("⚡ Blackout Time (Hours)", 0.0, 24.0, 24.0, help="지정된 시간에 전력이 끊겨 ACH와 모든 설비가 정지합니다.")

st.sidebar.header("🛠️ Tools")
tool = st.sidebar.radio("Equipment", ["Supply", "Exhaust", "Purifier", "Eraser"])
if st.sidebar.button("Clear All"): st.session_state.equipment_map = {}

# =========================================================
# Main UI
# =========================================================
st.title("Ultimate Underground Simulator (O2/Temp/Power)")

elapsed_t = st.select_slider("🕒 Time Machine (Hours Elapsed)", options=np.round(np.linspace(0, 24, 241), 1), value=2.0)

# 시뮬레이션 연산
vol, base_c, base_a, base_t, base_o2 = compute_transient_baseline(room_w, room_d, room_h, ach, n_st, n_si, n_ly, elapsed_t, panic_mode, blackout_t)
current_ach = ach if elapsed_t <= blackout_t else 0.0

# 상단 알림 (전력 & 산소 상태)
if elapsed_t > blackout_t:
    st.error(f"⚡ **블랙아웃 발생:** {blackout_t}시간 부로 전력이 끊겨 모든 환기 설비가 정지되었습니다. (현재 환기율: 0 ACH)")
elif panic_mode:
    st.warning("⚠️ **패닉 모드:** 사람들의 호흡량과 발열량이 3배로 폭증한 상태입니다.")

if base_o2 < 18.0:
    st.error(f"☠️ **질식 위험!** 산소 농도가 {base_o2:.1f}%로 치명적인 수준입니다. (18% 미만)")
elif base_o2 < 19.5:
    st.warning(f"⚠️ **산소 부족 주의:** 산소 농도가 {base_o2:.1f}%로 떨어졌습니다. 두통 및 호흡 곤란 발생 (19.5% 미만)")

# 메트릭 표시
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("CO2 Level", f"{base_c:.0f} ppm", f"{base_c - OUTDOOR_CO2_PPM:.0f} ▲" if base_c > OUTDOOR_CO2_PPM else "")
m2.metric("Oxygen (O2)", f"{base_o2:.1f} %", f"{base_o2 - OUTDOOR_O2_PCT:.1f} ▼" if base_o2 < OUTDOOR_O2_PCT else "")
m3.metric("Temperature", f"{base_t:.1f} °C", f"{base_t - OUTDOOR_TEMP_C:.1f} ▲" if base_t > OUTDOOR_TEMP_C else "")
m4.metric("Abs. Humidity", f"{base_a:.1f} g/m³", f"{base_a - OUTDOOR_AH_G_M3:.1f} ▲" if base_a > OUTDOOR_AH_G_M3 else "")
m5.metric("System Flow", f"{max(current_ach * vol, 0):.0f} m³/h")

# 트렌드 차트
c1, c2 = st.columns(2)
fig_air, fig_therm = make_trend_charts(room_w, room_d, room_h, ach, n_st, n_si, n_ly, elapsed_t, panic_mode, blackout_t)
c1.plotly_chart(fig_air, use_container_width=True)
c2.plotly_chart(fig_therm, use_container_width=True)

# 설비 맵 필드 연산 (전력이 끊기면 t_factor = 0 으로 로컬 정화 효과 소멸)
eqs = equipment_list_from_map(st.session_state.equipment_map, room_w, room_d)
(X, Y), g_step = get_grid(room_w, room_d)
Z_c, Z_a, Z_t = np.ones_like(X)*base_c, np.ones_like(X)*base_a, np.ones_like(X)*base_t

if elapsed_t <= blackout_t:
    t_factor = (1.0 - math.exp(-ach*elapsed_t)) if ach > 0 else 1.0
    for eq in eqs:
        gauss = np.exp(-((X - (eq['x']+0.5))**2 + (Y - (eq['y']+0.5))**2) / (2.0 * max(eq['radius_m'], g_step)**2))
        Z_c -= (eq['ppm_red']/max(1+0.1*ach,1)) * t_factor * gauss
        Z_a -= (eq['ah_red']/max(1+0.1*ach,1)) * t_factor * gauss
        Z_t -= (eq['temp_red']/max(1+0.1*ach,1)) * t_factor * gauss

# 히트맵 탭
t1, t2, t3 = st.tabs(["🔥 CO2", "💧 Humidity", "🌡️ Temperature"])
with t1: st.plotly_chart(make_heatmap_fig(room_w, room_d, X, Y, Z_c, eqs, elapsed_t, n_st, n_si, n_ly, "CO2 (ppm)", "Turbo", 400, 3000, "ppm"), use_container_width=False)
with t2: st.plotly_chart(make_heatmap_fig(room_w, room_d, X, Y, Z_a, eqs, elapsed_t, n_st, n_si, n_ly, "Humidity (g/m³)", "Tealrose", 5, 30, "g/m³"), use_container_width=False)
with t3: st.plotly_chart(make_heatmap_fig(room_w, room_d, X, Y, Z_t, eqs, elapsed_t, n_st, n_si, n_ly, "Temperature (°C)", "Thermal", 15, 45, "°C"), use_container_width=False)

# 에디터
st.divider()
nx, ny, dx, dy = 11, 11, room_w/10, room_d/10
st.markdown(f"### 📍 Equipment Placement Grid")
legend_cols = st.columns(5)
for i, (k, v) in enumerate(EQUIPMENT_TYPES.items()): legend_cols[i].write(f"{v['label']} {k}")
for iy in range(ny):
    cols = st.columns([0.8] + [1]*nx)
    cols[0].write(f"Y{iy}\n({iy*dy:.0f}m)")
    for ix in range(nx):
        et = st.session_state.equipment_map.get((ix, iy))
        lbl = EQUIPMENT_TYPES[et]["label"] if et in EQUIPMENT_TYPES else "·"
        if cols[ix+1].button(lbl, key=f"c_{ix}_{iy}", use_container_width=True):
            if tool == "Eraser": st.session_state.equipment_map.pop((ix, iy), None)
            else: st.session_state.equipment_map[(ix, iy)] = tool
            st.rerun()
