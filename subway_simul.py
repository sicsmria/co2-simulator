# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("Room Settings")

room_w = st.sidebar.slider("Room Width (m)", min_value=2.0, max_value=5000.0, value=10.0, step=1.0)
room_d = st.sidebar.slider("Room Depth (m)", min_value=2.0, max_value=1000.0, value=5.0, step=1.0)
room_h = st.sidebar.slider("Room Height (m)", min_value=2.0, max_value=6.0, value=3.0, step=0.1)

ach = st.sidebar.slider("Ventilation ACH (1/h)", min_value=0.1, max_value=20.0, value=3.0, step=0.1)

st.sidebar.header("Population")
n_standing = st.sidebar.number_input("Standing people", min_value=0, max_value=3000, value=5, step=1)
n_sitting = st.sidebar.number_input("Sitting people", min_value=0, max_value=3000, value=3, step=1)
n_lying = st.sidebar.number_input("Lying people", min_value=0, max_value=3000, value=1, step=1)

st.sidebar.header("Equipment Editor")
selected_tool = st.sidebar.radio(
    "Equipment Tool",
    ["Supply", "Exhaust", "Purifier", "Eraser"],
    index=0,
)

editor_step = st.sidebar.select_slider(
    "Cell Size (m)",
    options=[0.5, 1.0, 1.5, 2.0],
    value=DEFAULT_EDITOR_CELL_M,
)

btn_col1, btn_col2 = st.sidebar.columns(2)

with btn_col1:
    if st.button("Randomize Population"):
        st.session_state.layout_seed += 1

with btn_col2:
    if st.button("Clear Equipments"):
        st.session_state.equipment_map = {}

trim_equipment_map(room_w, room_d, editor_step)
