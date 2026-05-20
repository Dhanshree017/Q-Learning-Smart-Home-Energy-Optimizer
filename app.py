# =====================================================================
# 📂 FILE: app.py
# 📝 PURPOSE: Frontend Framework supporting Dataset Analysis & Sandbox Homes
# =====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from model import DataDrivenSmartHomeEnv, QLearningAgent

st.set_page_config(page_title="RL Energy Matrix", layout="wide")

# Custom UI Pale Yellow Styling Blocks Injector
st.markdown(
    """
    <style>
    .stApp { background-color: #FEFBF0; }
    .custom-card {
        padding: 15px; border-radius: 12px; background-color: #FFFFFF;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03); text-align: center;
        margin-bottom: 15px; border: 1px solid #EFEBE0;
    }
    .card-title { font-size: 13px; color: #6E6A5F; text-transform: uppercase; font-weight: bold; }
    .card-val { font-size: 20px; color: #1A1A1A; font-weight: 800; margin-top: 5px; }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def ingest_production_dataset():
    try:
        df = pd.read_csv("smart_home_energy_consumption_large.csv")
        if 'Time' in df.columns:
            df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
            if df['Hour'].isnull().all():
                df['Hour'] = pd.to_numeric(df['Time'], errors='coerce').astype(int) % 24
    except Exception:
        np.random.seed(42)
        records = 100000
        appliances = ['Fridge', 'Heater', 'AC', 'Microwave', 'Fan', 'Lights', 'Oven', 'Washer']
        df = pd.DataFrame({
            'Home ID': np.random.randint(1001, 1040, size=records),
            'Appliance Type': np.random.choice(appliances, size=records),
            'Energy Consumption (kWh)': np.random.exponential(scale=0.45, size=records) + 0.02,
            'Hour': np.random.randint(0, 24, size=records),
            'Outdoor Temperature (°C)': np.random.normal(loc=18, scale=9, size=records),
            'Season': np.random.choice(['Winter', 'Summer', 'Spring', 'Fall'], size=records),
            'Household Size': np.random.randint(1, 6, size=records)
        })
    return df

raw_data = ingest_production_dataset()

st.title("🔋 Smart Home RL Energy Orchestration System")
st.write("Train value-based Q-learning models against preset dataset tracks or map your personal home setup explicitly.")

# --- MODE SELECTOR TABS ---
mode_choice = st.radio("⚡ Select Application Interface Mode:", ["📋 Analyze Dataset Records", "🛠️ Simulate My Custom House Sandbox"], horizontal=True)

custom_hours = {}
power_infrastructure = "Government Grid Only"
solar_kw = 0.0
selected_home, selected_season, selected_size = 1001, "Summer", 4

if mode_choice == "📋 Analyze Dataset Records":
    st.sidebar.header("🕹️ Dataset Matrix Parameters")
    available_homes = sorted(raw_data['Home ID'].unique())
    selected_home = st.sidebar.selectbox("🎯 Target Household ID", available_homes)
    selected_season = st.sidebar.radio("☀️ Active Climate Season", ['Winter', 'Summer', 'Spring', 'Fall'], index=1)
    selected_size = st.sidebar.slider("👥 Family Size (Occupants)", 1, 6, 4)
    st.sidebar.markdown("---")
    power_infrastructure = st.sidebar.selectbox("🔌 Infrastructure Backup Strategy", ["Hybrid (Grid + Battery Storage)", "Government Grid Only"])
    solar_kw = st.sidebar.select_slider("☀️ Solar Array Generation Capability (kW)", options=[0.5, 1.0, 1.5, 2.0], value=1.5) if "Hybrid" in power_infrastructure else 0.0

else:
    # --- SANDBOX OPERATION CONTROLS ---
    st.sidebar.header("🏢 Home Hardware Matrix")
    power_infrastructure = st.sidebar.radio("🔌 Primary Power Sourcing Mode", ["Government Grid Only", "Hybrid (Grid + Battery Backup)"])
    solar_kw = st.sidebar.select_slider("☀️ Connected Solar Capacity Yield (kW)", options=[0.5, 1.0, 1.5, 2.0], value=1.0) if "Hybrid" in power_infrastructure else 0.0
    selected_season = st.sidebar.selectbox("🍂 Simulated Season Frame", ["Summer", "Winter", "Spring", "Fall"])
    
    st.subheader("⚙️ Configure Daily Appliance Timelines (24H Format)")
    st.write("Slide intervals to denote the exact hours your home systems run actively:")
    
    col_a, col_b = st.columns(2)
    with col_a:
        with st.expander("🌀 Cooling Frameworks (AC & Fans)"):
            ac_s, ac_e = st.slider("❄️ Air Conditioner Active Hours", 0, 24, (12, 18))
            f1_s, f1_e = st.slider("🍃 Primary Ceiling Fan (Fan 1) Hours", 0, 24, (0, 24))
            f2_s, f2_e = st.slider("🍃 Secondary Bedroom Fan (Fan 2) Hours", 0, 24, (22, 7))
        with st.expander("📺 Entertainment & Kitchen Assets"):
            tv_s, tv_e = st.slider("🖥️ Living Room TV Usage Hours", 0, 24, (19, 23))
            st.caption("🥦 Note: Refrigerator load is processed internally as an un-sheddable 24-hour constant link.")
            
    with col_b:
        with st.expander("🧺 Heavy Mechanical Frameworks"):
            motor_s, motor_e = st.slider("🚰 Water Pump Motor Running Hours", 0, 24, (6, 8))
            washer_s, washer_e = st.slider("🧼 Clothes Washing Machine Hours", 0, 24, (9, 11))

    custom_hours = {
        'ac_start': ac_s, 'ac_end': ac_e, 'fan1_start': f1_s, 'fan1_end': f1_e,
        'fan2_start': f2_s, 'fan2_end': f2_e, 'tv_start': tv_s, 'tv_end': tv_e,
        'motor_start': motor_s, 'motor_end': motor_e, 'washer_start': washer_s, 'washer_end': washer_e
    }

total_episodes = st.sidebar.number_input("🤖 RL Optimization Iterations", 100, 5000, 1500, step=100)

# --- RENDERING TOP CARDS BASED ON SELECTED SEASONS AND MODE ---
m1, m2, m3, m4 = st.columns(4)
season_emoji = {"Winter": "❄️ Winter Block", "Summer": "☀️ Summer Block", "Spring": "🌱 Spring Block", "Fall": "🍂 Fall Block"}.get(selected_season, "🏡")
infra_label = "🏛️ Pure Govt Grid" if power_infrastructure == "Government Grid Only" else "🔋 Hybrid Storage"

with m1:
    st.markdown(f'<div class="custom-card"><div class="card-title">Operating Mode</div><div class="card-val">{"📊 Data Mode" if mode_choice == "📋 Analyze Dataset Records" else "🛠️ Sandbox"}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="custom-card"><div class="card-title">Climate Block</div><div class="card-val">{season_emoji}</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="custom-card"><div class="card-title">Grid Architecture</div><div class="card-val">{infra_label}</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="custom-card"><div class="card-title">Solar Array Size</div><div class="card-val">⚡ {solar_kw} kW</div></div>', unsafe_allow_html=True)

# --- EXECUTION ENGINE CONTROLLER ---
if st.button("🚀 Execute Q-Learning Policy Engine"):
    
    is_battery = (power_infrastructure != "Government Grid Only")
    env_mode = "Sandbox" if mode_choice == "🛠️ Simulate My Custom House Sandbox" else "Dataset"
    
    env = DataDrivenSmartHomeEnv(
        df=raw_data, home_id=selected_home, season=selected_season, 
        household_size=selected_size, solar_efficiency=solar_kw,
        mode=env_mode, custom_hours=custom_hours, battery_enabled=is_battery
    )
    agent = QLearningAgent()
    cost_convergence = []
    
    with st.status("🧠 Processing Bellman Optimization Loop...", expanded=True) as status_box:
        progress_bar = st.progress(0)
        for episode in range(total_episodes):
            state = env.reset()
            done = False
            while not done:
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                agent.learn(state, action, reward, next_state, done)
                state = next_state
            cost_convergence.append(env.total_cost)
            if episode % max(1, (total_episodes // 10)) == 0:
                progress_bar.progress(episode / total_episodes)
                
        progress_bar.progress(1.0)
        status_box.update(label="RL Optimization Cycle Finalized!", state="complete", expanded=False)

    # --- RENDER RESULTS VISUALIZATIONS ---
    st.divider()
    l_col, r_col = st.columns([3, 2])
    
    with l_col:
        st.subheader("📉 Policy Value Convergence Graph")
        curve_df = pd.DataFrame({"Total Daily Cost ($)": cost_convergence}).reset_index().rename(columns={"index": "Episode"})
        fig_curve = px.line(curve_df, x="Episode", y="Total Daily Cost ($)", title="Cost Index Reduction Matrix", color_discrete_sequence=["#16A34A"])
        fig_curve.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_curve, use_container_width=True)
        
    with r_col:
        st.subheader("⚡ Calculated 24-Hour Active Load Graph")
        hourly_demands = [sum(env.get_appliance_load_matrix(h).values()) for h in range(24)]
        load_df = pd.DataFrame({"Hour": [f"{h:02d}:00" for h in range(24)], "Load demand (kWh)": hourly_demands})
        fig_load = px.bar(load_df, x="Hour", y="Load demand (kWh)", title="Calculated Hourly Electricity Envelope", color_discrete_sequence=["#0284C7"])
        fig_load.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_load, use_container_width=True)

    # --- 24-HOUR OPTIMAL TRACE SYSTEM ---
    st.subheader("🎯 Optimal Action Decisions Schedule Matrix")
    env.reset()
    test_state = (0, int(env.battery // 20))
    schedule_records = []
    
    for h in range(24):
        optimal_action = np.argmax(agent.q_table[test_state[0], test_state[1]])
        action_string = ["🔌 Sourced Grid", "🔋 Battery Draw", "🌱 Eco Optimization"][optimal_action]
        
        # If the user's setup lacks a battery, clear out the text trace to reflect grid sourcing
        if not is_battery:
            action_string = "🔌 Sourced Grid" if optimal_action != 2 else "🌱 Eco Optimization"
            
        hardware_loads = env.get_appliance_load_matrix(h)
        total_h_load = sum(hardware_loads.values())
        
        schedule_records.append({
            "Hour Timestamp": f"{h:02d}:00",
            "Total Load Sensed": f"{total_h_load:.2f} kWh",
            "Agent Control Strategy Choice": action_string,
            "Battery Buffer Metric": f"{int(env.battery)}%" if is_battery else "❌ No Battery Installed (Pure Grid Home)"
        })
        test_state, _, _ = env.step(optimal_action)
        
    st.dataframe(pd.DataFrame(schedule_records), use_container_width=True, hide_index=True)