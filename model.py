# =====================================================================
# 📂 FILE: model.py
# 📝 PURPOSE: Advanced Dual-Mode RL MDP Engine & Custom Load Sourcing
# =====================================================================

import numpy as np
import pandas as pd
import random

class DataDrivenSmartHomeEnv:
    """
    Empirical Reinforcement Learning Environment representing a Markov Decision Process (MDP).
    Supports both Dataset-Informed Ingestion and Manual Sandbox Profile Synthesis.
    """
    def __init__(self, df, home_id, season, household_size, solar_efficiency, 
                 mode="Dataset", custom_hours=None, battery_enabled=True):
        self.hours = 24
        self.solar_efficiency = solar_efficiency
        self.mode = mode
        self.custom_hours = custom_hours if custom_hours else {}
        self.battery_enabled = battery_enabled  # True = Hybrid, False = Government Grid Only
        
        # Ingestion pipeline filter matrix
        self.filtered_df = df[
            (df['Home ID'] == home_id) & 
            (df['Season'] == season) & 
            (df['Household Size'] == household_size)
        ]
        if self.filtered_df.empty and df is not None:
            self.filtered_df = df[df['Season'] == season]
            
        self.reset()

    def get_appliance_load_matrix(self, hour):
        """
        Determines the load matrix profile based on execution mode.
        """
        # --- MODE A: CUSTOM SANDBOX HOUSE (User Configured) ---
        if self.mode == "Sandbox":
            load_map = {
                'Fan 1': 0.08 if self.custom_hours.get('fan1_start', 0) <= hour <= self.custom_hours.get('fan1_end', 24) else 0.0,
                'Fan 2': 0.08 if self.custom_hours.get('fan2_start', 0) <= hour <= self.custom_hours.get('fan2_end', 24) else 0.0,
                'AC': 1.50 if self.custom_hours.get('ac_start', 0) <= hour <= self.custom_hours.get('ac_end', 24) else 0.0,
                'TV': 0.12 if self.custom_hours.get('tv_start', 0) <= hour <= self.custom_hours.get('tv_end', 24) else 0.0,
                'Fridge': 0.20,  # Constant essential baseline
                'Washing Machine': 0.50 if self.custom_hours.get('washer_start', 0) <= hour <= self.custom_hours.get('washer_end', 24) else 0.0,
                'Water Motor': 1.00 if self.custom_hours.get('motor_start', 0) <= hour <= self.custom_hours.get('motor_end', 24) else 0.0,
                'Lights & Other': 0.15 if 18 <= hour <= 23 else 0.03
            }
            return load_map

        # --- MODE B: DATASET HOUSE (Historical Load) ---
        hourly_slice = self.filtered_df[self.filtered_df['Hour'] == hour]
        appliances = ['Fridge', 'Heater', 'AC', 'Microwave', 'Fan', 'Lights', 'Oven', 'Washer']
        load_map = {appliance: 0.0 for appliance in appliances}
        
        if hourly_slice.empty:
            return {'Fridge': 0.15, 'Heater': 0.0, 'AC': 0.40, 'Microwave': 0.0, 'Fan': 0.10, 'Lights': 0.05, 'Oven': 0.0, 'Washer': 0.0}
            
        for appliance in appliances:
            app_data = hourly_slice[hourly_slice['Appliance Type'] == appliance]
            if not app_data.empty:
                load_map[appliance] = float(app_data['Energy Consumption (kWh)'].sum())
                
        return load_map

    def reset(self):
        self.hour = 0
        self.battery = 50.0 if self.battery_enabled else 0.0
        self.total_cost = 0.0
        return (self.hour, int(self.battery // 20))

    def step(self, action):
        # Time-of-Use pricing framework (Peak vs Off-Peak Grid Cost Matrix)
        price_per_kwh = 0.85 if (10 <= self.hour <= 18) else 0.35
        
        appliance_loads = self.get_appliance_load_matrix(self.hour)
        total_demand = sum(appliance_loads.values())
        
        # Solar Storage Management Processing Loop
        if self.battery_enabled:
            solar_generation = self.solar_efficiency if (11 <= self.hour <= 16) else 0.0
            self.battery = min(100.0, self.battery + (solar_generation * 12.5))
        else:
            self.battery = 0.0  # Kept flat locked if on pure government grid infrastructure

        reward = 0.0
        
        # Override action choice if battery system is physically absent
        if not self.battery_enabled and action == 1:
            action = 0  # Force revert straight to grid draw
            
        # Decision Matrix Tree Logic
        if action == 0:  # Pure Grid Sourcing
            step_cost = total_demand * price_per_kwh
            reward = -step_cost
            self.total_cost += step_cost
            
        elif action == 1:  # Battery Discharge Allocation
            if self.battery > 20.0:
                self.battery -= 20.0
                reward = 5.0
                step_cost = 0.0
            else:
                step_cost = total_demand * price_per_kwh
                reward = -step_cost - 15.0  
                self.total_cost += step_cost
                
        elif action == 2:  # Controlled Eco-Mode Optimization
            optimized_demand = total_demand * 0.70
            step_cost = optimized_demand * price_per_kwh
            reward = -step_cost + 2.0
            self.total_cost += step_cost

        self.hour += 1
        done = self.hour >= 24
        
        next_state_bucket = (self.hour, int(self.battery // 20))
        return next_state_bucket, reward, done


class QLearningAgent:
    def __init__(self, learning_rate=0.15, discount_factor=0.95, exploration_rate=0.20):
        self.q_table = np.zeros((26, 6, 3))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.eps = exploration_rate

    def choose_action(self, state):
        if random.random() < self.eps:
            return random.randint(0, 2)
        return np.argmax(self.q_table[state[0], state[1]])

    def learn(self, state, action, reward, next_state, done):
        current_prediction = self.q_table[state[0], state[1], action]
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state[0], next_state[1]])
        self.q_table[state[0], state[1], action] += self.lr * (target - current_prediction)