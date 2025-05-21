"""
Continuous simulation of a room with reinforcement learning control.
This version uses the original room dimensions:
  - Floor: 10 m x 5 m (50 m²)
  - Ceiling height: 2.4 m
and the original heat battery volume (0.1 m³) for improved stability.
The simulation runs multiple episodes (each = 1 day) and saves the Q-table
to disk ("qtable_improved.pkl") so that learning persists across script restarts.
Auto-tuning methods are included to boost exploration or reset hyperparameters if performance stalls.
The reward function has been updated to penalize high temperatures (above 30°C) and reward temperatures near 22°C.
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import os
import time

# For reproducibility
random.seed(0)
np.random.seed(0)

# ---------------------
# 0. Physical Constants and Environment Parameters
# ---------------------
# Define physical constants early so they are available globally.
c_luft = 1005         # Specific heat capacity of air [J/(kg*K)]
rho_luft = 1.2        # Air density [kg/m³]

# Original room dimensions
L_gulv = 10.0         # Room length [m]
B_gulv = 5.0          # Room width [m]
H_vegg = 2.4          # Room height [m]
A_gulv = L_gulv * B_gulv   # Floor area: 50 m²
A_tak = A_gulv             # Ceiling area: 50 m²

# Windows and doors
H_vindu = 1.0
B_vindu = 1.0
H_doer = 2.1
B_doer = 1.0
A_vindu = H_vindu * B_vindu
A_doer = H_doer * B_doer

# Exterior wall area (approximate)
A_yttervegg = 2 * L_gulv * H_vegg + 2 * B_gulv * H_vegg - A_doer - A_vindu

# Room volume
V_rom = L_gulv * B_gulv * H_vegg

# Transmission parameters
U_yttervegg = 0.22
U_tak = 0.18
U_gulv = 0.18
U_doer = 1.2
U_vindu = 1.2

UA_yttervegg = U_yttervegg * A_yttervegg
UA_tak = U_tak * A_tak
UA_gulv = U_gulv * A_gulv
UA_doer = U_doer * A_doer
UA_vindu = U_vindu * A_vindu

# Ventilation and infiltration
k_vent = 1.2        # [(m³/h)/m²]
k_luftlekk_50pa = 1.5  # [h⁻¹]

# Internal gains
P_arealspes_utstyrsvarme = 8
P_arealspes_belysningsvarme = 7
P_arealspes_personvarme = 60
A_person = 1.8

# Heat recovery
virkgrad_vg = 0.7

# Solar parameters
k_solfaktor = 0.26
P_sol_maks = 500

# Heat battery (original small volume)
V_vb = 0.1           # [m³]
td_vb = 5

# Simulation time settings
ts = 60                      # Time step [s]
t_stop_min = 1440            # 1440 minutes = 24 hours
N_sim = int((t_stop_min * 60) / ts)

# Delay arrays (for room and heat battery)
td_rom = 60
N_delay_rom = int(round(td_rom / ts)) + 1
delay_array_rom = np.zeros(N_delay_rom)
N_delay_vb = int(round(td_vb / ts)) + 1
delay_array_vb = np.zeros(N_delay_vb)

# PID parameters for the heat battery will be computed below after the agent definitions.

# ---------------------
# 1. RL Agent with Adjustable Hyperparameters and Auto-Tuning
# ---------------------

class RLAgent:
    def __init__(self,
                 actions,
                 alpha=0.1,
                 alpha_min=0.01,
                 alpha_decay=0.9999,  # slower decay
                 gamma=0.95,
                 epsilon=0.2,
                 epsilon_min=0.01,
                 epsilon_decay=0.9999):  # slower decay
        self.actions = actions          # Discrete actions in Watts
        self.alpha = alpha              # Learning rate
        self.alpha_min = alpha_min
        self.alpha_decay = alpha_decay
        self.gamma = gamma              # Discount factor
        self.epsilon = epsilon          # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = {}                     # Q-table
    
    def choose_action(self, state):
        if state not in self.Q:
            self.Q[state] = {a: 0.0 for a in self.actions}
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_vals = self.Q[state]
            max_val = max(q_vals.values())
            if np.isnan(max_val):
                return random.choice(self.actions)
            best_actions = [a for a, q in q_vals.items() if q == max_val]
            return random.choice(best_actions) if best_actions else random.choice(self.actions)
    
    def update(self, state, action, reward, next_state):
        if state not in self.Q:
            self.Q[state] = {a: 0.0 for a in self.actions}
        if next_state not in self.Q:
            self.Q[next_state] = {a: 0.0 for a in self.actions}
        q_current = self.Q[state][action]
        q_next_max = max(self.Q[next_state].values())
        self.Q[state][action] += self.alpha * (reward + self.gamma * q_next_max - q_current)
    
    def decay_hyperparams(self):
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def boost_epsilon(self, boost_value=0.1):
        self.epsilon = min(0.5, self.epsilon + boost_value)
        print(f"Boosted epsilon to {self.epsilon:.4f}")

def auto_tune_hyperparameters(rl_agent):
    # A simple placeholder: reset epsilon and alpha to higher values.
    old_epsilon = rl_agent.epsilon
    old_alpha = rl_agent.alpha
    rl_agent.epsilon = 0.2
    rl_agent.alpha = 0.1
    print(f"Auto-tuned hyperparameters: epsilon {old_epsilon:.4f} -> {rl_agent.epsilon:.4f}, alpha {old_alpha:.4f} -> {rl_agent.alpha:.4f}")

# ---------------------
# 2. State Discretization (Coarse Binning)
# ---------------------

def bin_temperature(T):
    T_clipped = max(-20, min(100, T))
    return 5 * int(round(T_clipped / 5.0))

def bin_occupancy(n):
    if n == 0:
        return 0
    elif 1 <= n <= 10:
        return 1
    elif 11 <= n <= 20:
        return 2
    else:
        return 3

def bin_hour(hour):
    return int(hour // 2)

def discretize_state(T, occupancy, t):
    hour = (t / 3600) % 24
    return (bin_temperature(T), bin_occupancy(occupancy), bin_hour(hour))

# ---------------------
# 3. Reward Function (with Temperature Penalties/Bonuses)
# ---------------------

def compute_reward(T_new, T_old, occupancy, u, u_prev, energy_avg, gamma=0.95, lambda_phi=0.1):
    """
    Computes a reward that strongly incentivizes maintaining the target temperature
    (22°C when occupied, 18°C when not) and provides a heavy bonus if the temperature is within ±1°C.
    Moderate penalties are applied for temperature deviations, control changes, energy usage, and for high temperatures.
    
    Parameters:
      T_old: previous temperature
      T_new: current temperature
      occupancy: current occupancy (0, 1, or 2)
      u: current control signal (heating power)
      u_prev: previous control signal
      energy_avg: rolling average of control usage
      gamma: discount factor for potential shaping
      lambda_phi: weight for the potential function
      
    Returns:
      reward: a scalar reward for the transition.
    """
    # Determine the target temperature.
    target = 22 if occupancy > 0 else 18
    
    # Calculate the absolute error.
    error = abs(T_new - target)
    
    # Heavy bonus for being within ±1°C of the target.
    bonus = 5000 if error <= 1 else 0
    
    # Moderate penalty for deviation from target.
    deviation_penalty = 100 * error  # 10 points penalty per degree off
    
    # Moderate penalty for abrupt changes in control.
    control_penalty = 1 * abs(u - u_prev)
    
    # Moderate penalty for energy usage.
    energy_penalty = 1 * u
    
    # Moderate penalty for high temperatures (if above 30°C).
    high_temp_penalty = 0
    if T_new > 30:
        high_temp_penalty = 20 * (T_new - 30)
    
    # Base reward combines the bonus and penalties.
    R_base = bonus - (deviation_penalty + control_penalty + energy_penalty + high_temp_penalty)
    
    # Potential-based shaping:
    phi_old = -lambda_phi * (T_old - target) ** 2
    phi_new = -lambda_phi * (T_new - target) ** 2
    shaping = gamma * phi_new - phi_old
    
    # Final reward.
    reward = R_base + shaping
    return reward





# ---------------------
# 4. Environment Derived Values and PID Parameters for the Heat Battery
# ---------------------
# Compute PID parameters for the heat battery based on the small room parameters.
tcc_vb = td_vb
Ki_vb = 1 / (c_luft * rho_luft * V_vb)
Kc_TC2 = 1 / (Ki_vb * (td_vb + tcc_vb))
Ti_TC2 = 2 * (td_vb + tcc_vb)
pid_TC2_par = (Kc_TC2, Ti_TC2, 0)
u_TC2_man = 0
u_TC2_max = 1000
u_TC2_min = 0
u_TC2_par = (u_TC2_max, u_TC2_min)

def fun_pid_controller(r_k, y_k, u_man, ui_km1, pid_par, u_par, ts):
    (Kc, Ti, _) = pid_TC2_par  # Use the computed PID parameters
    (u_max, u_min) = u_TC2_par
    e_k = r_k - y_k
    up_k = Kc * e_k
    ui_k = ui_km1 + (Kc * ts / Ti) * e_k
    ui_min = u_min - u_man - up_k
    ui_max = u_max - u_man - up_k
    ui_k = np.clip(ui_k, ui_min, ui_max)
    u_k = u_man + up_k + ui_k
    u_k = np.clip(u_k, u_min, u_max)
    return (u_k, ui_k)

# ---------------------
# 5. Single Episode Simulation with Persistent Occupancy Dynamics
# ---------------------

def run_episode(episode, rl_agent):
    # Reset delay arrays
    delay_array_rom[:] = 0
    delay_array_vb[:] = 0
    
    T_rom_k = 18.0
    T_vg_prim_ut_init = 10 + virkgrad_vg * (22 - 10)
    T_vb_ut_k = T_vg_prim_ut_init
    ui_TC2_km1 = 0
    occupancy_current = 0  # Persistent occupancy
    
    # Initialize baseline occupancy at 7 AM
    T_rom_prev = T_rom_k
    u_prev = 0
    u_history = []
    
    t_array = np.zeros(N_sim)
    T_rom_array = np.zeros(N_sim)
    T_ute_array = np.zeros(N_sim)
    u_TC1_array = np.zeros(N_sim)
    u_TC2_array = np.zeros(N_sim)
    occupancy_array = np.zeros(N_sim)
    reward_array = np.zeros(N_sim)
    
    for k in range(N_sim):
        T_rom_k = np.clip(T_rom_k, -20, 100)
        t_k = k * ts
        t_array[k] = t_k
        hour = (t_k / 3600) % 24
        
        T_ute_k = 10 + 5 * np.sin(2*np.pi*(hour - 6)/24)
        T_ute_array[k] = T_ute_k
        
        # Persistent occupancy update once per hour:
        if int(t_k) % 3600 == 0:
            if 7 <= hour < 18:
                if int(hour) == 7:
                    occupancy_current = np.random.randint(10, 15)
                else:
                    delta = np.random.randint(-1, 1)
                    occupancy_current = np.clip(occupancy_current + delta, 0, 30)
            else:
                occupancy_current = 0
        n_personer_k = occupancy_current
        occupancy_array[k] = n_personer_k
        
        r_TC1_k = 22
        r_TC2_k = 19
        
        P_transm = (UA_yttervegg * (T_ute_k - T_rom_k) +
                    UA_tak * (T_ute_k - T_rom_k) +
                    UA_gulv * (T_ute_k - T_rom_k) +
                    UA_doer * (T_ute_k - T_rom_k) +
                    UA_vindu * (T_ute_k - T_rom_k))
        
        F_infilt = k_luftlekk_50pa * V_rom / 3600
        P_infilt = c_luft * rho_luft * F_infilt * (T_ute_k - T_rom_k)
        
        if 10 <= hour <= 16:
            solar_factor = np.cos((hour - 13) * np.pi/6)
            P_sol_vert = P_sol_maks * max(solar_factor, 0)
        else:
            P_sol_vert = 0
        P_solvarme = P_sol_vert * A_vindu * k_solfaktor
        
        if 7 <= hour < 18:
            belysning_on_off = 1
            utstyr_on_off = 1
        else:
            belysning_on_off = 0
            utstyr_on_off = 0
        P_utstyr = utstyr_on_off * P_arealspes_utstyrsvarme * A_gulv
        P_belysning = belysning_on_off * P_arealspes_belysningsvarme * A_gulv
        P_person = P_arealspes_personvarme * A_person * n_personer_k
        
        F_vent = (k_vent / 3600) * A_gulv
        T_tilluft_k = T_vb_ut_k
        P_vent = c_luft * rho_luft * F_vent * (T_tilluft_k - T_rom_k)
        
        T_vg_sek_inn = T_rom_k
        T_vg_prim_inn = T_ute_k
        T_vg_prim_ut = T_vg_prim_inn + virkgrad_vg * (T_vg_sek_inn - T_vg_prim_inn)
        
        (u_TC2_k, ui_TC2_k) = fun_pid_controller(r_TC2_k, T_tilluft_k, 0, ui_TC2_km1,
                                                  pid_TC2_par, u_TC2_par, ts)
        u_TC2_array[k] = u_TC2_k
        
        P_vb_delayed = delay_array_vb[-1]
        delay_array_vb[1:] = delay_array_vb[0:-1]
        delay_array_vb[0] = u_TC2_k
        dT_vb = (P_vb_delayed + c_luft * rho_luft * F_vent * (T_vg_prim_ut - T_vb_ut_k)) / (c_luft * rho_luft * V_vb)
        T_vb_ut_k = np.clip(T_vb_ut_k + ts * dT_vb, -20, 100)
        ui_TC2_km1 = ui_TC2_k
        
        state = discretize_state(T_rom_k, n_personer_k, t_k)
        action = rl_agent.choose_action(state)
        u_TC1_k = action
        u_TC1_array[k] = u_TC1_k
        
        u_history.append(u_TC1_k)
        if len(u_history) > 60:
            u_history.pop(0)
        energy_avg = np.mean(u_history) if u_history else 0
        
        P_rom_delayed = delay_array_rom[-1]
        delay_array_rom[1:] = delay_array_rom[0:-1]
        delay_array_rom[0] = u_TC1_k
        
        dT_rom = (P_rom_delayed + P_vent + P_transm + P_infilt +
                  P_solvarme + P_utstyr + P_belysning + P_person) / (c_luft * rho_luft * V_rom)
        T_rom_kp1 = np.clip(T_rom_k + ts * dT_rom, -20, 100)
        
        reward = compute_reward(T_rom_kp1, T_rom_prev, n_personer_k, u_TC1_k, u_prev, energy_avg)
        reward_array[k] = reward
        
        next_state = discretize_state(T_rom_kp1, n_personer_k, t_k + ts)
        rl_agent.update(state, action, reward, next_state)
        
        T_rom_prev = T_rom_k
        T_rom_k = T_rom_kp1
        u_prev = u_TC1_k
        T_rom_array[k] = T_rom_k
    
    cumulative_reward = np.sum(reward_array)
    return t_array, T_rom_array, u_TC1_array, u_TC2_array, occupancy_array, T_ute_array, cumulative_reward

# ---------------------
# 6. Continuous Loop with Auto-Tuning of Hyperparameters and Persistent Q-table
# ---------------------

# Power the model can pump into the system 
actions = [0, 2000]#, 5000, 8000, 10000]
qtable_filename = "hvacRL1.pkl"

if os.path.exists(qtable_filename):
    with open(qtable_filename, "rb") as f:
        saved_Q = pickle.load(f)
    rl_agent = RLAgent(actions, alpha=0.1, alpha_decay=0.9999,
                       gamma=0.95, epsilon=0.2, epsilon_decay=0.9999)
    rl_agent.Q = saved_Q
    print(f"Loaded existing Q-table from {qtable_filename}")
else:
    rl_agent = RLAgent(actions, alpha=0.1, alpha_decay=0.9999,
                       gamma=0.95, epsilon=0.2, epsilon_decay=0.9999)
    print("Initialized new Q-table.")

fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
line_temp, = axs[0].plot([], [], label='Room Temperature [°C]')
axs[0].set_ylabel('Temperature [°C]')
axs[0].grid(); axs[0].legend()

line_u, = axs[1].plot([], [], label='RL Heating [W]')
line_pid, = axs[1].plot([], [], label='PID Battery [W]')
axs[1].set_ylabel('Control Signals [W]')
axs[1].grid(); axs[1].legend()

line_occ, = axs[2].plot([], [], label='Occupancy [persons]')
line_out, = axs[2].plot([], [], label='Outdoor Temp [°C]')
axs[2].set_xlabel('Time [h]')
axs[2].set_ylabel('Occupancy / Outdoor temperature')
axs[2].grid(); axs[2].legend()

plt.tight_layout()

episode = 0
best_reward = -float('inf')
stagnation_counter = 0  # Counts episodes with no improvement

try:
    while True:
        print(f"Starting episode {episode}...")
        t_arr, T_rom_arr, u_TC1_arr, u_TC2_arr, occ_arr, T_ute_arr, cum_reward = run_episode(episode, rl_agent)
        
        rl_agent.decay_hyperparams()
        
        with open(qtable_filename, "wb") as f:
            pickle.dump(rl_agent.Q, f)
        
        print(f"Episode {episode} cumulative reward: {cum_reward:.2f}, alpha={rl_agent.alpha:.4f}, epsilon={rl_agent.epsilon:.4f}")
        
        if cum_reward > best_reward:
            best_reward = cum_reward
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            if stagnation_counter >= 50:
                rl_agent.boost_epsilon(0.1)
            if stagnation_counter >= 100:
                auto_tune_hyperparameters(rl_agent)
                stagnation_counter = 0
        
        t_hours = t_arr / 3600.0
        line_temp.set_data(t_hours, T_rom_arr)
        line_u.set_data(t_hours, u_TC1_arr)
        line_pid.set_data(t_hours, u_TC2_arr)
        line_occ.set_data(t_hours, occ_arr)
        line_out.set_data(t_hours, T_ute_arr)
        
        for ax in axs:
            ax.relim()
            ax.autoscale_view()
        plt.draw()
        plt.pause(0.1)
        
        episode += 1

except KeyboardInterrupt:
    print("Simulation stopped by user.")
