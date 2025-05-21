"""
DQN-based HVAC Control Simulation with Auto-Scaled Plots

This script uses a Deep Q-Network (DQN) agent to control the HVAC system
for a room with the original small dimensions:
  - Floor: 10 m x 5 m (50 m²)
  - Ceiling height: 2.4 m
  - Heat battery volume: 0.1 m³
The state is a 3-element vector (temperature, occupancy, hour), normalized.
Actions are discrete heating power levels: [0, 2000, 5000, 8000, 10000] Watts.
A replay buffer is used to store experiences, and the network is trained periodically.
The reward function uses potential-based shaping plus additional bonuses.
Plots are updated interactively:
  - The top subplot shows the temperature profile over a 24-hour period with x-axis ticks from 0 to 24.
  - The bottom subplot shows cumulative reward over episodes.
The y-axes auto-scale based on the data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import pickle
import matplotlib.pyplot as plt

# For reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# ---------------------
# 0. Environment Constants (Original Room)
# ---------------------
c_luft = 1005         # J/(kg*K)
rho_luft = 1.2        # kg/m³

L_gulv = 10.0         # Room length [m]
B_gulv = 5.0          # Room width [m]
H_vegg = 2.4          # Room height [m]
A_gulv = L_gulv * B_gulv   # 50 m² floor area
A_tak = A_gulv             # 50 m² ceiling area

H_vindu = 1.0
B_vindu = 1.0
H_doer = 2.1
B_doer = 1.0
A_vindu = H_vindu * B_vindu
A_doer = H_doer * B_doer
A_yttervegg = 2 * L_gulv * H_vegg + 2 * B_gulv * H_vegg - A_doer - A_vindu
V_rom = L_gulv * B_gulv * H_vegg

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

# Ventilation parameters (reduced to keep ventilation power low)
k_vent = 0.5       # [(m³/h)/m²]
k_luftlekk_50pa = 1.5

P_arealspes_utstyrsvarme = 8
P_arealspes_belysningsvarme = 7
P_arealspes_personvarme = 60
A_person = 1.8

virkgrad_vg = 0.7

k_solfaktor = 0.26
P_sol_maks = 500

V_vb = 0.1         # Original heat battery volume
td_vb = 5

ts = 60            # Timestep: 60 seconds
t_stop_min = 1440  # 24 hours in minutes
N_sim = int((t_stop_min * 60) / ts)

td_rom = 60
N_delay_rom = int(round(td_rom / ts)) + 1
delay_array_rom = np.zeros(N_delay_rom)
N_delay_vb = int(round(td_vb / ts)) + 1
delay_array_vb = np.zeros(N_delay_vb)

# ---------------------
# 1. State Representation Helpers
# ---------------------
def discretize_state(T, occupancy, t):
    """
    Returns a tuple: (temp_bin, occupancy_bin, hour_bin)
    - temp_bin: rounded temperature (in steps of 5°C)
    - occupancy_bin: bucket occupancy (0,1,2,3)
    - hour_bin: floor(hour/2)
    """
    T_clipped = max(-20, min(100, T))
    temp_bin = 5 * int(round(T_clipped / 5.0))
    if occupancy == 0:
        occ_bin = 0
    elif 1 <= occupancy <= 10:
        occ_bin = 1
    elif 11 <= occupancy <= 20:
        occ_bin = 2
    else:
        occ_bin = 3
    hour = (t / 3600) % 24
    hour_bin = int(hour // 2)
    return (temp_bin, occ_bin, hour_bin)

def state_to_vector(state_tuple):
    """
    Converts a discrete state tuple into a normalized vector.
    Normalization:
      - Temperature divided by 100 (approx. range: -0.2 to 1.0)
      - Occupancy divided by 3 (range 0 to 1)
      - Hour divided by 11 (range 0 to 1)
    """
    temp, occ, hour = state_tuple
    return np.array([temp / 100.0, occ / 3.0, hour / 11.0], dtype=np.float32)

# ---------------------
# 2. DQN Network and Replay Buffer
# ---------------------
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, buffer_size=10000):
        self.memory = []
        self.buffer_size = buffer_size
    def add(self, experience):
        self.memory.append(experience)
        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

# ---------------------
# 3. DQN Agent
# ---------------------
class DQNAgent:
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, update_every=5):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_every = update_every
        self.t_step = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qnetwork_local = QNetwork(state_size, action_size).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) >= self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
    
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            return torch.argmax(action_values).item()
    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau=0.01)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

# ---------------------
# 4. Reward Function (Potential-based Shaping with Bonuses)
# ---------------------
def compute_reward(T_new, T_old, occupancy, u, u_prev, energy_avg, gamma=0.95, lambda_phi=0.1):
    target = 22 if occupancy > 0 else 18
    energy_cost = 1 * u
    smoothness_penalty = 1 * abs(u - u_prev)
    avg_penalty = 0.5 if energy_avg > 10000 else 0.0
    high_temp_penalty = 0.0
    if T_new > 30:
        high_temp_penalty = 2 * (T_new - 30) ** 2
    elif T_new > 40:
        high_temp_penalty = 10 * (T_new - 30) ** 2
    R_base = - (energy_cost + smoothness_penalty + avg_penalty + high_temp_penalty)
    
    phi_old = -lambda_phi * (T_old - target) ** 2
    phi_new = -lambda_phi * (T_new - target) ** 2
    shaping = gamma * phi_new - phi_old
    
    bonus = 100000 if abs(T_new - target) <= 1 else 0.0
    improvement_bonus = max(0, abs(T_old - target) - abs(T_new - target)) * 100
    efficiency_bonus = (max(0, 1000 - u) * 0.05) if abs(T_new - target) <= 1 else 0.0
    
    reward = R_base + shaping + bonus + improvement_bonus + efficiency_bonus
    return reward

# ---------------------
# 5. Simulation Environment (Run One Episode for DQN)
# ---------------------
def run_episode_dqn(agent):
    delay_array_rom[:] = 0
    delay_array_vb[:] = 0
    T_rom = 22.0
    T_vg_prim_ut_init = 10 + virkgrad_vg * (22 - 10)
    T_vb_ut = T_vg_prim_ut_init
    ui_TC2_prev = 0
    occupancy_current = 0
    T_rom_prev = T_rom
    u_prev = 0
    u_history = []
    
    t_array = np.zeros(N_sim)
    T_rom_array = np.zeros(N_sim)
    u_TC1_array = np.zeros(N_sim)
    occupancy_array = np.zeros(N_sim)
    cum_reward = 0.0
    
    for k in range(N_sim):
        T_rom = np.clip(T_rom, -20, 100)
        t = k * ts
        t_array[k] = t
        hour = (t / 3600) % 24
        
        # Outdoor temperature: sinusoidal pattern
        T_out = 10 + 5 * np.sin(2 * np.pi * (hour - 6) / 24)
        
        # Persistent occupancy update once per hour
        if int(t) % 3600 == 0:
            if 7 <= hour < 18:
                if int(hour) == 7:
                    occupancy_current = np.random.randint(1, 3)  # 1 or 2 people
                else:
                    delta = np.random.randint(-1, 2)
                    occupancy_current = np.clip(occupancy_current + delta, 0, 2)
            else:
                occupancy_current = 0
        occupancy_array[k] = occupancy_current
        
        # For DQN, we use the state vector
        state_tuple = discretize_state(T_rom, occupancy_current, t)
        state_vec = state_to_vector(state_tuple)
        
        action_index = agent.act(state_vec)
        actions_list = [0, 2000, 5000, 8000, 10000]
        u = actions_list[action_index]
        
        # Simplified dynamics: temperature update
        T_rom_new = T_rom + 0.01 * u + 0.005 * (T_out - T_rom)
        
        u_history.append(u)
        if len(u_history) > 60:
            u_history.pop(0)
        energy_avg = np.mean(u_history)
        
        r = compute_reward(T_rom_new, T_rom, occupancy_current, u, u_prev, energy_avg)
        cum_reward += r
        
        next_state_tuple = discretize_state(T_rom_new, occupancy_current, t + ts)
        next_state_vec = state_to_vector(next_state_tuple)
        done = (k == N_sim - 1)
        
        agent.step(state_vec, action_index, r, next_state_vec, done)
        
        T_rom_prev = T_rom
        T_rom = T_rom_new
        u_prev = u
        T_rom_array[k] = T_rom
    
    return t_array, T_rom_array, cum_reward

# ---------------------
# 6. Main Training Loop and Plotting
# ---------------------
state_size = 3
action_size = 5
actions_list = [0, 2000, 5000, 8000, 10000]

agent = DQNAgent(state_size, action_size, lr=1e-3, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, buffer_size=10000, batch_size=64)

qtable_filename = "dqn_model.pkl"
if os.path.exists(qtable_filename):
    with open(qtable_filename, "rb") as f:
        state_dict = torch.load(f, weights_only=False)
    agent.qnetwork_local.load_state_dict(state_dict)
    agent.qnetwork_target.load_state_dict(state_dict)
    print("Loaded saved DQN model.")


# Set up interactive plotting with two subplots: Temperature profile and Reward history
plt.ion()
fig, axs = plt.subplots(2, 1, figsize=(12, 10))
temp_ax = axs[0]    # Temperature profile for current episode
reward_ax = axs[1]  # Cumulative reward over episodes

# Configure temperature plot: fixed x-axis (0 to 24 h), auto y-axis
temp_ax.set_xlabel("Time [h]")
temp_ax.set_ylabel("Room Temperature [°C]")
temp_ax.set_xticks(np.arange(0, 25, 1))
temp_ax.set_xlim(0, 24)
temp_ax.grid()
temp_ax.legend()

# Configure reward plot: auto scaling on both axes
reward_ax.set_xlabel("Episode")
reward_ax.set_ylabel("Cumulative Reward")
reward_ax.grid()
reward_ax.legend()

rewards_history = []
episodes = 1000

for ep in range(episodes):
    t_arr, T_rom_arr, cum_reward = run_episode_dqn(agent)
    rewards_history.append(cum_reward)
    
    # Update Temperature Plot:
    t_hours = t_arr / 3600.0  # Convert seconds to hours
    temp_ax.clear()
    temp_ax.plot(t_hours, T_rom_arr, label="Room Temp [°C]", color="tab:blue")
    temp_ax.set_xlabel("Time [h]")
    temp_ax.set_ylabel("Room Temperature [°C]")
    temp_ax.set_xticks(np.arange(0, 25, 1))
    temp_ax.set_xlim(0, 24)
    # No fixed y-axis limits are set; y-axis auto-scales.
    temp_ax.legend()
    temp_ax.grid()
    
    # Update Reward Plot:
    reward_ax.clear()
    reward_ax.plot(np.arange(len(rewards_history)), rewards_history, label="Cumulative Reward", color="tab:red")
    reward_ax.set_xlabel("Episode")
    reward_ax.set_ylabel("Cumulative Reward")
    reward_ax.legend()
    reward_ax.grid()
    
    plt.draw()
    plt.pause(0.1)
    
    print(f"Episode {ep} reward: {cum_reward:.2f}, epsilon: {agent.epsilon:.4f}")
    
    # Save model every 100 episodes
    if ep % 100 == 0:
        torch.save(agent.qnetwork_local.state_dict(), qtable_filename)

plt.ioff()
plt.show()
