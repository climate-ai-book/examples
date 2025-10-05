"""
Carbon Cycle and Greenhouse Gas Modeling Framework
Implements ML methods for emissions quantification:

Knowledge-guided ML for carbon fluxes
Random forest for agricultural N₂O prediction
Deep Q-Network for emissions optimization
3D soil carbon mapping

Based on Chapter 10: Carbon Cycle and Greenhouse Gas Modeling
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
from collections import deque
import warnings
warnings.filterwarnings('ignore')
class KnowledgeGuidedCarbonModel:
"""
Knowledge-guided machine learning for carbon fluxes
Combines data-driven learning with physical constraints
"""

def __init__(self):
    """Initialize KGML carbon model"""
    # Temperature response (Q10 relationship)
    self.q10 = 2.0
    self.t_ref = 20.0  # °C
    
    # Moisture response parameters
    self.wfps_opt = 0.6  # Optimal water-filled pore space
    self.wfps_width = 0.08
    
    # Placeholder for neural network weights
    # In practice, use trained deep learning model
    self.weights_initialized = False
    
def temp_response_function(self, temp: np.ndarray) -> np.ndarray:
    """
    Q10 temperature response
    
    R(T) = Q10^((T-T_ref)/10)
    """
    return self.q10 ** ((temp - self.t_ref) / 10.0)

def moisture_response_function(self, wfps: np.ndarray) -> np.ndarray:
    """
    Soil moisture response (Gaussian)
    
    Optimal around 60% WFPS
    """
    return np.exp(-((wfps - self.wfps_opt) ** 2) / self.wfps_width)

def neural_network_component(self, features: np.ndarray) -> np.ndarray:
    """
    Data-driven component (placeholder)
    
    In practice, use trained neural network
    """
    # Simple linear combination as placeholder
    weights = np.array([0.3, 0.2, 0.1, 0.15, 0.1, 0.05, 0.1])
    
    if features.shape[1] >= len(weights):
        return np.dot(features[:, :len(weights)], weights)
    else:
        return np.sum(features, axis=1) * 0.2

def predict_flux(self, temp: np.ndarray, wfps: np.ndarray,
                features: np.ndarray, carbon_input: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Knowledge-guided prediction combining physics and ML
    
    Flux = f_NN(features) · Φ_temp(T) · Φ_moisture(WFPS)
    """
    # Physical constraint functions
    temp_factor = self.temp_response_function(temp)
    moisture_factor = self.moisture_response_function(wfps)
    
    # Neural network component
    nn_output = self.neural_network_component(features)
    
    # Combined prediction
    flux = nn_output * temp_factor * moisture_factor
    
    # Mass balance constraint (emissions can't exceed inputs)
    if carbon_input is not None:
        flux = np.minimum(flux, carbon_input)
    
    return np.maximum(flux, 0)  # Non-negative emissions
class AgriculturalN2OPredictor:
"""
Random forest-based N₂O emissions prediction for agriculture
Captures episodic, pulsed emissions following management events
"""

def __init__(self):
    """Initialize N₂O predictor"""
    self.feature_names = []
    
def engineer_features(self, data: Dict) -> np.ndarray:
    """
    Create features for N₂O prediction
    """
    features = []
    
    # Soil properties
    features.append(data.get('clay_content', 20))
    features.append(data.get('organic_carbon', 2.0))
    features.append(data.get('ph', 6.5))
    features.append(data.get('bulk_density', 1.3))
    
    # Environmental conditions
    features.append(data.get('soil_temperature', 15))
    features.append(data.get('wfps', 0.5))
    features.append(data.get('air_temperature', 18))
    
    # Nitrogen management
    features.append(data.get('n_fertilizer_rate', 0))
    features.append(data.get('days_since_fertilization', 30))
    
    # Rainfall
    features.append(data.get('precipitation_1day', 0))
    features.append(data.get('precipitation_7day', 5))
    features.append(data.get('days_since_rain', 10))
    
    # Critical interactions for N₂O
    wfps = data.get('wfps', 0.5)
    n_rate = data.get('n_fertilizer_rate', 0)
    temp = data.get('soil_temperature', 15)
    
    features.append(wfps * n_rate)  # WFPS × N interaction
    features.append(temp * wfps)  # Temp × WFPS interaction
    
    return np.array(features)

def predict_n2o_flux(self, features: np.ndarray,
                    include_pulse: bool = True) -> Tuple[float, float]:
    """
    Predict N₂O flux with baseline and pulse components
    
    F_N₂O = F_baseline + F_pulse
    """
    # Baseline emissions (always present)
    baseline_flux = self._baseline_emissions(features)
    
    # Pulse emissions (following fertilization/rain events)
    pulse_flux = 0.0
    if include_pulse:
        pulse_flux = self._pulse_emissions(features)
    
    total_flux = baseline_flux + pulse_flux
    
    return total_flux, pulse_flux

def _baseline_emissions(self, features: np.ndarray) -> float:
    """
    Baseline (background) N₂O emissions
    """
    # Simple empirical model
    temp = features[4]
    wfps = features[5]
    organic_c = features[1]
    
    # Temperature effect
    temp_effect = np.exp(0.1 * (temp - 10))
    
    # Moisture effect (peak around 60% WFPS)
    moisture_effect = np.exp(-((wfps - 0.6) ** 2) / 0.05)
    
    # Substrate effect
    substrate_effect = organic_c / 2.0
    
    baseline = 0.5 * temp_effect * moisture_effect * substrate_effect
    
    return max(baseline, 0.1)  # Minimum background

def _pulse_emissions(self, features: np.ndarray) -> float:
    """
    Episodic pulse emissions following events
    """
    n_rate = features[7]
    days_since_fert = features[8]
    wfps = features[5]
    days_since_rain = features[11]
    
    # Fertilization pulse (decays over ~30 days)
    if n_rate > 0 and days_since_fert < 30:
        fert_pulse = (n_rate / 100) * np.exp(-days_since_fert / 10)
    else:
        fert_pulse = 0
    
    # Rain pulse (short-lived, ~5 days)
    if days_since_rain < 5 and wfps > 0.6:
        rain_pulse = 2.0 * np.exp(-days_since_rain)
    else:
        rain_pulse = 0
    
    return fert_pulse + rain_pulse
class EmissionsOptimizationDQN:
"""
Deep Q-Network for emissions reduction optimization
Learns optimal control policies through reinforcement learning
"""

def __init__(self, state_dim: int, action_dim: int):
    """
    Initialize DQN agent
    
    Args:
        state_dim: Dimension of state space
        action_dim: Number of possible actions
    """
    self.state_dim = state_dim
    self.action_dim = action_dim
    
    # Experience replay memory
    self.memory = deque(maxlen=5000)
    
    # Hyperparameters
    self.gamma = 0.95  # Discount factor
    self.epsilon = 1.0  # Exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    
    # Simplified Q-table (placeholder for neural network)
    self.q_table = np.zeros((100, action_dim))
    
def discretize_state(self, state: np.ndarray) -> int:
    """
    Discretize continuous state for Q-table
    """
    # Simple binning (in practice, use neural network)
    state_hash = int(np.sum(state * np.arange(len(state))) % 100)
    return state_hash

def select_action(self, state: np.ndarray) -> int:
    """
    Epsilon-greedy action selection
    """
    if np.random.rand() < self.epsilon:
        return np.random.randint(self.action_dim)
    
    state_idx = self.discretize_state(state)
    return np.argmax(self.q_table[state_idx])

def store_experience(self, state: np.ndarray, action: int,
                    reward: float, next_state: np.ndarray, done: bool):
    """
    Store experience in replay memory
    """
    self.memory.append((state, action, reward, next_state, done))

def compute_reward(self, production: float, emissions: float,
                  cost: float, weights: Dict = None) -> float:
    """
    Multi-objective reward function
    
    R = w_prod·Production - w_emis·Emissions - w_cost·Cost
    """
    if weights is None:
        weights = {'production': 10.0, 'emissions': 5.0, 'cost': 2.0}
    
    reward = (weights['production'] * production -
             weights['emissions'] * emissions -
             weights['cost'] * cost)
    
    return reward

def update_q_values(self, learning_rate: float = 0.1):
    """
    Update Q-values using Bellman equation
    """
    if len(self.memory) < 32:
        return
    
    # Sample mini-batch
    batch = np.random.choice(len(self.memory), 32, replace=False)
    
    for idx in batch:
        state, action, reward, next_state, done = self.memory[idx]
        
        state_idx = self.discretize_state(state)
        next_state_idx = self.discretize_state(next_state)
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_idx])
        
        # Q-learning update
        self.q_table[state_idx, action] += learning_rate * (
            target - self.q_table[state_idx, action]
        )
    
    # Decay exploration
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay
class SoilCarbon3DMapper:
"""
3D soil organic carbon mapping
Predicts SOC across horizontal space and depth
"""

def __init__(self):
    """Initialize 3D SOC mapper"""
    self.depth_levels = np.array([0, 10, 20, 30, 50, 100])  # cm
    
def predict_soc_profile(self, surface_features: Dict,
                       depth: float = 30.0) -> np.ndarray:
    """
    Predict SOC concentration at given depth
    
    SOC(z) = SOC_surface · exp(-β·z)
    """
    # Surface SOC from features
    soc_surface = self._predict_surface_soc(surface_features)
    
    # Depth decay coefficient
    beta = self._compute_depth_decay(surface_features)
    
    # SOC profile
    if isinstance(depth, (int, float)):
        depths = np.array([depth])
    else:
        depths = depth
    
    soc_profile = soc_surface * np.exp(-beta * depths / 100)  # Convert cm to m
    
    return soc_profile

def _predict_surface_soc(self, features: Dict) -> float:
    """
    Predict surface SOC from environmental features
    """
    # Climate effect
    temp = features.get('mean_annual_temp', 15)
    precip = features.get('mean_annual_precip', 800)
    
    temp_effect = np.exp(-temp / 15)
    precip_effect = precip / 1000
    
    # Vegetation effect
    ndvi = features.get('ndvi_mean', 0.5)
    veg_effect = ndvi * 50
    
    # Soil texture effect
    clay = features.get('clay_content', 25)
    clay_effect = clay * 0.3
    
    # Topography
    elevation = features.get('elevation', 500)
    elev_effect = elevation / 500
    
    soc_surface = (temp_effect * 30 +
                  precip_effect * 20 +
                  veg_effect +
                  clay_effect +
                  elev_effect * 5)
    
    return max(soc_surface, 5)  # Minimum 5 kg C/m²

def _compute_depth_decay(self, features: Dict) -> float:
    """
    Compute depth decay coefficient
    
    Varies with soil type and land use
    """
    # Soil type effect
    clay = features.get('clay_content', 25)
    # Higher clay → slower decay (more stable)
    clay_factor = 1.0 - (clay / 100) * 0.3
    
    # Land use effect
    landuse = features.get('landuse_type', 'cropland')
    if landuse == 'forest':
        lu_factor = 0.7  # Slower decay
    elif landuse == 'grassland':
        lu_factor = 0.85
    else:  # cropland
        lu_factor = 1.0
    
    # Base decay coefficient
    beta_base = 0.03  # per cm
    
    beta = beta_base * clay_factor * lu_factor
    
    return beta
def demonstrate_carbon_ghg_framework():
"""
Demonstrate carbon cycle and GHG modeling framework
"""
print("="*70)
print("Carbon Cycle and Greenhouse Gas Modeling Framework")
print("Chapter 10: Carbon Cycle and Greenhouse Gas Modeling")
print("="*70)
np.random.seed(42)

# 1. Knowledge-Guided Carbon Model
print("\n" + "-"*70)
print("1. Knowledge-Guided Machine Learning for Carbon Fluxes")
print("-"*70)

kgml_model = KnowledgeGuidedCarbonModel()

# Environmental conditions
n_samples = 100
temp = np.random.uniform(5, 30, n_samples)
wfps = np.random.uniform(0.2, 0.8, n_samples)
features = np.random.randn(n_samples, 10)

print(f"Sample size: {n_samples}")
print(f"Temperature range: {temp.min():.1f} - {temp.max():.1f}°C")
print(f"WFPS range: {wfps.min():.2f} - {wfps.max():.2f}")

# Predict fluxes
carbon_flux = kgml_model.predict_flux(temp, wfps, features)

print(f"\nCarbon Flux Predictions:")
print(f"  Mean: {carbon_flux.mean():.3f} g C/m²/day")
print(f"  Range: [{carbon_flux.min():.3f}, {carbon_flux.max():.3f}]")
print(f"  Std dev: {carbon_flux.std():.3f}")

# 2. Agricultural N₂O Prediction
print("\n" + "-"*70)
print("2. Agricultural N₂O Emissions Prediction")
print("-"*70)

n2o_predictor = AgriculturalN2OPredictor()

# Scenario 1: Post-fertilization
scenario_fert = {
    'clay_content': 30,
    'organic_carbon': 2.5,
    'ph': 6.8,
    'bulk_density': 1.2,
    'soil_temperature': 22,
    'wfps': 0.65,
    'air_temperature': 25,
    'n_fertilizer_rate': 150,  # kg N/ha
    'days_since_fertilization': 3,
    'precipitation_1day': 15,
    'precipitation_7day': 25,
    'days_since_rain': 1
}

features_fert = n2o_predictor.engineer_features(scenario_fert)
total_flux_fert, pulse_fert = n2o_predictor.predict_n2o_flux(features_fert)

print(f"Post-Fertilization Scenario:")
print(f"  N application: {scenario_fert['n_fertilizer_rate']} kg N/ha")
print(f"  Days since fertilization: {scenario_fert['days_since_fertilization']}")
print(f"  WFPS: {scenario_fert['wfps']:.2f}")
print(f"\nN₂O Emissions:")
print(f"  Total flux: {total_flux_fert:.3f} g N₂O-N/m²/day")
print(f"  Pulse component: {pulse_fert:.3f} g N₂O-N/m²/day")
print(f"  Baseline component: {total_flux_fert - pulse_fert:.3f} g N₂O-N/m²/day")

# Scenario 2: Baseline conditions
scenario_base = scenario_fert.copy()
scenario_base['n_fertilizer_rate'] = 0
scenario_base['days_since_fertilization'] = 60
scenario_base['days_since_rain'] = 15
scenario_base['wfps'] = 0.45

features_base = n2o_predictor.engineer_features(scenario_base)
total_flux_base, pulse_base = n2o_predictor.predict_n2o_flux(features_base)

print(f"\nBaseline Scenario:")
print(f"  Total flux: {total_flux_base:.3f} g N₂O-N/m²/day")
print(f"  Reduction from peak: {((total_flux_fert - total_flux_base)/total_flux_fert)*100:.1f}%")

# 3. Emissions Optimization with RL
print("\n" + "-"*70)
print("3. Reinforcement Learning for Emissions Optimization")
print("-"*70)

state_dim = 10
action_dim = 5
dqn_agent = EmissionsOptimizationDQN(state_dim, action_dim)

print(f"DQN Configuration:")
print(f"  State dimension: {state_dim}")
print(f"  Action space: {action_dim}")
print(f"  Initial exploration: {dqn_agent.epsilon:.2f}")

# Simulate learning episodes
n_episodes = 50
rewards_history = []

for episode in range(n_episodes):
    state = np.random.randn(state_dim)
    total_reward = 0
    
    for step in range(20):
        action = dqn_agent.select_action(state)
        
        # Simulate environment response
        production = np.random.uniform(80, 100)
        emissions = np.random.uniform(10, 30) * (1 - action * 0.1)
        cost = np.random.uniform(20, 40) + action * 2
        
        reward = dqn_agent.compute_reward(production, emissions, cost)
        next_state = np.random.randn(state_dim)
        done = (step == 19)
        
        dqn_agent.store_experience(state, action, reward, next_state, done)
        dqn_agent.update_q_values()
        
        total_reward += reward
        state = next_state
    
    rewards_history.append(total_reward)

print(f"\nLearning Progress:")
print(f"  Episodes: {n_episodes}")
print(f"  Average reward (first 10): {np.mean(rewards_history[:10]):.2f}")
print(f"  Average reward (last 10): {np.mean(rewards_history[-10:]):.2f}")
print(f"  Final exploration rate: {dqn_agent.epsilon:.3f}")

# 4. 3D Soil Carbon Mapping
print("\n" + "-"*70)
print("4. 3D Soil Organic Carbon Mapping")
print("-"*70)

soc_mapper = SoilCarbon3DMapper()

# Define site characteristics
site_features = {
    'mean_annual_temp': 18,
    'mean_annual_precip': 900,
    'ndvi_mean': 0.7,
    'clay_content': 28,
    'elevation': 450,
    'landuse_type': 'grassland'
}

print(f"Site Characteristics:")
print(f"  Temperature: {site_features['mean_annual_temp']}°C")
print(f"  Precipitation: {site_features['mean_annual_precip']} mm")
print(f"  NDVI: {site_features['ndvi_mean']:.2f}")
print(f"  Land use: {site_features['landuse_type']}")

# Predict SOC profile
depths = np.array([0, 10, 20, 30, 50, 100])  # cm
soc_profile = soc_mapper.predict_soc_profile(site_features, depths)

print(f"\nSOC Profile Predictions:")
for depth, soc in zip(depths, soc_profile):
    print(f"  {depth:3d} cm: {soc:.2f} kg C/m²")

total_soc_100cm = np.trapz(soc_profile, depths)
print(f"\nTotal SOC (0-100 cm): {total_soc_100cm:.2f} kg C/m²")

# Visualize results
visualize_carbon_ghg_results(
    temp, wfps, carbon_flux,
    rewards_history,
    depths, soc_profile,
    total_flux_fert, total_flux_base
)

print("\n" + "="*70)
print("Carbon cycle and GHG modeling framework demonstration complete!")
print("="*70)
def visualize_carbon_ghg_results(temp, wfps, flux, rewards, depths, soc, n2o_fert, n2o_base):
"""Visualize carbon cycle and GHG modeling results"""
fig = plt.figure(figsize=(16, 10))
# Plot 1: Temperature vs Carbon Flux
ax1 = plt.subplot(2, 3, 1)
scatter = ax1.scatter(temp, flux, c=wfps, cmap='Blues', alpha=0.6, s=50)
ax1.set_xlabel('Temperature (°C)')
ax1.set_ylabel('Carbon Flux (g C/m²/day)')
ax1.set_title('Knowledge-Guided Carbon Flux\n(Color = WFPS)')
plt.colorbar(scatter, ax=ax1, label='WFPS')
ax1.grid(True, alpha=0.3)

# Plot 2: WFPS vs Carbon Flux
ax2 = plt.subplot(2, 3, 2)
scatter2 = ax2.scatter(wfps, flux, c=temp, cmap='RdYlBu_r', alpha=0.6, s=50)
ax2.set_xlabel('Water-Filled Pore Space')
ax2.set_ylabel('Carbon Flux (g C/m²/day)')
ax2.set_title('Moisture Response\n(Color = Temperature)')
plt.colorbar(scatter2, ax=ax2, label='Temperature (°C)')
ax2.grid(True, alpha=0.3)

# Plot 3: RL Learning Curve
ax3 = plt.subplot(2, 3, 3)
episodes = np.arange(len(rewards))
ax3.plot(episodes, rewards, 'b-', alpha=0.5, linewidth=1)

# Moving average
window = 5
moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
ax3.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label='Moving Average')

ax3.set_xlabel('Episode')
ax3.set_ylabel('Total Reward')
ax3.set_title('RL Emissions Optimization\n(Learning Progress)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: SOC Depth Profile
ax4 = plt.subplot(2, 3, 4)
ax4.plot(soc, depths, 'o-', linewidth=2, markersize=8, color='brown')
ax4.fill_betweenx(depths, 0, soc, alpha=0.3, color='brown')
ax4.set_ylabel('Depth (cm)')
ax4.set_xlabel('SOC (kg C/m²)')
ax4.set_title('3D Soil Carbon Profile')
ax4.invert_yaxis()
ax4.grid(True, alpha=0.3)

# Plot 5: N₂O Emissions Comparison
ax5 = plt.subplot(2, 3, 5)
scenarios = ['Post-\nFertilization', 'Baseline']
emissions = [n2o_fert, n2o_base]
colors = ['red', 'green']

bars = ax5.bar(scenarios, emissions, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax5.set_ylabel('N₂O Flux (g N₂O-N/m²/day)')
ax5.set_title('Agricultural N₂O Emissions')
ax5.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars, emissions):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 6: Cumulative SOC
ax6 = plt.subplot(2, 3, 6)
cumulative_soc = np.cumsum(soc)
ax6.plot(depths, cumulative_soc, 'o-', linewidth=2, markersize=8, color='darkgreen')
ax6.fill_between(depths, 0, cumulative_soc, alpha=0.3, color='green')
ax6.set_xlabel('Depth (cm)')
ax6.set_ylabel('Cumulative SOC (kg C/m²)')
ax6.set_title('Cumulative Soil Carbon Stock')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('carbon_ghg_modeling_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nResults saved to 'carbon_ghg_modeling_results.png'")
if name == "main":
demonstrate_carbon_ghg_framework()
