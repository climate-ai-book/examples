"""
Renewable Energy and Climate Variability Framework
Implements ML methods for power generation forecasting:

CNN-LSTM for solar PV forecasting
Ensemble methods for wind power prediction
SHAP for explainable AI
Reinforcement learning for grid optimization

Based on Chapter 11: Renewable Energy and Climate Variability
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
from collections import deque
import warnings
warnings.filterwarnings('ignore')
class SolarIrradianceModel:
"""
Solar irradiance calculation and forecasting
Computes surface irradiance from atmospheric and geometric factors
"""

def __init__(self):
    """Initialize solar irradiance model"""
    self.solar_constant = 1361.0  # W/m² at top of atmosphere
    
def solar_zenith_angle(self, latitude: float, day_of_year: int,
                      hour: float) -> float:
    """
    Calculate solar zenith angle
    
    θ_z = arccos(sin(φ)sin(δ) + cos(φ)cos(δ)cos(h))
    """
    # Solar declination
    delta = 23.45 * np.sin(np.radians(360/365 * (day_of_year - 81)))
    delta_rad = np.radians(delta)
    
    # Hour angle
    hour_angle = 15 * (hour - 12)  # degrees
    hour_angle_rad = np.radians(hour_angle)
    
    # Latitude in radians
    lat_rad = np.radians(latitude)
    
    # Zenith angle
    cos_zenith = (np.sin(lat_rad) * np.sin(delta_rad) +
                 np.cos(lat_rad) * np.cos(delta_rad) * np.cos(hour_angle_rad))
    
    zenith_angle = np.arccos(np.clip(cos_zenith, -1, 1))
    
    return np.degrees(zenith_angle)

def clear_sky_irradiance(self, zenith_angle: float,
                        aerosol_depth: float = 0.1,
                        precipitable_water: float = 1.0) -> Dict[str, float]:
    """
    Clear-sky irradiance model (simplified)
    
    Returns direct, diffuse, and total irradiance
    """
    if zenith_angle >= 90:
        return {'direct': 0, 'diffuse': 0, 'total': 0}
    
    # Air mass
    zenith_rad = np.radians(zenith_angle)
    air_mass = 1 / np.cos(zenith_rad)
    
    # Atmospheric transmittance (simplified Beer-Lambert)
    tau_rayleigh = np.exp(-0.09 * air_mass)
    tau_aerosol = np.exp(-aerosol_depth * air_mass)
    tau_water = np.exp(-0.077 * (precipitable_water * air_mass) ** 0.3)
    
    transmittance = tau_rayleigh * tau_aerosol * tau_water
    
    # Direct normal irradiance
    dni = self.solar_constant * transmittance
    
    # Direct horizontal irradiance
    dhi_direct = dni * np.cos(zenith_rad)
    
    # Diffuse horizontal irradiance (simplified)
    dhi_diffuse = self.solar_constant * np.cos(zenith_rad) * 0.1 * (1 - transmittance)
    
    # Total
    ghi_total = dhi_direct + dhi_diffuse
    
    return {
        'direct': dhi_direct,
        'diffuse': dhi_diffuse,
        'total': ghi_total
    }
class PhotovoltaicForecaster:
"""
PV power forecasting using hybrid CNN-LSTM approach
Combines spatial pattern recognition with temporal modeling
"""

def __init__(self):
    """Initialize PV forecaster"""
    # System parameters
    self.panel_efficiency = 0.18
    self.panel_area = 1.6  # m² per panel
    self.system_efficiency = 0.85  # Inverter and other losses
    
    # Temperature coefficients
    self.temp_coeff = -0.004  # per °C
    self.noct = 45  # Nominal operating cell temperature
    
def cell_temperature(self, ambient_temp: float, irradiance: float,
                    wind_speed: float = 1.0) -> float:
    """
    Estimate cell temperature
    
    T_cell = T_ambient + (NOCT - 20)/800 · G · (1 - η/τα)
    """
    # Simplified model
    delta_t = (self.noct - 20) / 800 * irradiance
    
    # Wind cooling effect
    wind_correction = 1 / (1 + 0.05 * wind_speed)
    
    cell_temp = ambient_temp + delta_t * wind_correction
    
    return cell_temp

def pv_power(self, irradiance: float, ambient_temp: float,
            wind_speed: float = 1.0, n_panels: int = 100) -> float:
    """
    Calculate PV power output
    
    P_PV = G · A · η · η_system · temp_factor
    """
    if irradiance <= 0:
        return 0
    
    # Cell temperature
    t_cell = self.cell_temperature(ambient_temp, irradiance, wind_speed)
    
    # Temperature correction factor
    temp_factor = 1 + self.temp_coeff * (t_cell - 25)
    
    # Power calculation
    power = (irradiance * self.panel_area * self.panel_efficiency *
            self.system_efficiency * temp_factor * n_panels)
    
    return max(power, 0)

def simple_lstm_forecast(self, irradiance_history: np.ndarray,
                        forecast_horizon: int = 24) -> np.ndarray:
    """
    Simple LSTM-style forecast (placeholder)
    
    In practice, use trained deep learning model
    """
    # Use persistence with trend
    trend = np.mean(np.diff(irradiance_history[-10:]))
    
    forecast = []
    last_value = irradiance_history[-1]
    
    for h in range(forecast_horizon):
        # Add trend with decay
        next_value = last_value + trend * np.exp(-h / 10)
        
        # Add diurnal pattern (simplified)
        hour = h % 24
        diurnal_factor = max(0, np.sin(np.pi * (hour - 6) / 12))
        next_value *= diurnal_factor
        
        forecast.append(max(next_value, 0))
        last_value = next_value
    
    return np.array(forecast)
class WindPowerForecaster:
"""
Wind power forecasting with ensemble methods
Combines multiple models and NWP sources
"""

def __init__(self, turbine_params: Optional[Dict] = None):
    """
    Initialize wind power forecaster
    
    Args:
        turbine_params: Turbine characteristics
    """
    if turbine_params is None:
        turbine_params = {
            'rated_power': 2000,  # kW
            'cut_in_speed': 3,     # m/s
            'rated_speed': 12,     # m/s
            'cut_out_speed': 25    # m/s
        }
    
    self.params = turbine_params
    
def power_curve(self, wind_speed: float) -> float:
    """
    Turbine power curve
    
    Piecewise function with cut-in, rated, and cut-out speeds
    """
    v = wind_speed
    v_ci = self.params['cut_in_speed']
    v_r = self.params['rated_speed']
    v_co = self.params['cut_out_speed']
    p_rated = self.params['rated_power']
    
    if v < v_ci or v >= v_co:
        return 0
    elif v < v_r:
        # Cubic approximation in transition region
        power = p_rated * ((v - v_ci) / (v_r - v_ci)) ** 3
        return power
    else:  # v_r <= v < v_co
        return p_rated

def adjust_for_density(self, power: float, air_density: float,
                      reference_density: float = 1.225) -> float:
    """
    Adjust power for air density
    
    P_adjusted = P · (ρ/ρ_ref)
    """
    return power * (air_density / reference_density)

def ensemble_forecast(self, wind_forecasts: List[np.ndarray],
                     weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create ensemble wind power forecast
    
    Returns mean and standard deviation
    """
    if weights is None:
        weights = np.ones(len(wind_forecasts)) / len(wind_forecasts)
    
    # Convert wind speeds to power
    power_forecasts = []
    for wind_forecast in wind_forecasts:
        power = np.array([self.power_curve(v) for v in wind_forecast])
        power_forecasts.append(power)
    
    power_forecasts = np.array(power_forecasts)
    
    # Weighted ensemble mean
    ensemble_mean = np.average(power_forecasts, axis=0, weights=weights)
    
    # Ensemble spread
    ensemble_std = np.std(power_forecasts, axis=0)
    
    return ensemble_mean, ensemble_std
class ExplainableAIAnalyzer:
"""
SHAP-style explainable AI for renewable energy
Analyzes feature importance and interactions
"""

def __init__(self):
    """Initialize explainable AI analyzer"""
    self.feature_names = []
    
def compute_feature_importance(self, model_predictions: np.ndarray,
                               features: np.ndarray,
                               feature_names: List[str]) -> Dict:
    """
    Compute feature importance (simplified SHAP approximation)
    
    Returns importance scores for each feature
    """
    self.feature_names = feature_names
    n_features = features.shape[1]
    
    # Simplified importance: correlation with output
    importance = {}
    
    for i, name in enumerate(feature_names):
        correlation = np.corrcoef(features[:, i], model_predictions)[0, 1]
        importance[name] = abs(correlation)
    
    return importance

def partial_dependence(self, model_func: callable,
                      features: np.ndarray,
                      feature_idx: int,
                      n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute partial dependence plot
    
    PDP_j(x_j) = E_{X_{-j}}[f(x_j, X_{-j})]
    """
    # Range of feature values
    feature_values = features[:, feature_idx]
    x_range = np.linspace(feature_values.min(), feature_values.max(), n_points)
    
    # Compute average prediction for each value
    pdp_values = []
    
    for x_val in x_range:
        # Create dataset with feature set to x_val
        temp_features = features.copy()
        temp_features[:, feature_idx] = x_val
        
        # Average predictions
        predictions = model_func(temp_features)
        pdp_values.append(np.mean(predictions))
    
    return x_range, np.array(pdp_values)
class SmartGridOptimizer:
"""
Reinforcement learning for smart grid control
Optimizes dispatch and storage under renewable variability
"""

def __init__(self, battery_capacity: float = 1000):
    """
    Initialize smart grid optimizer
    
    Args:
        battery_capacity: Battery capacity in kWh
    """
    self.battery_capacity = battery_capacity
    self.battery_efficiency = 0.90
    self.max_charge_rate = battery_capacity * 0.5  # C-rate = 0.5
    
    # Q-table (simplified)
    self.q_table = {}
    self.learning_rate = 0.1
    self.discount_factor = 0.95
    self.epsilon = 0.1
    
def get_actions(self) -> List[str]:
    """Available actions"""
    return ['charge', 'discharge', 'hold']

def state_to_key(self, soc: float, renewable_power: float,
                demand: float, price: float) -> Tuple:
    """
    Discretize state for Q-table
    """
    soc_bin = int(soc * 10)  # 10 bins
    power_bin = int(renewable_power / 500)
    demand_bin = int(demand / 500)
    price_bin = int(price * 10)
    
    return (soc_bin, power_bin, demand_bin, price_bin)

def select_action(self, state_key: Tuple) -> str:
    """
    Epsilon-greedy action selection
    """
    if np.random.rand() < self.epsilon:
        return np.random.choice(self.get_actions())
    
    # Get Q-values for state
    if state_key not in self.q_table:
        self.q_table[state_key] = {a: 0 for a in self.get_actions()}
    
    q_values = self.q_table[state_key]
    return max(q_values, key=q_values.get)

def compute_reward(self, action: str, renewable_power: float,
                  demand: float, price: float, soc: float) -> float:
    """
    Multi-objective reward function
    
    R = w_cost·Cost_reduction - w_curtail·Curtailment - w_degradation·Degradation
    """
    # Grid power needed
    if action == 'charge':
        grid_power = demand + self.max_charge_rate - renewable_power
        curtailment = 0
    elif action == 'discharge':
        grid_power = max(0, demand - self.max_charge_rate - renewable_power)
        curtailment = max(0, renewable_power - demand - self.max_charge_rate)
    else:  # hold
        grid_power = max(0, demand - renewable_power)
        curtailment = max(0, renewable_power - demand)
    
    # Cost (negative is good)
    cost = -grid_power * price
    
    # Curtailment penalty
    curtailment_penalty = -curtailment * 0.5
    
    # Battery degradation penalty
    if action in ['charge', 'discharge']:
        degradation_penalty = -0.1
    else:
        degradation_penalty = 0
    
    reward = cost + curtailment_penalty + degradation_penalty
    
    return reward

def update_q_value(self, state: Tuple, action: str, reward: float,
                  next_state: Tuple):
    """
    Q-learning update
    
    Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    """
    if state not in self.q_table:
        self.q_table[state] = {a: 0 for a in self.get_actions()}
    if next_state not in self.q_table:
        self.q_table[next_state] = {a: 0 for a in self.get_actions()}
    
    current_q = self.q_table[state][action]
    max_next_q = max(self.q_table[next_state].values())
    
    new_q = current_q + self.learning_rate * (
        reward + self.discount_factor * max_next_q - current_q
    )
    
    self.q_table[state][action] = new_q
def demonstrate_renewable_energy_framework():
"""
Demonstrate renewable energy and climate variability framework
"""
print("="*70)
print("Renewable Energy and Climate Variability Framework")
print("Chapter 11: Renewable Energy and Climate Variability")
print("="*70)
np.random.seed(42)

# 1. Solar Irradiance Modeling
print("\n" + "-"*70)
print("1. Solar Irradiance Calculation")
print("-"*70)

solar_model = SolarIrradianceModel()

latitude = 35.0  # degrees
day_of_year = 172  # June 21 (summer solstice)

print(f"Location: {latitude}°N")
print(f"Date: Day {day_of_year} (June 21)")

# Calculate for different hours
hours = [6, 9, 12, 15, 18]
print(f"\nSolar Irradiance Throughout Day:")

for hour in hours:
    zenith = solar_model.solar_zenith_angle(latitude, day_of_year, hour)
    irradiance = solar_model.clear_sky_irradiance(zenith)
    
    print(f"  {hour:02d}:00 - Zenith: {zenith:.1f}°, GHI: {irradiance['total']:.0f} W/m²")

# 2. PV Power Forecasting
print("\n" + "-"*70)
print("2. Photovoltaic Power Forecasting")
print("-"*70)

pv_forecaster = PhotovoltaicForecaster()

# Current conditions
current_irradiance = 800  # W/m²
ambient_temp = 28  # °C
wind_speed = 3  # m/s
n_panels = 200

print(f"System Configuration:")
print(f"  Number of panels: {n_panels}")
print(f"  Panel efficiency: {pv_forecaster.panel_efficiency*100:.1f}%")
print(f"  Panel area: {pv_forecaster.panel_area} m²")

current_power = pv_forecaster.pv_power(
    current_irradiance, ambient_temp, wind_speed, n_panels
)

cell_temp = pv_forecaster.cell_temperature(ambient_temp, current_irradiance, wind_speed)

print(f"\nCurrent Conditions:")
print(f"  Irradiance: {current_irradiance} W/m²")
print(f"  Ambient temperature: {ambient_temp}°C")
print(f"  Cell temperature: {cell_temp:.1f}°C")
print(f"  Power output: {current_power:.1f} kW")

# Forecast
irradiance_history = np.random.uniform(200, 900, 48)
forecast = pv_forecaster.simple_lstm_forecast(irradiance_history, forecast_horizon=24)

print(f"\n24-Hour Forecast:")
print(f"  Mean: {forecast.mean():.0f} W/m²")
print(f"  Peak: {forecast.max():.0f} W/m²")

# 3. Wind Power Forecasting
print("\n" + "-"*70)
print("3. Wind Power Ensemble Forecasting")
print("-"*70)

wind_forecaster = WindPowerForecaster()

print(f"Turbine Parameters:")
print(f"  Rated power: {wind_forecaster.params['rated_power']} kW")
print(f"  Cut-in speed: {wind_forecaster.params['cut_in_speed']} m/s")
print(f"  Rated speed: {wind_forecaster.params['rated_speed']} m/s")
print(f"  Cut-out speed: {wind_forecaster.params['cut_out_speed']} m/s")

# Multiple wind forecasts (from different NWP models)
wind_forecasts = [
    np.random.uniform(4, 15, 24),
    np.random.uniform(3, 14, 24),
    np.random.uniform(5, 16, 24)
]

ensemble_mean, ensemble_std = wind_forecaster.ensemble_forecast(wind_forecasts)

print(f"\nEnsemble Forecast (24 hours):")
print(f"  Mean power: {ensemble_mean.mean():.0f} kW")
print(f"  Mean uncertainty: {ensemble_std.mean():.0f} kW")
print(f"  Capacity factor: {(ensemble_mean.mean() / wind_forecaster.params['rated_power'])*100:.1f}%")

# 4. Explainable AI Analysis
print("\n" + "-"*70)
print("4. Explainable AI for Weather-Energy Interactions")
print("-"*70)

xai_analyzer = ExplainableAIAnalyzer()

# Synthetic data
n_samples = 200
features = np.random.randn(n_samples, 5)
feature_names = ['Temperature', 'Wind Speed', 'Cloud Cover', 'Humidity', 'Pressure']

# Simple model: power depends on features
model_predictions = (0.5 * features[:, 0] +
                    0.8 * features[:, 1] -
                    0.6 * features[:, 2] +
                    0.2 * features[:, 3] +
                    0.1 * features[:, 4] +
                    np.random.randn(n_samples) * 0.1)

# Feature importance
importance = xai_analyzer.compute_feature_importance(
    model_predictions, features, feature_names
)

print(f"Feature Importance (SHAP-style):")
sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
for name, score in sorted_features:
    print(f"  {name:15s}: {score:.3f}")

# 5. Smart Grid Optimization
print("\n" + "-"*70)
print("5. Smart Grid Control with Reinforcement Learning")
print("-"*70)

grid_optimizer = SmartGridOptimizer(battery_capacity=1000)

print(f"Battery Storage Configuration:")
print(f"  Capacity: {grid_optimizer.battery_capacity} kWh")
print(f"  Efficiency: {grid_optimizer.battery_efficiency*100:.0f}%")
print(f"  Max charge rate: {grid_optimizer.max_charge_rate} kW")

# Simulate learning episodes
n_episodes = 100
total_rewards = []

for episode in range(n_episodes):
    soc = 0.5  # Start at 50% SOC
    episode_reward = 0
    
    for step in range(24):  # 24 hours
        # Simulate conditions
        renewable_power = np.random.uniform(0, 800)
        demand = np.random.uniform(400, 1000)
        price = np.random.uniform(0.1, 0.3)
        
        # State
        state = grid_optimizer.state_to_key(soc, renewable_power, demand, price)
        
        # Select action
        action = grid_optimizer.select_action(state)
        
        # Compute reward
        reward = grid_optimizer.compute_reward(
            action, renewable_power, demand, price, soc
        )
        
        # Update SOC (simplified)
        if action == 'charge' and soc < 0.95:
            soc += 0.05
        elif action == 'discharge' and soc > 0.05:
            soc -= 0.05
        
        # Next state
        next_state = grid_optimizer.state_to_key(soc, renewable_power, demand, price)
        
        # Q-learning update
        grid_optimizer.update_q_value(state, action, reward, next_state)
        
        episode_reward += reward
    
    total_rewards.append(episode_reward)

print(f"\nLearning Progress:")
print(f"  Episodes: {n_episodes}")
print(f"  Average reward (first 10): {np.mean(total_rewards[:10]):.1f}")
print(f"  Average reward (last 10): {np.mean(total_rewards[-10:]):.1f}")
print(f"  Improvement: {np.mean(total_rewards[-10:]) - np.mean(total_rewards[:10]):.1f}")

# Visualize results
visualize_renewable_energy_results(
    hours, [solar_model.clear_sky_irradiance(
        solar_model.solar_zenith_angle(latitude, day_of_year, h)
    )['total'] for h in range(24)],
    ensemble_mean, ensemble_std,
    importance, total_rewards
)

print("\n" + "="*70)
print("Renewable energy framework demonstration complete!")
print("="*70)
def visualize_renewable_energy_results(hours, solar_irradiance, wind_power,
wind_uncertainty, feature_importance, rl_rewards):
"""Visualize renewable energy modeling results"""
fig = plt.figure(figsize=(16, 10))
# Plot 1: Solar irradiance diurnal cycle
ax1 = plt.subplot(2, 3, 1)
hours_24 = np.arange(24)
ax1.plot(hours_24, solar_irradiance, 'o-', linewidth=2, markersize=6, color='orange')
ax1.fill_between(hours_24, 0, solar_irradiance, alpha=0.3, color='yellow')
ax1.set_xlabel('Hour of Day')
ax1.set_ylabel('Global Horizontal Irradiance (W/m²)')
ax1.set_title('Solar Irradiance Diurnal Pattern')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 23])

# Plot 2: Wind power forecast with uncertainty
ax2 = plt.subplot(2, 3, 2)
hours_forecast = np.arange(len(wind_power))
ax2.plot(hours_forecast, wind_power, 'b-', linewidth=2, label='Ensemble Mean')
ax2.fill_between(hours_forecast,
                 wind_power - wind_uncertainty,
                 wind_power + wind_uncertainty,
                 alpha=0.3, color='blue', label='±1 Std Dev')
ax2.set_xlabel('Forecast Hour')
ax2.set_ylabel('Wind Power (kW)')
ax2.set_title('Wind Power Ensemble Forecast')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Feature importance
ax3 = plt.subplot(2, 3, 3)
features = list(feature_importance.keys())
importances = list(feature_importance.values())
colors = plt.cm.viridis(np.linspace(0, 1, len(features)))

bars = ax3.barh(features, importances, color=colors, edgecolor='black')
ax3.set_xlabel('Importance Score')
ax3.set_title('Feature Importance (SHAP-style)')
ax3.grid(True, alpha=0.3, axis='x')

# Plot 4: RL learning curve
ax4 = plt.subplot(2, 3, 4)
episodes = np.arange(len(rl_rewards))
ax4.plot(episodes, rl_rewards, 'g-', alpha=0.5, linewidth=1)

# Moving average
window = 10
if len(rl_rewards) >= window:
    moving_avg = np.convolve(rl_rewards, np.ones(window)/window, mode='valid')
    ax4.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label='Moving Average')

ax4.set_xlabel('Episode')
ax4.set_ylabel('Total Reward')
ax4.set_title('RL Grid Optimization Learning')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Power generation comparison
ax5 = plt.subplot(2, 3, 5)
generation_types = ['Solar\n(Peak)', 'Wind\n(Mean)', 'Wind\n(Peak)']
generation_values = [
    max(solar_irradiance) * 0.2,  # Convert to approximate power
    wind_power.mean(),
    wind_power.max()
]
colors_gen = ['orange', 'blue', 'darkblue']

bars = ax5.bar(generation_types, generation_values, color=colors_gen,
              alpha=0.7, edgecolor='black', linewidth=2)
ax5.set_ylabel('Power (kW)')
ax5.set_title('Generation Capacity Comparison')
ax5.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars, generation_values):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.0f}', ha='center', va='bottom', fontweight='bold')

# Plot 6: Cumulative RL rewards
ax6 = plt.subplot(2, 3, 6)
cumulative_rewards = np.cumsum(rl_rewards)
ax6.plot(episodes, cumulative_rewards, 'purple', linewidth=2)
ax6.fill_between(episodes, 0, cumulative_rewards, alpha=0.3, color='purple')
ax6.set_xlabel('Episode')
ax6.set_ylabel('Cumulative Reward')
ax6.set_title('RL Cumulative Performance')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('renewable_energy_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nResults saved to 'renewable_energy_results.png'")
if name == "main":
demonstrate_renewable_energy_framework()
