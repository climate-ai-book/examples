"""
Climate Risk Assessment and Adaptation Framework
Implements ML methods for climate risk analysis:

Flood susceptibility mapping with ensemble methods
Reinforcement learning for adaptive management
Infrastructure resilience assessment
Multi-criteria adaptation planning

Based on Chapter 13: Climate Risk Assessment and Adaptation
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import warnings
warnings.filterwarnings('ignore')
@dataclass
class ClimateRiskComponents:
"""Components of climate risk assessment"""
hazard: float
exposure: float
vulnerability: float
def calculate_risk(self) -> float:
    """
    Calculate integrated risk
    
    Risk = Hazard × Exposure × Vulnerability
    """
    return self.hazard * self.exposure * self.vulnerability
class FloodSusceptibilityMapper:
"""
Flood susceptibility mapping using ensemble methods
Integrates topographic, hydrological, and meteorological factors
"""

def __init__(self):
    """Initialize flood susceptibility mapper"""
    # Model weights (simplified ensemble)
    self.weights = None
    self.feature_names = []
    
def extract_flood_factors(self, location_data: Dict) -> np.ndarray:
    """
    Extract conditioning factors for flood susceptibility
    
    S_flood(x) = f_ensemble(T_topo, H_hydro, M_meteor, L_landuse, S_soil)
    """
    factors = []
    
    # Topographic factors
    factors.append(location_data.get('elevation', 100))
    factors.append(location_data.get('slope', 5))
    factors.append(location_data.get('twi', 10))  # Topographic wetness index
    factors.append(location_data.get('curvature', 0))
    
    # Hydrological factors
    factors.append(location_data.get('distance_to_river', 1000))
    factors.append(location_data.get('drainage_density', 0.5))
    factors.append(location_data.get('stream_power_index', 5))
    
    # Meteorological factors
    factors.append(location_data.get('annual_precipitation', 800))
    factors.append(location_data.get('rainfall_intensity', 50))
    factors.append(location_data.get('days_heavy_rain', 10))
    
    # Land use factors
    factors.append(location_data.get('impervious_fraction', 0.3))
    factors.append(location_data.get('vegetation_cover', 0.6))
    
    # Soil factors
    factors.append(location_data.get('soil_permeability', 0.5))
    factors.append(location_data.get('soil_moisture', 0.3))
    
    self.feature_names = [
        'elevation', 'slope', 'twi', 'curvature',
        'distance_to_river', 'drainage_density', 'stream_power',
        'precipitation', 'rainfall_intensity', 'heavy_rain_days',
        'impervious', 'vegetation', 'permeability', 'soil_moisture'
    ]
    
    return np.array(factors)

def train_susceptibility_model(self, X: np.ndarray, y: np.ndarray):
    """
    Train flood susceptibility model
    
    Simplified linear model (in practice, use Random Forest/XGBoost)
    """
    # Add bias term
    X_with_bias = np.column_stack([np.ones(len(X)), X])
    
    # Least squares solution
    self.weights = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]

def predict_susceptibility(self, X: np.ndarray) -> np.ndarray:
    """
    Predict flood susceptibility
    
    Returns probability between 0 and 1
    """
    if self.weights is None:
        raise ValueError("Model must be trained first")
    
    # Add bias term
    X_with_bias = np.column_stack([np.ones(len(X)), X])
    
    # Predict and apply sigmoid
    logits = X_with_bias @ self.weights
    susceptibility = 1 / (1 + np.exp(-logits))
    
    return susceptibility

def hybrid_prediction(self, physical_model_output: float,
                     ml_features: np.ndarray) -> float:
    """
    Hybrid physics-ML flood prediction
    
    h_hybrid = h_physical + Δh_ML
    """
    # ML correction
    ml_correction = self.predict_susceptibility(ml_features.reshape(1, -1))[0]
    
    # Combine (simplified)
    hybrid_prediction = physical_model_output + ml_correction * 0.5
    
    return hybrid_prediction
class AdaptiveFloodManagementRL:
"""
Reinforcement learning for adaptive flood management
Optimizes coastal protection strategies over time
"""

def __init__(self, n_actions: int = 5):
    """
    Initialize RL agent
    
    Args:
        n_actions: Number of possible adaptation actions
    """
    self.n_actions = n_actions
    self.actions = [
        'build_seawall',
        'restore_wetland',
        'update_building_codes',
        'improve_drainage',
        'managed_retreat'
    ]
    
    # Q-table (simplified)
    self.q_table = {}
    self.learning_rate = 0.1
    self.discount_factor = 0.95
    self.epsilon = 0.1
    
def discretize_state(self, sea_level_rise: float, storm_prob: float,
                    exposure: float, budget: float) -> Tuple:
    """
    Discretize continuous state for Q-table
    
    State includes: SLR, storm probability, exposure, budget
    """
    slr_bin = int(sea_level_rise / 0.1)  # 10cm bins
    storm_bin = int(storm_prob * 10)
    exposure_bin = int(exposure / 1000)
    budget_bin = int(budget / 10000)
    
    return (slr_bin, storm_bin, exposure_bin, budget_bin)

def select_action(self, state: Tuple) -> int:
    """
    Epsilon-greedy action selection
    """
    if np.random.rand() < self.epsilon:
        return np.random.randint(self.n_actions)
    
    # Get Q-values for state
    if state not in self.q_table:
        self.q_table[state] = np.zeros(self.n_actions)
    
    return np.argmax(self.q_table[state])

def compute_multi_objective_reward(self, action_idx: int,
                                   protection_benefit: float,
                                   cost: float,
                                   ecosystem_service: float,
                                   carbon: float,
                                   equity: float,
                                   weights: Optional[Dict] = None) -> float:
    """
    Multi-objective reward function
    
    R_multi = w^T · [R_protection, -C_cost, R_ecosystem, -E_carbon, R_equity]
    """
    if weights is None:
        weights = {
            'protection': 0.4,
            'cost': 0.2,
            'ecosystem': 0.2,
            'carbon': 0.1,
            'equity': 0.1
        }
    
    reward = (weights['protection'] * protection_benefit -
             weights['cost'] * cost +
             weights['ecosystem'] * ecosystem_service -
             weights['carbon'] * carbon +
             weights['equity'] * equity)
    
    return reward

def update_q_value(self, state: Tuple, action: int, reward: float,
                  next_state: Tuple):
    """
    Q-learning update
    
    Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    """
    if state not in self.q_table:
        self.q_table[state] = np.zeros(self.n_actions)
    if next_state not in self.q_table:
        self.q_table[next_state] = np.zeros(self.n_actions)
    
    current_q = self.q_table[state][action]
    max_next_q = np.max(self.q_table[next_state])
    
    new_q = current_q + self.learning_rate * (
        reward + self.discount_factor * max_next_q - current_q
    )
    
    self.q_table[state][action] = new_q
class InfrastructureResilienceAssessor:
"""
Infrastructure resilience assessment
Evaluates robustness and recovery capacity
"""

def __init__(self):
    """Initialize resilience assessor"""
    self.baseline_functionality = 1.0
    
def simulate_disruption(self, initial_functionality: float,
                       disruption_magnitude: float,
                       recovery_rate: float,
                       days: int = 100) -> np.ndarray:
    """
    Simulate infrastructure functionality after disruption
    
    Q(t) follows exponential recovery
    """
    functionality = np.zeros(days)
    
    # Immediate drop
    functionality[0] = initial_functionality * (1 - disruption_magnitude)
    
    # Exponential recovery
    for t in range(1, days):
        recovery = recovery_rate * (self.baseline_functionality - functionality[t-1])
        functionality[t] = functionality[t-1] + recovery
        functionality[t] = min(functionality[t], self.baseline_functionality)
    
    return functionality

def calculate_resilience(self, functionality: np.ndarray,
                        redundancy: float = 0.2,
                        cascade_prob: float = 0.1) -> float:
    """
    Calculate resilience metric
    
    R = ∫ Q(t)/Q₀ dt + λ_redundancy·D - λ_cascade·P_cascade
    """
    # Integration of functionality
    recovery_integral = np.trapz(functionality / self.baseline_functionality)
    
    # Redundancy bonus
    redundancy_bonus = 5.0 * redundancy
    
    # Cascade penalty
    cascade_penalty = 10.0 * cascade_prob
    
    resilience = recovery_integral + redundancy_bonus - cascade_penalty
    
    return resilience

def recovery_time(self, functionality: np.ndarray,
                 threshold: float = 0.9) -> int:
    """
    Calculate time to recovery
    
    Days until functionality reaches threshold
    """
    recovery_idx = np.where(functionality >= threshold)[0]
    
    if len(recovery_idx) > 0:
        return recovery_idx[0]
    else:
        return len(functionality)
class FinancialClimateRiskAssessor:
"""
Financial risk assessment with climate factors
Integrates physical and transition risks
"""

def __init__(self):
    """Initialize financial risk assessor"""
    self.baseline_default_rate = 0.05
    
def assess_physical_risk(self, asset_value: float,
                        flood_probability: float,
                        damage_fraction: float = 0.3) -> float:
    """
    Assess physical risk from climate hazards
    
    PhysicalRisk = Asset_value × P(hazard) × Damage_fraction
    """
    expected_loss = asset_value * flood_probability * damage_fraction
    return expected_loss

def assess_transition_risk(self, revenue: float,
                          carbon_intensity: float,
                          carbon_price: float = 50) -> float:
    """
    Assess transition risk from policy/technology changes
    
    TransitionRisk = Revenue × Carbon_intensity × Carbon_price
    """
    transition_cost = revenue * carbon_intensity * carbon_price / 1000
    return transition_cost

def predict_default_probability(self, financial_health: float,
                                climate_risk_exposure: float,
                                sector_vulnerability: float,
                                geography_risk: float) -> float:
    """
    Predict default probability with climate factors
    
    P_default = f_ML(F_financial, R_climate, S_sector, G_geography)
    """
    # Simplified logistic model
    score = (0.4 * (1 - financial_health) +
            0.3 * climate_risk_exposure +
            0.2 * sector_vulnerability +
            0.1 * geography_risk)
    
    # Apply sigmoid
    default_prob = self.baseline_default_rate * (1 + np.tanh(score))
    
    return min(default_prob, 1.0)
class AdaptationPathwayOptimizer:
"""
Multi-criteria adaptation pathway analysis
Evaluates and ranks adaptation options
"""

def __init__(self):
    """Initialize pathway optimizer"""
    self.criteria = [
        'effectiveness',
        'cost',
        'co_benefits',
        'equity',
        'flexibility'
    ]
    
def score_adaptation_option(self, option_data: Dict,
                           weights: Optional[Dict] = None) -> float:
    """
    Multi-criteria scoring
    
    Score = Σ wⱼ·Cⱼ(a)
    """
    if weights is None:
        weights = {c: 1.0 / len(self.criteria) for c in self.criteria}
    
    total_score = 0
    
    for criterion in self.criteria:
        criterion_value = option_data.get(criterion, 0.5)
        weight = weights.get(criterion, 0.2)
        
        total_score += weight * criterion_value
    
    return total_score

def rank_options(self, options: List[Dict],
                weights: Optional[Dict] = None) -> List[Tuple[int, float]]:
    """
    Rank adaptation options by score
    
    Returns list of (index, score) tuples sorted by score
    """
    scores = []
    
    for i, option in enumerate(options):
        score = self.score_adaptation_option(option, weights)
        scores.append((i, score))
    
    # Sort by score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores

def robust_decision_making(self, options: List[Dict],
                          scenarios: List[Dict],
                          weights: Optional[Dict] = None) -> int:
    """
    Identify robust option across scenarios
    
    Returns index of option with best worst-case performance
    """
    option_worst_scores = []
    
    for option in options:
        scenario_scores = []
        
        for scenario in scenarios:
            # Modify option performance based on scenario
            adjusted_option = option.copy()
            for key in adjusted_option:
                if key in scenario:
                    adjusted_option[key] *= scenario[key]
            
            score = self.score_adaptation_option(adjusted_option, weights)
            scenario_scores.append(score)
        
        # Worst case for this option
        worst_score = min(scenario_scores)
        option_worst_scores.append(worst_score)
    
    # Choose option with best worst-case
    return np.argmax(option_worst_scores)
def demonstrate_climate_risk_framework():
"""
Demonstrate climate risk assessment and adaptation framework
"""
print("="*70)
print("Climate Risk Assessment and Adaptation Framework")
print("Chapter 13: Climate Risk Assessment and Adaptation")
print("="*70)
np.random.seed(42)

# 1. Climate Risk Components
print("\n" + "-"*70)
print("1. Fundamental Climate Risk Assessment")
print("-"*70)

# High risk scenario
risk_high = ClimateRiskComponents(
    hazard=0.8,
    exposure=0.9,
    vulnerability=0.7
)

# Low risk scenario
risk_low = ClimateRiskComponents(
    hazard=0.3,
    exposure=0.5,
    vulnerability=0.4
)

print(f"High Risk Scenario:")
print(f"  Hazard: {risk_high.hazard:.2f}")
print(f"  Exposure: {risk_high.exposure:.2f}")
print(f"  Vulnerability: {risk_high.vulnerability:.2f}")
print(f"  → Total Risk: {risk_high.calculate_risk():.3f}")

print(f"\nLow Risk Scenario:")
print(f"  Hazard: {risk_low.hazard:.2f}")
print(f"  Exposure: {risk_low.exposure:.2f}")
print(f"  Vulnerability: {risk_low.vulnerability:.2f}")
print(f"  → Total Risk: {risk_low.calculate_risk():.3f}")

print(f"\nRisk Reduction: {((risk_high.calculate_risk() - risk_low.calculate_risk()) / risk_high.calculate_risk()) * 100:.1f}%")

# 2. Flood Susceptibility Mapping
print("\n" + "-"*70)
print("2. Flood Susceptibility Mapping")
print("-"*70)

flood_mapper = FloodSusceptibilityMapper()

# Generate synthetic training data
n_samples = 200
X_train = np.random.randn(n_samples, 14)

# Synthetic labels (higher susceptibility for low elevation, high rainfall, etc.)
y_train = (0.5 - 0.05 * X_train[:, 0] +  # Elevation (negative effect)
          0.1 * X_train[:, 7] +  # Precipitation (positive effect)
          0.08 * X_train[:, 10] +  # Impervious (positive effect)
          np.random.randn(n_samples) * 0.1)
y_train = 1 / (1 + np.exp(-y_train))  # Sigmoid

# Train model
flood_mapper.train_susceptibility_model(X_train, y_train)

# Test locations
location_high_risk = {
    'elevation': 5,
    'slope': 2,
    'twi': 15,
    'curvature': -0.5,
    'distance_to_river': 50,
    'drainage_density': 0.8,
    'stream_power_index': 10,
    'annual_precipitation': 1200,
    'rainfall_intensity': 80,
    'days_heavy_rain': 20,
    'impervious_fraction': 0.7,
    'vegetation_cover': 0.2,
    'soil_permeability': 0.2,
    'soil_moisture': 0.6
}

location_low_risk = {
    'elevation': 100,
    'slope': 15,
    'twi': 5,
    'curvature': 0.5,
    'distance_to_river': 2000,
    'drainage_density': 0.2,
    'stream_power_index': 2,
    'annual_precipitation': 600,
    'rainfall_intensity': 30,
    'days_heavy_rain': 5,
    'impervious_fraction': 0.1,
    'vegetation_cover': 0.9,
    'soil_permeability': 0.8,
    'soil_moisture': 0.2
}

X_test_high = flood_mapper.extract_flood_factors(location_high_risk)
X_test_low = flood_mapper.extract_flood_factors(location_low_risk)

susc_high = flood_mapper.predict_susceptibility(X_test_high.reshape(1, -1))[0]
susc_low = flood_mapper.predict_susceptibility(X_test_low.reshape(1, -1))[0]

print(f"Location 1 (High Risk Area):")
print(f"  Elevation: {location_high_risk['elevation']}m")
print(f"  Distance to river: {location_high_risk['distance_to_river']}m")
print(f"  Impervious fraction: {location_high_risk['impervious_fraction']:.1%}")
print(f"  → Flood susceptibility: {susc_high:.2%}")

print(f"\nLocation 2 (Low Risk Area):")
print(f"  Elevation: {location_low_risk['elevation']}m")
print(f"  Distance to river: {location_low_risk['distance_to_river']}m")
print(f"  Impervious fraction: {location_low_risk['impervious_fraction']:.1%}")
print(f"  → Flood susceptibility: {susc_low:.2%}")

# 3. Reinforcement Learning for Adaptation
print("\n" + "-"*70)
print("3. Adaptive Flood Management with RL")
print("-"*70)

rl_agent = AdaptiveFloodManagementRL()

print(f"Available Actions:")
for i, action in enumerate(rl_agent.actions):
    print(f"  {i}: {action.replace('_', ' ').title()}")

# Simulate learning
n_episodes = 50
total_rewards = []

for episode in range(n_episodes):
    # Initial state
    slr = np.random.uniform(0, 0.5)
    storm_prob = np.random.uniform(0.1, 0.4)
    exposure = np.random.uniform(1000, 5000)
    budget = np.random.uniform(10000, 50000)
    
    state = rl_agent.discretize_state(slr, storm_prob, exposure, budget)
    
    episode_reward = 0
    
    for step in range(20):
        # Select action
        action = rl_agent.select_action(state)
        
        # Simulate outcomes
        protection = np.random.uniform(50, 100)
        cost = np.random.uniform(5000, 20000)
        ecosystem = np.random.uniform(10, 50) if action == 1 else np.random.uniform(0, 20)
        carbon = np.random.uniform(100, 500) if action == 0 else np.random.uniform(0, 100)
        equity = np.random.uniform(30, 80)
        
        # Compute reward
        reward = rl_agent.compute_multi_objective_reward(
            action, protection, cost, ecosystem, carbon, equity
        )
        
        # Update state
        slr += 0.01
        next_state = rl_agent.discretize_state(slr, storm_prob, exposure, budget)
        
        # Q-learning update
        rl_agent.update_q_value(state, action, reward, next_state)
        
        episode_reward += reward
        state = next_state
    
    total_rewards.append(episode_reward)

print(f"\nRL Training Results:")
print(f"  Episodes: {n_episodes}")
print(f"  Initial average reward: {np.mean(total_rewards[:10]):.1f}")
print(f"  Final average reward: {np.mean(total_rewards[-10:]):.1f}")
print(f"  Improvement: {np.mean(total_rewards[-10:]) - np.mean(total_rewards[:10]):.1f}")

# 4. Infrastructure Resilience
print("\n" + "-"*70)
print("4. Infrastructure Resilience Assessment")
print("-"*70)

resilience_assessor = InfrastructureResilienceAssessor()

# Scenario 1: Major disruption, slow recovery
func_major = resilience_assessor.simulate_disruption(
    initial_functionality=1.0,
    disruption_magnitude=0.7,
    recovery_rate=0.05,
    days=100
)

# Scenario 2: Minor disruption, fast recovery
func_minor = resilience_assessor.simulate_disruption(
    initial_functionality=1.0,
    disruption_magnitude=0.3,
    recovery_rate=0.15,
    days=100
)

resilience_major = resilience_assessor.calculate_resilience(
    func_major, redundancy=0.1, cascade_prob=0.3
)
resilience_minor = resilience_assessor.calculate_resilience(
    func_minor, redundancy=0.4, cascade_prob=0.05
)

recovery_major = resilience_assessor.recovery_time(func_major, threshold=0.9)
recovery_minor = resilience_assessor.recovery_time(func_minor, threshold=0.9)

print(f"Scenario 1: Major Disruption")
print(f"  Initial functionality drop: 70%")
print(f"  Recovery rate: Slow (5%/day)")
print(f"  Resilience score: {resilience_major:.1f}")
print(f"  Recovery time: {recovery_major} days")

print(f"\nScenario 2: Minor Disruption")
print(f"  Initial functionality drop: 30%")
print(f"  Recovery rate: Fast (15%/day)")
print(f"  Resilience score: {resilience_minor:.1f}")
print(f"  Recovery time: {recovery_minor} days")

# 5. Financial Climate Risk
print("\n" + "-"*70)
print("5. Financial Climate Risk Assessment")
print("-"*70)

fin_risk_assessor = FinancialClimateRiskAssessor()

# Company A: High climate exposure
asset_value_a = 10_000_000
flood_prob_a = 0.05
revenue_a = 50_000_000
carbon_intensity_a = 0.8

physical_risk_a = fin_risk_assessor.assess_physical_risk(
    asset_value_a, flood_prob_a
)
transition_risk_a = fin_risk_assessor.assess_transition_risk(
    revenue_a, carbon_intensity_a
)
default_prob_a = fin_risk_assessor.predict_default_probability(
    financial_health=0.6,
    climate_risk_exposure=0.7,
    sector_vulnerability=0.6,
    geography_risk=0.5
)

print(f"Company A (High Climate Exposure):")
print(f"  Physical risk (expected loss): ${physical_risk_a:,.0f}")
print(f"  Transition risk (carbon cost): ${transition_risk_a:,.0f}")
print(f"  Default probability: {default_prob_a:.2%}")

# Company B: Low climate exposure
asset_value_b = 10_000_000
flood_prob_b = 0.01
revenue_b = 50_000_000
carbon_intensity_b = 0.2

physical_risk_b = fin_risk_assessor.assess_physical_risk(
    asset_value_b, flood_prob_b
)
transition_risk_b = fin_risk_assessor.assess_transition_risk(
    revenue_b, carbon_intensity_b
)
default_prob_b = fin_risk_assessor.predict_default_probability(
    financial_health=0.8,
    climate_risk_exposure=0.2,
    sector_vulnerability=0.3,
    geography_risk=0.2
)

print(f"\nCompany B (Low Climate Exposure):")
print(f"  Physical risk (expected loss): ${physical_risk_b:,.0f}")
print(f"  Transition risk (carbon cost): ${transition_risk_b:,.0f}")
print(f"  Default probability: {default_prob_b:.2%}")

# 6. Adaptation Pathway Optimization
print("\n" + "-"*70)
print("6. Multi-Criteria Adaptation Planning")
print("-"*70)

pathway_optimizer = AdaptationPathwayOptimizer()

# Define adaptation options
options = [
    {
        'name': 'Build seawall',
        'effectiveness': 0.9,
        'cost': 0.3,  # Higher cost = lower score
        'co_benefits': 0.2,
        'equity': 0.5,
        'flexibility': 0.3
    },
    {
        'name': 'Restore wetlands',
        'effectiveness': 0.7,
        'cost': 0.7,  # Lower cost = higher score
        'co_benefits': 0.9,
        'equity': 0.8,
        'flexibility': 0.6
    },
    {
        'name': 'Update building codes',
        'effectiveness': 0.6,
        'cost': 0.9,
        'co_benefits': 0.5,
        'equity': 0.7,
        'flexibility': 0.8
    },
    {
        'name': 'Improve drainage',
        'effectiveness': 0.5,
        'cost': 0.8,
        'co_benefits': 0.6,
        'equity': 0.6,
        'flexibility': 0.7
    }
]

# Equal weights
ranked = pathway_optimizer.rank_options(options)

print(f"Adaptation Option Rankings (Equal Weights):")
for rank, (idx, score) in enumerate(ranked, 1):
    print(f"  {rank}. {options[idx]['name']:25s} - Score: {score:.3f}")

# Prioritize effectiveness and equity
weights_equity = {
    'effectiveness': 0.3,
    'cost': 0.1,
    'co_benefits': 0.2,
    'equity': 0.3,
    'flexibility': 0.1
}

ranked_equity = pathway_optimizer.rank_options(options, weights_equity)

print(f"\nRankings (Prioritizing Effectiveness & Equity):")
for rank, (idx, score) in enumerate(ranked_equity, 1):
    print(f"  {rank}. {options[idx]['name']:25s} - Score: {score:.3f}")

# Visualize results
visualize_climate_risk_results(
    func_major, func_minor,
    total_rewards,
    [susc_high, susc_low],
    options, ranked
)

print("\n" + "="*70)
print("Climate risk framework demonstration complete!")
print("="*70)
def visualize_climate_risk_results(func_major, func_minor, rl_rewards,
susceptibility, options, rankings):
"""Visualize climate risk assessment results"""
fig = plt.figure(figsize=(16, 10))
days = len(func_major)
time = np.arange(days)

# Plot 1: Infrastructure resilience
ax1 = plt.subplot(2, 3, 1)
ax1.plot(time, func_major, 'r-', linewidth=2, label='Major Disruption')
ax1.plot(time, func_minor, 'g-', linewidth=2, label='Minor Disruption')
ax1.axhline(y=0.9, color='k', linestyle='--', alpha=0.5, label='90% Recovery')
ax1.fill_between(time, 0, func_major, alpha=0.2, color='red')
ax1.fill_between(time, 0, func_minor, alpha=0.2, color='green')
ax1.set_xlabel('Days After Disruption')
ax1.set_ylabel('Functionality')
ax1.set_title('Infrastructure Recovery Trajectories')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1.1])

# Plot 2: RL learning curve
ax2 = plt.subplot(2, 3, 2)
episodes = np.arange(len(rl_rewards))
ax2.plot(episodes, rl_rewards, 'b-', alpha=0.5, linewidth=1)

# Moving average
window = 5
if len(rl_rewards) >= window:
    moving_avg = np.convolve(rl_rewards, np.ones(window)/window, mode='valid')
    ax2.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label='Moving Average')

ax2.set_xlabel('Episode')
ax2.set_ylabel('Total Reward')
ax2.set_title('RL Adaptive Management Learning')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Flood susceptibility comparison
ax3 = plt.subplot(2, 3, 3)
locations = ['High Risk\nArea', 'Low Risk\nArea']
colors_risk = ['red', 'green']

bars = ax3.bar(locations, susceptibility, color=colors_risk,
              alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_ylabel('Flood Susceptibility')
ax3.set_title('Flood Susceptibility Assessment')
ax3.set_ylim([0, 1])
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars, susceptibility):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1%}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Adaptation option scores
ax4 = plt.subplot(2, 3, 4)
option_names = [options[idx]['name'] for idx, _ in rankings]
scores = [score for _, score in rankings]
colors_adapt = plt.cm.viridis(np.linspace(0.2, 0.8, len(option_names)))

bars4 = ax4.barh(option_names, scores, color=colors_adapt,
                 edgecolor='black', linewidth=1)
ax4.set_xlabel('Score')
ax4.set_title('Adaptation Option Rankings')
ax4.set_xlim([0, max(scores) * 1.2])
ax4.grid(True, alpha=0.3, axis='x')

# Add value labels
for bar, score in zip(bars4, scores):
    width = bar.get_width()
    ax4.text(width, bar.get_y() + bar.get_height()/2.,
            f'{score:.3f}', ha='left', va='center',
            fontweight='bold', fontsize=9)

# Plot 5: Risk components visualization
ax5 = plt.subplot(2, 3, 5)

# High risk scenario
hazard_high, exposure_high, vuln_high = 0.8, 0.9, 0.7
risk_high = hazard_high * exposure_high * vuln_high

# Low risk scenario
hazard_low, exposure_low, vuln_low = 0.3, 0.5, 0.4
risk_low = hazard_low * exposure_low * vuln_low

x = np.arange(2)
width = 0.2

ax5.bar(x - 1.5*width, [hazard_high, hazard_low], width,
       label='Hazard', color='red', alpha=0.7)
ax5.bar(x - 0.5*width, [exposure_high, exposure_low], width,
       label='Exposure', color='orange', alpha=0.7)
ax5.bar(x + 0.5*width, [vuln_high, vuln_low], width,
       label='Vulnerability', color='yellow', alpha=0.7)
ax5.bar(x + 1.5*width, [risk_high, risk_low], width,
       label='Total Risk', color='darkred', alpha=0.7)

ax5.set_ylabel('Value')
ax5.set_title('Risk Component Decomposition')
ax5.set_xticks(x)
ax5.set_xticklabels(['High Risk\nScenario', 'Low Risk\nScenario'])
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')
ax5.set_ylim([0, 1])

# Plot 6: Resilience comparison
ax6 = plt.subplot(2, 3, 6)

from climate_risk_framework import InfrastructureResilienceAssessor
assessor = InfrastructureResilienceAssessor()

resilience_major = assessor.calculate_resilience(
    func_major, redundancy=0.1, cascade_prob=0.3
)
resilience_minor = assessor.calculate_resilience(
    func_minor, redundancy=0.4, cascade_prob=0.05
)

scenarios = ['Major\nDisruption', 'Minor\nDisruption']
resilience_values = [resilience_major, resilience_minor]
colors_res = ['red', 'green']

bars6 = ax6.bar(scenarios, resilience_values, color=colors_res,
                alpha=0.7, edgecolor='black', linewidth=2)
ax6.set_ylabel('Resilience Score')
ax6.set_title('Infrastructure Resilience Comparison')
ax6.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars6, resilience_values):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('climate_risk_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nResults saved to 'climate_risk_results.png'")
if name == "main":
demonstrate_climate_risk_framework()
