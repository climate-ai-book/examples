"""
Regional and Seasonal Climate Prediction Framework
Implements ML methods for sub-seasonal to seasonal forecasting:

LSTM for sub-seasonal temperature prediction
Ensemble methods for seasonal precipitation
Drought forecasting with climate indices
Multi-model ensemble integration

Based on Chapter 14: Regional and Seasonal Climate Prediction
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')
@dataclass
class ClimateIndices:
"""Climate indices for seasonal prediction"""
nino34: float  # ENSO index
mjo_rmm1: float  # MJO Real-time Multivariate Index 1
mjo_rmm2: float  # MJO Real-time Multivariate Index 2
nao: float  # North Atlantic Oscillation
pdo: float  # Pacific Decadal Oscillation
def mjo_amplitude(self) -> float:
    """Calculate MJO amplitude"""
    return np.sqrt(self.mjo_rmm1**2 + self.mjo_rmm2**2)

def mjo_phase(self) -> float:
    """Calculate MJO phase (1-8)"""
    angle = np.arctan2(self.mjo_rmm2, self.mjo_rmm1)
    phase = (angle * 4 / np.pi) + 4.5
    return int(phase % 8) + 1
class SubSeasonalLSTMForecaster:
"""
LSTM-based sub-seasonal climate forecasting
Predicts temperature/precipitation at 1-4 week lead times
"""

def __init__(self, n_features: int = 15, n_lead_weeks: int = 4):
    """
    Initialize LSTM forecaster
    
    Args:
        n_features: Number of input features
        n_lead_weeks: Number of weeks to forecast
    """
    self.n_features = n_features
    self.n_lead_weeks = n_lead_weeks
    
    # Simplified LSTM state (placeholder for actual implementation)
    self.cell_state = None
    self.hidden_state = None
    self.weights_trained = False
    
def extract_features(self, indices: ClimateIndices,
                    week_of_year: int,
                    recent_temp: np.ndarray) -> np.ndarray:
    """
    Extract features for sub-seasonal prediction
    
    Features include climate indices, seasonal cycle, and recent observations
    """
    features = []
    
    # Climate indices
    features.append(indices.nino34)
    features.append(indices.nino34**2)  # Nonlinear ENSO effect
    features.append(indices.mjo_rmm1)
    features.append(indices.mjo_rmm2)
    features.append(indices.mjo_amplitude())
    features.append(indices.nao)
    features.append(indices.pdo)
    
    # Seasonal cycle
    features.append(np.sin(2 * np.pi * week_of_year / 52))
    features.append(np.cos(2 * np.pi * week_of_year / 52))
    
    # Recent temperature anomalies
    if len(recent_temp) >= 4:
        features.append(recent_temp[-1])  # Last week
        features.append(recent_temp[-2])  # 2 weeks ago
        features.append(np.mean(recent_temp[-4:]))  # 4-week mean
        features.append(recent_temp[-1] - recent_temp[-2])  # Trend
    else:
        features.extend([0, 0, 0, 0])
    
    # Interaction terms
    features.append(indices.nino34 * indices.mjo_amplitude())
    features.append(indices.nao * np.sin(2 * np.pi * week_of_year / 52))
    
    return np.array(features[:self.n_features])

def simple_lstm_forecast(self, features: np.ndarray) -> np.ndarray:
    """
    Simplified LSTM forecast (placeholder)
    
    In practice, use trained TensorFlow/PyTorch LSTM model
    """
    # Simplified linear model with decay for demonstration
    forecasts = np.zeros(self.n_lead_weeks)
    
    # Base forecast from features
    base_forecast = 0.3 * features[0] + 0.2 * features[10] + 0.1 * features[7]
    
    # Forecast decays with lead time (skill decreases)
    for week in range(self.n_lead_weeks):
        decay = np.exp(-week / 3)
        forecasts[week] = base_forecast * decay
    
    return forecasts

def forecast_with_uncertainty(self, features: np.ndarray,
                              n_ensemble: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate ensemble forecast with uncertainty
    
    Returns mean and standard deviation for each lead time
    """
    ensemble = np.zeros((n_ensemble, self.n_lead_weeks))
    
    for i in range(n_ensemble):
        # Add perturbations to features
        perturbed_features = features + np.random.randn(len(features)) * 0.1
        ensemble[i] = self.simple_lstm_forecast(perturbed_features)
    
    forecast_mean = ensemble.mean(axis=0)
    forecast_std = ensemble.std(axis=0)
    
    return forecast_mean, forecast_std
class SeasonalPrecipitationPredictor:
"""
Seasonal precipitation prediction using climate indices
Probabilistic forecasts of above/near/below normal categories
"""

def __init__(self):
    """Initialize seasonal precipitation predictor"""
    self.categories = ['Below Normal', 'Near Normal', 'Above Normal']
    self.n_categories = len(self.categories)
    
def extract_seasonal_predictors(self, indices: ClimateIndices,
                                season: str,
                                sst_anomaly: float) -> np.ndarray:
    """
    Extract predictors for seasonal precipitation
    
    Args:
        indices: Climate indices
        season: Season name (DJF, MAM, JJA, SON)
        sst_anomaly: Regional SST anomaly
    """
    predictors = []
    
    # ENSO influence (strongest predictor for many regions)
    predictors.append(indices.nino34)
    predictors.append(indices.nino34**2)  # Nonlinear effects
    predictors.append(indices.nino34**3)  # Asymmetry
    
    # MJO (sub-seasonal modulation)
    predictors.append(indices.mjo_amplitude())
    
    # Other large-scale patterns
    predictors.append(indices.nao)
    predictors.append(indices.pdo)
    
    # SST patterns
    predictors.append(sst_anomaly)
    predictors.append(sst_anomaly * indices.nino34)  # Interaction
    
    # Seasonal encoding
    season_map = {'DJF': 0, 'MAM': 1, 'JJA': 2, 'SON': 3}
    season_idx = season_map.get(season, 0)
    predictors.append(np.sin(2 * np.pi * season_idx / 4))
    predictors.append(np.cos(2 * np.pi * season_idx / 4))
    
    return np.array(predictors)

def predict_probability(self, predictors: np.ndarray) -> np.ndarray:
    """
    Predict probability of each precipitation category
    
    Returns probabilities [Below, Near, Above]
    """
    # Simplified logistic model (in practice, use trained classifier)
    
    # Score based on ENSO (primary predictor)
    enso_score = predictors[0]
    
    # Adjust for other predictors
    adjustment = 0.1 * predictors[4] + 0.05 * predictors[6]
    
    total_score = enso_score + adjustment
    
    # Map score to probabilities
    if total_score > 0.5:  # Strong El Niño-like
        probs = [0.15, 0.25, 0.60]
    elif total_score > 0.2:
        probs = [0.20, 0.35, 0.45]
    elif total_score > -0.2:
        probs = [0.30, 0.40, 0.30]
    elif total_score > -0.5:
        probs = [0.45, 0.35, 0.20]
    else:  # Strong La Niña-like
        probs = [0.60, 0.25, 0.15]
    
    return np.array(probs)

def forecast_tercile_category(self, predictors: np.ndarray) -> str:
    """
    Forecast most likely tercile category
    """
    probs = self.predict_probability(predictors)
    category_idx = np.argmax(probs)
    return self.categories[category_idx]
class DroughtForecaster:
"""
Seasonal drought forecasting using ML
Predicts drought indices (SPI, SPEI) at seasonal lead times
"""

def __init__(self):
    """Initialize drought forecaster"""
    self.severity_categories = [
        'Wet', 'Normal', 'Moderate Drought',
        'Severe Drought', 'Extreme Drought'
    ]
    
def calculate_spi(self, precipitation: np.ndarray,
                 timescale: int = 3) -> float:
    """
    Calculate Standardized Precipitation Index
    
    Args:
        precipitation: Historical precipitation (months)
        timescale: Accumulation period (months)
    """
    if len(precipitation) < timescale:
        return 0
    
    # Sum over timescale
    precip_sum = np.sum(precipitation[-timescale:])
    
    # Normalize (simplified - should use fitted distribution)
    mean_precip = np.mean(precipitation)
    std_precip = np.std(precipitation)
    
    if std_precip > 0:
        spi = (precip_sum / timescale - mean_precip) / std_precip
    else:
        spi = 0
    
    return spi

def forecast_drought_onset(self, indices: ClimateIndices,
                          recent_precipitation: np.ndarray,
                          soil_moisture: float,
                          vegetation_index: float) -> Dict:
    """
    Forecast drought probability and severity
    
    Integrates climate indices, recent precip, soil moisture, and vegetation
    """
    # Current drought status
    current_spi = self.calculate_spi(recent_precipitation, timescale=3)
    
    # Climate index influence
    climate_score = -0.7 * indices.nino34 + 0.3 * indices.pdo
    
    # Persistence from current conditions
    persistence_score = current_spi * 0.5
    
    # Soil moisture and vegetation indicators
    surface_score = -0.3 * (soil_moisture - 0.5) - 0.2 * (vegetation_index - 0.5)
    
    # Combined forecast
    forecast_spi = persistence_score + 0.3 * climate_score + 0.2 * surface_score
    
    # Classify severity
    if forecast_spi >= 1.0:
        severity = 'Wet'
        probability = 0.7
    elif forecast_spi >= -0.5:
        severity = 'Normal'
        probability = 0.6
    elif forecast_spi >= -1.0:
        severity = 'Moderate Drought'
        probability = 0.6
    elif forecast_spi >= -1.5:
        severity = 'Severe Drought'
        probability = 0.6
    else:
        severity = 'Extreme Drought'
        probability = 0.7
    
    return {
        'forecast_spi': forecast_spi,
        'severity': severity,
        'probability': probability,
        'current_spi': current_spi
    }
class MultiModelEnsemble:
"""
Multi-model ensemble for seasonal prediction
Combines forecasts from multiple models with adaptive weighting
"""

def __init__(self, n_models: int = 5):
    """
    Initialize multi-model ensemble
    
    Args:
        n_models: Number of models in ensemble
    """
    self.n_models = n_models
    self.model_names = [f'Model_{i+1}' for i in range(n_models)]
    
def simulate_model_forecasts(self, true_value: float,
                             lead_time: int) -> np.ndarray:
    """
    Simulate forecasts from multiple models
    
    Models have different biases and skill levels
    """
    forecasts = np.zeros(self.n_models)
    
    # Skill degrades with lead time
    skill_decay = np.exp(-lead_time / 4)
    
    for i in range(self.n_models):
        # Each model has different bias and error
        bias = np.random.uniform(-0.5, 0.5)
        error = np.random.randn() * (1 - skill_decay) * 2
        
        forecasts[i] = true_value + bias + error
    
    return forecasts

def calculate_adaptive_weights(self, forecasts: np.ndarray,
                               historical_skill: np.ndarray,
                               lead_time: int) -> np.ndarray:
    """
    Calculate adaptive weights based on historical skill
    
    Weights vary with lead time and conditions
    """
    # Base weights from historical skill
    base_weights = historical_skill / historical_skill.sum()
    
    # Adjust for lead time (some models better at longer leads)
    lead_adjustment = np.ones(self.n_models)
    lead_adjustment[0] *= 1 + 0.1 * lead_time  # Model 1 better at long leads
    lead_adjustment[1] *= 1 - 0.05 * lead_time  # Model 2 better at short leads
    
    # Combine
    weights = base_weights * lead_adjustment
    weights /= weights.sum()  # Normalize
    
    return weights

def ensemble_mean(self, forecasts: np.ndarray,
                 weights: Optional[np.ndarray] = None) -> float:
    """
    Compute weighted ensemble mean
    """
    if weights is None:
        weights = np.ones(self.n_models) / self.n_models
    
    return np.sum(forecasts * weights)

def ensemble_uncertainty(self, forecasts: np.ndarray) -> float:
    """
    Quantify ensemble spread as uncertainty measure
    """
    return np.std(forecasts)
def demonstrate_seasonal_prediction_framework():
"""
Demonstrate seasonal climate prediction framework
"""
print("="*70)
print("Regional and Seasonal Climate Prediction Framework")
print("Chapter 14: Regional and Seasonal Climate Prediction")
print("="*70)
np.random.seed(42)

# 1. Climate Indices Setup
print("\n" + "-"*70)
print("1. Climate Indices for Seasonal Prediction")
print("-"*70)

# El Niño scenario
indices_elnino = ClimateIndices(
    nino34=1.5,
    mjo_rmm1=1.2,
    mjo_rmm2=-0.8,
    nao=0.5,
    pdo=0.3
)

# La Niña scenario
indices_lanina = ClimateIndices(
    nino34=-1.2,
    mjo_rmm1=0.5,
    mjo_rmm2=1.5,
    nao=-0.8,
    pdo=-0.5
)

print(f"El Niño Scenario:")
print(f"  Niño 3.4: {indices_elnino.nino34:.2f}°C")
print(f"  MJO Amplitude: {indices_elnino.mjo_amplitude():.2f}")
print(f"  MJO Phase: {indices_elnino.mjo_phase()}")
print(f"  NAO: {indices_elnino.nao:.2f}")

print(f"\nLa Niña Scenario:")
print(f"  Niño 3.4: {indices_lanina.nino34:.2f}°C")
print(f"  MJO Amplitude: {indices_lanina.mjo_amplitude():.2f}")
print(f"  MJO Phase: {indices_lanina.mjo_phase()}")
print(f"  NAO: {indices_lanina.nao:.2f}")

# 2. Sub-Seasonal Forecasting
print("\n" + "-"*70)
print("2. Sub-Seasonal Temperature Forecasting (LSTM)")
print("-"*70)

lstm_forecaster = SubSeasonalLSTMForecaster()

# Extract features
week_of_year = 10
recent_temp = np.array([0.5, 0.3, 0.8, 1.2])

features = lstm_forecaster.extract_features(
    indices_elnino, week_of_year, recent_temp
)

# Generate forecast
forecast_mean, forecast_std = lstm_forecaster.forecast_with_uncertainty(features)

print(f"Input Features:")
print(f"  Week of year: {week_of_year}")
print(f"  Recent temperature trend: {recent_temp[-1] - recent_temp[-2]:.2f}°C")
print(f"  Number of features: {len(features)}")

print(f"\nSub-Seasonal Temperature Forecast (anomaly):")
for week in range(len(forecast_mean)):
    print(f"  Week {week+1}: {forecast_mean[week]:+.2f} ± {forecast_std[week]:.2f}°C")

# 3. Seasonal Precipitation Prediction
print("\n" + "-"*70)
print("3. Seasonal Precipitation Prediction")
print("-"*70)

precip_predictor = SeasonalPrecipitationPredictor()

# Winter (DJF) forecasts
season = 'DJF'
sst_anomaly = 0.8

# El Niño case
predictors_elnino = precip_predictor.extract_seasonal_predictors(
    indices_elnino, season, sst_anomaly
)
probs_elnino = precip_predictor.predict_probability(predictors_elnino)
category_elnino = precip_predictor.forecast_tercile_category(predictors_elnino)

# La Niña case
predictors_lanina = precip_predictor.extract_seasonal_predictors(
    indices_lanina, season, -sst_anomaly
)
probs_lanina = precip_predictor.predict_probability(predictors_lanina)
category_lanina = precip_predictor.forecast_tercile_category(predictors_lanina)

print(f"Winter (DJF) Precipitation Forecast:")
print(f"\nEl Niño Conditions:")
print(f"  Most likely: {category_elnino}")
for i, cat in enumerate(precip_predictor.categories):
    print(f"  {cat:15s}: {probs_elnino[i]*100:5.1f}%")

print(f"\nLa Niña Conditions:")
print(f"  Most likely: {category_lanina}")
for i, cat in enumerate(precip_predictor.categories):
    print(f"  {cat:15s}: {probs_lanina[i]*100:5.1f}%")

# 4. Drought Forecasting
print("\n" + "-"*70)
print("4. Seasonal Drought Forecasting")
print("-"*70)

drought_forecaster = DroughtForecaster()

# Simulate recent precipitation (dry conditions)
recent_precipitation = np.array([30, 25, 20, 18, 15, 12])
soil_moisture = 0.25
vegetation_index = 0.35

drought_forecast = drought_forecaster.forecast_drought_onset(
    indices_lanina,  # La Niña often dry in some regions
    recent_precipitation,
    soil_moisture,
    vegetation_index
)

print(f"Drought Forecast:")
print(f"  Current SPI-3: {drought_forecast['current_spi']:.2f}")
print(f"  Forecast SPI-3: {drought_forecast['forecast_spi']:.2f}")
print(f"  Severity: {drought_forecast['severity']}")
print(f"  Confidence: {drought_forecast['probability']*100:.0f}%")

# 5. Multi-Model Ensemble
print("\n" + "-"*70)
print("5. Multi-Model Ensemble Integration")
print("-"*70)

mme = MultiModelEnsemble(n_models=5)

# Simulate for different lead times
true_temp_anomaly = 1.0
lead_times = [1, 2, 3, 4]

print(f"True temperature anomaly: {true_temp_anomaly:+.2f}°C")
print(f"\nEnsemble Forecasts by Lead Time:")

ensemble_results = []

for lead in lead_times:
    # Simulate model forecasts
    forecasts = mme.simulate_model_forecasts(true_temp_anomaly, lead)
    
    # Historical skill (simplified)
    historical_skill = np.array([0.8, 0.75, 0.85, 0.7, 0.78])
    
    # Calculate adaptive weights
    weights = mme.calculate_adaptive_weights(forecasts, historical_skill, lead)
    
    # Ensemble mean and spread
    ens_mean = mme.ensemble_mean(forecasts, weights)
    ens_spread = mme.ensemble_uncertainty(forecasts)
    
    ensemble_results.append({
        'lead': lead,
        'forecasts': forecasts,
        'mean': ens_mean,
        'spread': ens_spread
    })
    
    print(f"\nLead Time: {lead} week(s)")
    print(f"  Individual models: {', '.join([f'{f:+.2f}' for f in forecasts])}°C")
    print(f"  Ensemble mean: {ens_mean:+.2f}°C")
    print(f"  Ensemble spread: {ens_spread:.2f}°C")
    print(f"  Error: {abs(ens_mean - true_temp_anomaly):.2f}°C")

# Visualize results
visualize_seasonal_prediction_results(
    forecast_mean, forecast_std,
    probs_elnino, probs_lanina,
    precip_predictor.categories,
    ensemble_results,
    true_temp_anomaly
)

print("\n" + "="*70)
print("Seasonal prediction framework demonstration complete!")
print("="*70)
def visualize_seasonal_prediction_results(subseasonal_mean, subseasonal_std,
probs_elnino, probs_lanina, categories,
ensemble_results, true_value):
"""Visualize seasonal prediction results"""
fig = plt.figure(figsize=(16, 10))
# Plot 1: Sub-seasonal forecast with uncertainty
ax1 = plt.subplot(2, 3, 1)
weeks = np.arange(1, len(subseasonal_mean) + 1)

ax1.plot(weeks, subseasonal_mean, 'b-o', linewidth=2, markersize=8, label='Forecast')
ax1.fill_between(weeks,
                 subseasonal_mean - subseasonal_std,
                 subseasonal_mean + subseasonal_std,
                 alpha=0.3, color='blue', label='±1 Std Dev')
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax1.set_xlabel('Lead Time (weeks)')
ax1.set_ylabel('Temperature Anomaly (°C)')
ax1.set_title('Sub-Seasonal Temperature Forecast')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Precipitation probabilities (El Niño)
ax2 = plt.subplot(2, 3, 2)
colors_precip = ['brown', 'gray', 'blue']

bars2 = ax2.bar(categories, probs_elnino, color=colors_precip,
                alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_ylabel('Probability')
ax2.set_title('Precipitation Forecast (El Niño)')
ax2.set_ylim([0, 1])
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, prob in zip(bars2, probs_elnino):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{prob*100:.0f}%', ha='center', va='bottom', fontweight='bold')

# Plot 3: Precipitation probabilities (La Niña)
ax3 = plt.subplot(2, 3, 3)

bars3 = ax3.bar(categories, probs_lanina, color=colors_precip,
                alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_ylabel('Probability')
ax3.set_title('Precipitation Forecast (La Niña)')
ax3.set_ylim([0, 1])
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, prob in zip(bars3, probs_lanina):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{prob*100:.0f}%', ha='center', va='bottom', fontweight='bold')

# Plot 4: Multi-model ensemble spread
ax4 = plt.subplot(2, 3, 4)
lead_times = [r['lead'] for r in ensemble_results]
ens_means = [r['mean'] for r in ensemble_results]
ens_spreads = [r['spread'] for r in ensemble_results]

ax4.errorbar(lead_times, ens_means, yerr=ens_spreads,
            fmt='go-', linewidth=2, markersize=10,
            capsize=5, capthick=2, label='Ensemble Mean ± Spread')
ax4.axhline(y=true_value, color='r', linestyle='--',
           linewidth=2, label='True Value')
ax4.set_xlabel('Lead Time (weeks)')
ax4.set_ylabel('Temperature Anomaly (°C)')
ax4.set_title('Multi-Model Ensemble Forecast')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Individual model forecasts (Week 2)
ax5 = plt.subplot(2, 3, 5)
week2_forecasts = ensemble_results[1]['forecasts']
model_names = [f'M{i+1}' for i in range(len(week2_forecasts))]
colors_models = plt.cm.viridis(np.linspace(0.2, 0.8, len(week2_forecasts)))

bars5 = ax5.bar(model_names, week2_forecasts, color=colors_models,
                alpha=0.7, edgecolor='black', linewidth=2)
ax5.axhline(y=true_value, color='r', linestyle='--',
           linewidth=2, label='True Value')
ax5.axhline(y=ensemble_results[1]['mean'], color='green',
           linestyle='-', linewidth=2, label='Ensemble Mean')
ax5.set_ylabel('Temperature Anomaly (°C)')
ax5.set_title('Individual Model Forecasts (Week 2)')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Forecast error vs lead time
ax6 = plt.subplot(2, 3, 6)
errors = [abs(r['mean'] - true_value) for r in ensemble_results]
spreads = [r['spread'] for r in ensemble_results]

ax6.plot(lead_times, errors, 'ro-', linewidth=2, markersize=8,
        label='Forecast Error')
ax6.plot(lead_times, spreads, 'bs-', linewidth=2, markersize=8,
        label='Ensemble Spread')
ax6.set_xlabel('Lead Time (weeks)')
ax6.set_ylabel('Error / Spread (°C)')
ax6.set_title('Forecast Error and Ensemble Spread')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('seasonal_prediction_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nResults saved to 'seasonal_prediction_results.png'")
if name == "main":
demonstrate_seasonal_prediction_framework()
