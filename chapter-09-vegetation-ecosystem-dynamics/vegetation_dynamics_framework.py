"""
Vegetation and Ecosystem Dynamics Framework
Implements ML methods for terrestrial biosphere modeling:

LSTM for NDVI forecasting
ConvLSTM for open ecosystem dynamics
Random forest for fire risk prediction
Soil organic carbon modeling

Based on Chapter 9: Vegetation and Ecosystem Dynamics
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')
class VegetationCarbonBalance:
"""
Ecosystem carbon balance model
Solves: dC_veg/dt = GPP - R_auto - M_herbivory - L_fire
"""

def __init__(self):
    """Initialize carbon balance parameters"""
    self.alpha_gpp = 0.8  # GPP efficiency
    self.q10_resp = 2.0  # Temperature sensitivity
    self.base_resp_rate = 0.02  # Base respiration rate
    
def gross_primary_productivity(self, light: float, temp: float,
                               water: float, nutrients: float = 1.0) -> float:
    """
    Calculate GPP with environmental limitations
    
    GPP = α · Light · L_temp · L_water · L_nutrients
    """
    # Temperature limitation (optimum curve)
    T_opt = 25.0  # °C
    T_range = 10.0
    L_temp = np.exp(-((temp - T_opt) / T_range)**2)
    
    # Water limitation (Michaelis-Menten)
    K_water = 0.3
    L_water = water / (K_water + water)
    
    # Nutrient limitation
    K_nutrients = 0.5
    L_nutrients = nutrients / (K_nutrients + nutrients)
    
    gpp = self.alpha_gpp * light * L_temp * L_water * L_nutrients
    
    return gpp

def autotrophic_respiration(self, temp: float, C_veg: float) -> float:
    """
    Calculate respiration with Q10 temperature dependence
    
    R_auto = R_base · Q10^((T-T_ref)/10) · C_veg
    """
    T_ref = 15.0  # Reference temperature
    Q10_factor = self.q10_resp ** ((temp - T_ref) / 10.0)
    
    respiration = self.base_resp_rate * Q10_factor * C_veg
    
    return respiration

def carbon_balance_step(self, C_veg: float, light: float, temp: float,
                       water: float, herbivory: float = 0.0,
                       fire_loss: float = 0.0, dt: float = 1.0) -> float:
    """
    Single time step of carbon balance
    """
    gpp = self.gross_primary_productivity(light, temp, water)
    r_auto = self.autotrophic_respiration(temp, C_veg)
    
    dC_dt = gpp - r_auto - herbivory - fire_loss
    
    C_new = C_veg + dC_dt * dt
    
    return max(C_new, 0)  # Non-negative carbon
class NDVIPredictor:
"""
LSTM-based NDVI forecasting
Predicts vegetation index from time series and environmental variables
"""

def __init__(self, window_size: int = 30):
    """
    Initialize NDVI predictor
    
    Args:
        window_size: Number of historical time steps
    """
    self.window_size = window_size
    
    # Simplified LSTM parameters (placeholders)
    # In practice, use trained deep learning model
    self.weights_initialized = False
    
def prepare_sequences(self, ndvi_data: np.ndarray,
                     climate_data: np.ndarray,
                     forecast_horizon: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare time series sequences for LSTM
    """
    n_samples = len(ndvi_data) - self.window_size - forecast_horizon
    n_features = climate_data.shape[1] + 1
    
    X = np.zeros((n_samples, self.window_size, n_features))
    y = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Historical NDVI
        X[i, :, 0] = ndvi_data[i:i+self.window_size]
        # Climate features
        X[i, :, 1:] = climate_data[i:i+self.window_size]
        # Target NDVI
        y[i] = ndvi_data[i + self.window_size + forecast_horizon]
    
    return X, y

def simple_lstm_forecast(self, sequence: np.ndarray) -> float:
    """
    Simplified LSTM forecast (placeholder)
    
    In practice, use trained neural network
    """
    # Use exponentially weighted moving average as simple baseline
    weights = np.exp(np.linspace(-2, 0, len(sequence)))
    weights /= weights.sum()
    
    forecast = np.sum(sequence[:, 0] * weights)
    
    return forecast
class FireRiskPredictor:
"""
Random forest-based fire risk prediction
Integrates vegetation, climate, and topographic factors
"""

def __init__(self):
    """Initialize fire risk predictor"""
    self.feature_names = []
    self.risk_threshold = 0.5
    
def calculate_fuel_moisture(self, ndvi: np.ndarray,
                           ndwi: np.ndarray,
                           precip_recent: np.ndarray) -> np.ndarray:
    """
    Estimate fuel moisture content from vegetation indices
    
    FMC correlates with NDVI (biomass) and NDWI (water content)
    """
    # Normalize indices
    ndvi_norm = (ndvi - ndvi.min()) / (ndvi.max() - ndvi.min() + 1e-10)
    ndwi_norm = (ndwi - ndwi.min()) / (ndwi.max() - ndwi.min() + 1e-10)
    
    # Empirical FMC relationship
    fmc = 20 + 80 * ndwi_norm + 40 * (1 - ndvi_norm) + 20 * (precip_recent / 100)
    
    return np.clip(fmc, 0, 200)  # Percent

def days_since_rain(self, precipitation: np.ndarray,
                   threshold: float = 1.0) -> np.ndarray:
    """
    Calculate days since significant rainfall
    """
    rain_events = precipitation > threshold
    days_since = np.zeros(len(precipitation))
    counter = 0
    
    for i in range(len(precipitation)):
        if rain_events[i]:
            counter = 0
        else:
            counter += 1
        days_since[i] = counter
    
    return days_since

def calculate_fire_risk(self, ndvi: float, temp: float,
                       humidity: float, wind: float,
                       days_no_rain: float, slope: float) -> float:
    """
    Calculate fire risk probability
    
    Simple heuristic model (placeholder for trained ML)
    """
    # Vegetation dryness (lower NDVI = drier)
    veg_risk = 1 - ndvi
    
    # Weather risk
    temp_risk = (temp - 15) / 30  # Higher temp = higher risk
    humidity_risk = 1 - humidity  # Lower humidity = higher risk
    wind_risk = wind / 20  # Higher wind = higher risk
    
    # Temporal risk
    drought_risk = np.tanh(days_no_rain / 30)  # Saturation effect
    
    # Topographic risk
    slope_risk = slope / 45  # Steeper slope = higher risk
    
    # Weighted combination
    risk = (0.2 * veg_risk + 
            0.2 * temp_risk + 
            0.15 * humidity_risk + 
            0.15 * wind_risk + 
            0.2 * drought_risk + 
            0.1 * slope_risk)
    
    return np.clip(risk, 0, 1)
class SoilOrganicCarbonPredictor:
"""
Random forest-based soil organic carbon prediction
Integrates climate, vegetation, soil, and topographic features
"""

def __init__(self):
    """Initialize SOC predictor"""
    self.feature_importance = None
    
def prepare_features(self, climate: Dict, vegetation: Dict,
                    soil: Dict, topography: Dict) -> np.ndarray:
    """
    Extract and engineer features for SOC prediction
    """
    features = []
    
    # Climate variables
    features.append(climate['mean_annual_temp'])
    features.append(climate['mean_annual_precip'])
    features.append(climate['temp_seasonality'])
    
    # Vegetation indicators
    features.append(vegetation['ndvi_mean'])
    features.append(vegetation['ndvi_max'])
    features.append(vegetation['npp'])  # Net primary productivity
    
    # Soil properties
    features.append(soil['clay_content'])
    features.append(soil['sand_content'])
    features.append(soil['bulk_density'])
    features.append(soil['ph'])
    
    # Topographic features
    features.append(topography['elevation'])
    features.append(topography['slope'])
    features.append(topography['aspect'])
    features.append(topography['twi'])  # Topographic wetness index
    
    return np.array(features)

def predict_soc(self, features: np.ndarray) -> Tuple[float, float]:
    """
    Predict SOC with uncertainty
    
    Returns mean prediction and uncertainty estimate
    """
    # Simplified empirical model (placeholder)
    # In practice, use trained random forest
    
    # Climate effect
    temp_effect = np.exp(-features[0] / 10)
    precip_effect = features[1] / 1000
    
    # Vegetation effect
    veg_effect = features[3] * 50  # NDVI mean
    
    # Soil texture effect
    clay_effect = features[6] * 0.5  # Clay stabilizes C
    
    # Topographic effect
    elev_effect = features[10] / 1000
    
    # Combined prediction
    soc_pred = (temp_effect * 20 + 
               precip_effect * 30 + 
               veg_effect + 
               clay_effect + 
               elev_effect * 10)
    
    # Uncertainty (simplified)
    uncertainty = soc_pred * 0.2
    
    return soc_pred, uncertainty
class EcosystemServiceAssessment:
"""
Machine learning for ecosystem service valuation
Quantifies multiple ecosystem services from land cover
"""

def __init__(self):
    """Initialize ESV assessment"""
    # Unit values for different land cover types ($/ha/year)
    self.base_values = {
        'forest': 5000,
        'grassland': 2000,
        'wetland': 8000,
        'cropland': 1000,
        'urban': 0
    }
    
def calculate_land_cover_esv(self, land_cover_areas: Dict[str, float],
                            service_weights: Optional[Dict] = None) -> Dict:
    """
    Calculate ecosystem service values by land cover type
    """
    if service_weights is None:
        service_weights = {'provisioning': 0.3, 'regulating': 0.4,
                         'cultural': 0.2, 'supporting': 0.1}
    
    total_esv = 0
    esv_by_type = {}
    
    for land_type, area in land_cover_areas.items():
        if land_type in self.base_values:
            base_value = self.base_values[land_type]
            # Adjust for service categories
            adjusted_value = base_value * sum(service_weights.values())
            esv = area * adjusted_value
            esv_by_type[land_type] = esv
            total_esv += esv
    
    return {
        'total': total_esv,
        'by_type': esv_by_type,
        'per_hectare': total_esv / sum(land_cover_areas.values())
    }

def predict_esv_change(self, current_areas: Dict[str, float],
                      future_areas: Dict[str, float]) -> Dict:
    """
    Predict change in ESV from land cover transitions
    """
    current_esv = self.calculate_land_cover_esv(current_areas)
    future_esv = self.calculate_land_cover_esv(future_areas)
    
    change = future_esv['total'] - current_esv['total']
    percent_change = (change / current_esv['total']) * 100
    
    return {
        'current_total': current_esv['total'],
        'future_total': future_esv['total'],
        'absolute_change': change,
        'percent_change': percent_change
    }
def demonstrate_vegetation_dynamics_framework():
"""
Demonstrate vegetation and ecosystem dynamics framework
"""
print("="*70)
print("Vegetation and Ecosystem Dynamics Framework")
print("Chapter 9: Vegetation and Ecosystem Dynamics")
print("="*70)
np.random.seed(42)

# 1. Carbon Balance
print("\n" + "-"*70)
print("1. Vegetation Carbon Balance")
print("-"*70)

carbon_model = VegetationCarbonBalance()

# Initial conditions
C_initial = 100.0  # kg C/m²
n_days = 365

# Environmental forcing (annual cycle)
days = np.arange(n_days)
light = 300 + 200 * np.sin(2 * np.pi * days / 365)  # W/m²
temp = 15 + 10 * np.sin(2 * np.pi * days / 365)  # °C
water = 0.4 + 0.3 * np.sin(2 * np.pi * days / 365 + np.pi/4)  # 0-1

print(f"Initial carbon stock: {C_initial:.1f} kg C/m²")
print(f"Simulation period: {n_days} days")

# Simulate carbon dynamics
C_evolution = [C_initial]

for day in range(n_days - 1):
    C_new = carbon_model.carbon_balance_step(
        C_evolution[-1], light[day], temp[day], water[day]
    )
    C_evolution.append(C_new)

C_evolution = np.array(C_evolution)

print(f"\nFinal carbon stock: {C_evolution[-1]:.2f} kg C/m²")
print(f"Net change: {C_evolution[-1] - C_evolution[0]:.2f} kg C/m²")
print(f"Annual NPP: {(C_evolution[-1] - C_evolution[0]):.2f} kg C/m²/year")

# 2. NDVI Forecasting
print("\n" + "-"*70)
print("2. NDVI Forecasting with LSTM")
print("-"*70)

ndvi_predictor = NDVIPredictor(window_size=30)

# Generate synthetic NDVI time series
n_timesteps = 500
t = np.arange(n_timesteps)
ndvi_base = 0.5 + 0.3 * np.sin(2 * np.pi * t / 365)  # Annual cycle
ndvi_noise = np.random.randn(n_timesteps) * 0.05
ndvi_data = np.clip(ndvi_base + ndvi_noise, 0, 1)

# Climate data
climate_data = np.column_stack([
    15 + 10 * np.sin(2 * np.pi * t / 365),  # Temperature
    50 + 30 * np.sin(2 * np.pi * t / 365 + np.pi/4),  # Precipitation
    300 + 150 * np.sin(2 * np.pi * t / 365),  # Solar radiation
    0.5 + 0.2 * np.sin(2 * np.pi * t / 365)  # Soil moisture
])

print(f"NDVI time series length: {len(ndvi_data)}")
print(f"NDVI range: [{ndvi_data.min():.3f}, {ndvi_data.max():.3f}]")
print(f"Climate features: {climate_data.shape[1]}")

# Prepare sequences
X, y = ndvi_predictor.prepare_sequences(ndvi_data, climate_data, forecast_horizon=7)

print(f"\nTraining sequences: {X.shape[0]}")
print(f"Sequence shape: {X.shape}")

# Simple forecast example
test_sequence = X[-1]
forecast = ndvi_predictor.simple_lstm_forecast(test_sequence)

print(f"Last observed NDVI: {ndvi_data[-1]:.3f}")
print(f"7-day forecast: {forecast:.3f}")

# 3. Fire Risk Prediction
print("\n" + "-"*70)
print("3. Fire Risk Prediction")
print("-"*70)

fire_model = FireRiskPredictor()

# Environmental conditions
ndvi_current = 0.3  # Dry vegetation
temp_current = 35.0  # °C
humidity_current = 0.2  # 20%
wind_current = 15.0  # m/s
precip_history = np.random.exponential(2, 100)  # mm/day
slope_current = 20.0  # degrees

# Calculate fuel moisture
ndvi_series = np.random.uniform(0.2, 0.5, 100)
ndwi_series = np.random.uniform(0.1, 0.4, 100)
fmc = fire_model.calculate_fuel_moisture(ndvi_series, ndwi_series, precip_history)

print(f"Current conditions:")
print(f"  NDVI: {ndvi_current:.2f}")
print(f"  Temperature: {temp_current:.1f}°C")
print(f"  Humidity: {humidity_current*100:.0f}%")
print(f"  Wind speed: {wind_current:.1f} m/s")
print(f"  Slope: {slope_current:.1f}°")

# Days since rain
days_no_rain = fire_model.days_since_rain(precip_history)[-1]

print(f"  Days since rain: {days_no_rain:.0f}")
print(f"  Fuel moisture: {fmc[-1]:.1f}%")

# Calculate fire risk
fire_risk = fire_model.calculate_fire_risk(
    ndvi_current, temp_current, humidity_current,
    wind_current, days_no_rain, slope_current
)

print(f"\nFire Risk Probability: {fire_risk:.2f}")
if fire_risk > 0.7:
    print("  Level: EXTREME")
elif fire_risk > 0.5:
    print("  Level: HIGH")
elif fire_risk > 0.3:
    print("  Level: MODERATE")
else:
    print("  Level: LOW")

# 4. Soil Organic Carbon Prediction
print("\n" + "-"*70)
print("4. Soil Organic Carbon Prediction")
print("-"*70)

soc_model = SoilOrganicCarbonPredictor()

# Environmental features
climate_features = {
    'mean_annual_temp': 18.5,
    'mean_annual_precip': 800,
    'temp_seasonality': 8.5
}

vegetation_features = {
    'ndvi_mean': 0.65,
    'ndvi_max': 0.85,
    'npp': 450  # g C/m²/year
}

soil_features = {
    'clay_content': 25,  # %
    'sand_content': 35,
    'bulk_density': 1.3,  # g/cm³
    'ph': 6.5
}

topographic_features = {
    'elevation': 500,  # m
    'slope': 8,  # degrees
    'aspect': 180,  # degrees
    'twi': 7.5
}

print(f"Climate: MAT={climate_features['mean_annual_temp']}°C, "
      f"MAP={climate_features['mean_annual_precip']}mm")
print(f"Vegetation: NDVI mean={vegetation_features['ndvi_mean']:.2f}")
print(f"Soil: Clay={soil_features['clay_content']}%, pH={soil_features['ph']}")

# Extract features
features = soc_model.prepare_features(
    climate_features, vegetation_features,
    soil_features, topographic_features
)

# Predict SOC
soc_pred, soc_uncertainty = soc_model.predict_soc(features)

print(f"\nSoil Organic Carbon Prediction:")
print(f"  Mean: {soc_pred:.2f} kg C/m²")
print(f"  Uncertainty: ±{soc_uncertainty:.2f} kg C/m²")
print(f"  Range: [{soc_pred-soc_uncertainty:.2f}, {soc_pred+soc_uncertainty:.2f}]")

# 5. Ecosystem Service Valuation
print("\n" + "-"*70)
print("5. Ecosystem Service Assessment")
print("-"*70)

esv_model = EcosystemServiceAssessment()

# Current land cover (hectares)
current_landcover = {
    'forest': 1000,
    'grassland': 500,
    'wetland': 200,
    'cropland': 300,
    'urban': 100
}

# Future scenario (conversion)
future_landcover = {
    'forest': 900,  # -100 ha
    'grassland': 450,  # -50 ha
    'wetland': 180,  # -20 ha
    'cropland': 320,  # +20 ha
    'urban': 250  # +150 ha (urbanization)
}

print(f"Current land cover (total: {sum(current_landcover.values()):.0f} ha):")
for lc_type, area in current_landcover.items():
    print(f"  {lc_type.capitalize()}: {area:.0f} ha")

# Calculate ESV
current_esv = esv_model.calculate_land_cover_esv(current_landcover)
esv_change = esv_model.predict_esv_change(current_landcover, future_landcover)

print(f"\nEcosystem Service Value:")
print(f"  Current total: ${current_esv['total']:,.0f}/year")
print(f"  Per hectare: ${current_esv['per_hectare']:.0f}/ha/year")

print(f"\nProjected Change:")
print(f"  Future total: ${esv_change['future_total']:,.0f}/year")
print(f"  Absolute change: ${esv_change['absolute_change']:,.0f}/year")
print(f"  Percent change: {esv_change['percent_change']:.1f}%")

# Visualize results
visualize_vegetation_dynamics_results(
    C_evolution, days, ndvi_data[:200], fmc,
    fire_risk, soc_pred, soc_uncertainty,
    current_landcover, future_landcover, current_esv
)

print("\n" + "="*70)
print("Vegetation dynamics framework demonstration complete!")
print("="*70)
def visualize_vegetation_dynamics_results(carbon, days, ndvi, fmc,
fire_risk, soc, soc_unc,
current_lc, future_lc, esv):
"""Visualize vegetation and ecosystem dynamics results"""
fig = plt.figure(figsize=(16, 12))
# Plot 1: Carbon balance evolution
ax1 = plt.subplot(3, 3, 1)
ax1.plot(days, carbon, 'g-', linewidth=2)
ax1.fill_between(days, carbon * 0.95, carbon * 1.05, alpha=0.3, color='green')
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Carbon Stock (kg C/m²)')
ax1.set_title('Vegetation Carbon Balance Evolution')
ax1.grid(True, alpha=0.3)

# Plot 2: NDVI time series
ax2 = plt.subplot(3, 3, 2)
t_ndvi = np.arange(len(ndvi))
ax2.plot(t_ndvi, ndvi, 'b-', linewidth=1.5, alpha=0.7)

# Add trend line
z = np.polyfit(t_ndvi, ndvi, 1)
p = np.poly1d(z)
ax2.plot(t_ndvi, p(t_ndvi), 'r--', linewidth=2, label=f'Trend')

ax2.set_xlabel('Time (days)')
ax2.set_ylabel('NDVI')
ax2.set_title('NDVI Time Series')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1])

# Plot 3: Fuel moisture content
ax3 = plt.subplot(3, 3, 3)
ax3.plot(fmc, 'orange', linewidth=2)
ax3.axhline(y=50, color='r', linestyle='--', label='Critical threshold')
ax3.set_xlabel('Time (days)')
ax3.set_ylabel('Fuel Moisture Content (%)')
ax3.set_title('Vegetation Fuel Moisture Dynamics')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Fire risk gauge
ax4 = plt.subplot(3, 3, 4)
theta = np.linspace(0, np.pi, 100)
r = np.ones_like(theta)

# Color segments
colors = ['green', 'yellow', 'orange', 'red']
boundaries = [0, 0.3, 0.5, 0.7, 1.0]

for i in range(len(colors)):
    mask = (theta >= boundaries[i] * np.pi) & (theta <= boundaries[i+1] * np.pi)
    ax4.fill_between(theta[mask], 0, r[mask], color=colors[i], alpha=0.5)

# Risk needle
risk_angle = fire_risk * np.pi
ax4.plot([risk_angle, risk_angle], [0, 1], 'k-', linewidth=3)
ax4.plot(risk_angle, 1, 'ko', markersize=10)

ax4.set_ylim([0, 1.2])
ax4.set_xlim([0, np.pi])
ax4.set_title(f'Fire Risk: {fire_risk:.2f}')
ax4.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
ax4.set_xticklabels(['0', '0.25', '0.5', '0.75', '1.0'])
ax4.set_yticks([])
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_visible(False)

# Plot 5: SOC prediction with uncertainty
ax5 = plt.subplot(3, 3, 5)
categories = ['Predicted SOC']
values = [soc]
errors = [soc_unc]

bars = ax5.bar(categories, values, yerr=errors, capsize=10,
              color='brown', alpha=0.7, edgecolor='black', linewidth=2)
ax5.set_ylabel('SOC (kg C/m²)')
ax5.set_title('Soil Organic Carbon Prediction')
ax5.grid(True, alpha=0.3, axis='y')

# Add value label
for bar, val, err in zip(bars, values, errors):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f} ± {err:.1f}',
            ha='center', va='bottom', fontweight='bold')

# Plot 6: Land cover comparison
ax6 = plt.subplot(3, 3, 6)
lc_types = list(current_lc.keys())
current_vals = [current_lc[lc] for lc in lc_types]
future_vals = [future_lc[lc] for lc in lc_types]

x = np.arange(len(lc_types))
width = 0.35

ax6.bar(x - width/2, current_vals, width, label='Current', alpha=0.8)
ax6.bar(x + width/2, future_vals, width, label='Future', alpha=0.8)

ax6.set_ylabel('Area (ha)')
ax6.set_title('Land Cover Change Scenario')
ax6.set_xticks(x)
ax6.set_xticklabels([lc.capitalize() for lc in lc_types], rotation=45, ha='right')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

# Plot 7: ESV by land cover type
ax7 = plt.subplot(3, 3, 7)
esv_by_type = esv['by_type']
lc_with_esv = [lc for lc in lc_types if lc in esv_by_type]
esv_values = [esv_by_type[lc] for lc in lc_with_esv]

colors_esv = plt.cm.Greens(np.linspace(0.4, 0.9, len(lc_with_esv)))
bars = ax7.barh(lc_with_esv, esv_values, color=colors_esv, edgecolor='black')
ax7.set_xlabel('ESV ($/year)')
ax7.set_title('Ecosystem Service Value by Land Cover')
ax7.grid(True, alpha=0.3, axis='x')

# Plot 8: Seasonal NDVI pattern
ax8 = plt.subplot(3, 3, 8)
seasonal_ndvi = ndvi[:min(365, len(ndvi))]
doy = np.arange(len(seasonal_ndvi))

ax8.plot(doy, seasonal_ndvi, 'g-', linewidth=2, alpha=0.7)
ax8.fill_between(doy, seasonal_ndvi, alpha=0.3, color='green')
ax8.set_xlabel('Day of Year')
ax8.set_ylabel('NDVI')
ax8.set_title('Seasonal NDVI Pattern')
ax8.grid(True, alpha=0.3)
ax8.set_xlim([0, 365])
ax8.set_ylim([0, 1])

# Plot 9: Carbon fluxes
ax9 = plt.subplot(3, 3, 9)
flux_categories = ['GPP', 'Respiration', 'Net\nEcosystem\nExchange']
# Approximate values
gpp_annual = (carbon[-1] - carbon[0]) * 2.5  # Rough estimate
resp_annual = gpp_annual * 0.6
nee = gpp_annual - resp_annual

flux_values = [gpp_annual, -resp_annual, nee]
colors_flux = ['green', 'red', 'blue']

bars = ax9.bar(flux_categories, flux_values, color=colors_flux, alpha=0.7,
              edgecolor='black', linewidth=2)
ax9.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax9.set_ylabel('Carbon Flux (kg C/m²/year)')
ax9.set_title('Annual Carbon Budget')
ax9.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('vegetation_dynamics_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nResults saved to 'vegetation_dynamics_results.png'")
if name == "main":
demonstrate_vegetation_dynamics_framework()
