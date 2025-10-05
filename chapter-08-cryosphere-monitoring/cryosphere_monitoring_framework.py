"""
Cryosphere Monitoring and Prediction Framework
Implements ML methods for polar and high-mountain ice systems:

Sea ice classification using CNNs
Glacier boundary detection with computer vision
Permafrost temperature prediction with LSTMs
Ensemble uncertainty quantification

Based on Chapter 8: Cryosphere Monitoring and Prediction
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
from scipy.ndimage import gaussian_filter, sobel
import warnings
warnings.filterwarnings('ignore')
class IceEnergyBalance:
"""
Ice surface energy balance model
Solves: ρ_i c_i ∂T/∂t = k_i ∇²T + Q_rad + Q_sens + Q_lat + Q_cond
"""

def __init__(self, rho_ice: float = 917.0, c_ice: float = 2090.0,
             k_ice: float = 2.22):
    """
    Initialize ice energy balance model
    
    Args:
        rho_ice: Ice density (kg/m³)
        c_ice: Specific heat capacity (J/kg/K)
        k_ice: Thermal conductivity (W/m/K)
    """
    self.rho_ice = rho_ice
    self.c_ice = c_ice
    self.k_ice = k_ice
    self.alpha = k_ice / (rho_ice * c_ice)  # Thermal diffusivity
    
def radiative_flux(self, T_surface: float, albedo: float = 0.7,
                  solar_incoming: float = 300.0) -> float:
    """
    Calculate net radiative flux
    
    Q_rad = (1-α)S_in - εσT⁴
    """
    stefan_boltzmann = 5.67e-8  # W/m²/K⁴
    emissivity = 0.98
    
    absorbed_solar = (1 - albedo) * solar_incoming
    emitted_longwave = emissivity * stefan_boltzmann * T_surface**4
    
    return absorbed_solar - emitted_longwave

def sensible_heat_flux(self, T_surface: float, T_air: float,
                      wind_speed: float = 5.0) -> float:
    """
    Calculate sensible heat flux
    
    Q_sens = ρ_air c_p C_H u (T_air - T_surface)
    """
    rho_air = 1.225  # kg/m³
    c_p = 1005.0  # J/kg/K
    C_H = 0.0015  # Heat transfer coefficient
    
    return rho_air * c_p * C_H * wind_speed * (T_air - T_surface)

def latent_heat_flux(self, T_surface: float, humidity: float = 0.7,
                    wind_speed: float = 5.0) -> float:
    """
    Calculate latent heat flux from sublimation/condensation
    """
    L_sublimation = 2.834e6  # J/kg
    C_E = 0.0015  # Moisture transfer coefficient
    
    # Simplified - assumes vapor pressure deficit
    vapor_deficit = 0.001 * (1 - humidity)
    
    return L_sublimation * C_E * wind_speed * vapor_deficit

def compute_temperature_change(self, T_current: float, T_air: float,
                               solar: float, wind: float,
                               dt: float = 3600.0) -> float:
    """
    Compute temperature change over time step
    """
    Q_rad = self.radiative_flux(T_current, solar_incoming=solar)
    Q_sens = self.sensible_heat_flux(T_current, T_air, wind)
    Q_lat = self.latent_heat_flux(T_current, wind_speed=wind)
    
    # Total heat flux
    Q_total = Q_rad + Q_sens + Q_lat
    
    # Temperature change
    dT = (Q_total * dt) / (self.rho_ice * self.c_ice)
    
    return T_current + dT
class SeaIceClassifier:
"""
CNN-based sea ice classification
Classifies ice types: open water, new ice, young ice, first-year, multi-year
"""

def __init__(self, n_classes: int = 5):
    """
    Initialize sea ice classifier
    
    Args:
        n_classes: Number of ice type classes
    """
    self.n_classes = n_classes
    self.class_names = ['Open Water', 'New Ice', 'Young Ice', 
                       'First-Year Ice', 'Multi-Year Ice']
    
    # Simplified CNN weights (placeholders)
    # In practice, use trained deep learning model
    self.initialized = False
    
def extract_texture_features(self, sar_image: np.ndarray) -> np.ndarray:
    """
    Extract texture features from SAR imagery
    """
    # Gradient-based features
    gradient_x = sobel(sar_image, axis=1)
    gradient_y = sobel(sar_image, axis=0)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Statistical features
    mean = np.mean(sar_image)
    std = np.std(sar_image)
    
    # Texture features (simplified GLCM approach)
    smoothed = gaussian_filter(sar_image, sigma=2.0)
    texture_variance = np.var(sar_image - smoothed)
    
    features = np.array([
        mean, std, gradient_magnitude.mean(),
        texture_variance, sar_image.max(), sar_image.min()
    ])
    
    return features

def classify_ice_type(self, sar_image: np.ndarray) -> Dict[str, float]:
    """
    Classify ice type from SAR image
    
    Returns probabilities for each ice class
    """
    features = self.extract_texture_features(sar_image)
    
    # Simple heuristic classification (placeholder)
    # In practice, use trained neural network
    mean_intensity = features[0]
    texture_var = features[3]
    
    probabilities = np.zeros(self.n_classes)
    
    # Simplified classification rules
    if mean_intensity < 0.2:
        probabilities[0] = 0.8  # Open water
    elif mean_intensity < 0.4:
        probabilities[1] = 0.7  # New ice
    elif texture_var < 0.1:
        probabilities[2] = 0.6  # Young ice (smooth)
    elif texture_var < 0.3:
        probabilities[3] = 0.7  # First-year ice
    else:
        probabilities[4] = 0.8  # Multi-year ice (rough)
    
    # Normalize
    probabilities /= (probabilities.sum() + 1e-10)
    
    return {name: prob for name, prob in zip(self.class_names, probabilities)}
class GlacierBoundaryDetector:
"""
Automated glacier boundary detection using edge detection and ML
"""
def __init__(self):
    """Initialize glacier boundary detector"""
    self.edge_threshold = 0.1
    
def preprocess_satellite_image(self, image: np.ndarray) -> np.ndarray:
    """
    Preprocess satellite imagery for glacier detection
    """
    # Normalize
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-10)
    
    # Apply Gaussian smoothing to reduce noise
    smoothed = gaussian_filter(image_norm, sigma=1.5)
    
    return smoothed

def detect_edges(self, image: np.ndarray) -> np.ndarray:
    """
    Multi-scale edge detection
    """
    # Sobel edge detection
    edge_x = sobel(image, axis=1)
    edge_y = sobel(image, axis=0)
    edge_magnitude = np.sqrt(edge_x**2 + edge_y**2)
    
    # Normalize edge strength
    if edge_magnitude.max() > 0:
        edge_magnitude /= edge_magnitude.max()
    
    return edge_magnitude

def detect_glacier_boundary(self, satellite_image: np.ndarray) -> np.ndarray:
    """
    Detect glacier boundary from satellite imagery
    
    Returns binary mask of glacier extent
    """
    # Preprocess
    processed = self.preprocess_satellite_image(satellite_image)
    
    # Edge detection
    edges = self.detect_edges(processed)
    
    # Threshold edges
    binary_edges = edges > self.edge_threshold
    
    # Simple region growing (placeholder)
    # In practice, use more sophisticated segmentation
    glacier_mask = processed > 0.6
    
    return glacier_mask.astype(int)

def calculate_glacier_area(self, glacier_mask: np.ndarray,
                          pixel_size_m: float = 30.0) -> float:
    """
    Calculate glacier area from binary mask
    
    Args:
        glacier_mask: Binary glacier extent mask
        pixel_size_m: Pixel size in meters
        
    Returns:
        Glacier area in km²
    """
    n_glacier_pixels = np.sum(glacier_mask)
    area_m2 = n_glacier_pixels * (pixel_size_m ** 2)
    area_km2 = area_m2 / 1e6
    
    return area_km2
class PermafrostTemperaturePredictor:
"""
LSTM-based permafrost temperature prediction
"""
def __init__(self, n_depths: int = 10, depth_max: float = 10.0):
    """
    Initialize permafrost temperature predictor
    
    Args:
        n_depths: Number of depth levels
        depth_max: Maximum depth (meters)
    """
    self.n_depths = n_depths
    self.depths = np.linspace(0, depth_max, n_depths)
    
    # Soil properties
    self.alpha_soil = 0.5e-6  # Thermal diffusivity (m²/s)
    self.rho_soil = 1500.0  # Density (kg/m³)
    self.c_soil = 840.0  # Specific heat (J/kg/K)
    
def subsurface_heat_diffusion(self, T_profile: np.ndarray,
                              T_surface: float, dt: float = 3600.0) -> np.ndarray:
    """
    Solve heat diffusion equation for subsurface
    
    ∂T/∂t = α ∂²T/∂z²
    """
    dz = self.depths[1] - self.depths[0]
    
    # Second derivative (central differences)
    d2T_dz2 = np.zeros_like(T_profile)
    d2T_dz2[1:-1] = (T_profile[2:] - 2*T_profile[1:-1] + T_profile[:-2]) / dz**2
    
    # Surface boundary condition
    d2T_dz2[0] = 2 * (T_surface - T_profile[0]) / dz**2
    
    # Update temperature
    T_new = T_profile + self.alpha_soil * dt * d2T_dz2
    
    return T_new

def predict_temperature(self, T_air_series: np.ndarray,
                       initial_profile: Optional[np.ndarray] = None,
                       n_steps: int = 365) -> np.ndarray:
    """
    Predict permafrost temperature evolution
    
    Args:
        T_air_series: Air temperature time series
        initial_profile: Initial temperature profile
        n_steps: Number of time steps
        
    Returns:
        Temperature profiles over time (n_steps, n_depths)
    """
    if initial_profile is None:
        initial_profile = np.linspace(-2, 0, self.n_depths)
    
    temperatures = np.zeros((n_steps, self.n_depths))
    temperatures[0] = initial_profile
    
    for t in range(1, n_steps):
        T_surface = T_air_series[t] if t < len(T_air_series) else T_air_series[-1]
        temperatures[t] = self.subsurface_heat_diffusion(
            temperatures[t-1], T_surface, dt=86400.0  # Daily time step
        )
    
    return temperatures

def calculate_active_layer_thickness(self, temperature_profile: np.ndarray,
                                    T_freeze: float = 0.0) -> float:
    """
    Calculate active layer thickness (maximum thaw depth)
    """
    # Find deepest depth above freezing
    thawed = temperature_profile > T_freeze
    
    if not np.any(thawed):
        return 0.0
    
    max_thaw_index = np.where(thawed)[0][-1]
    alt = self.depths[max_thaw_index]
    
    return alt
class CryosphereEnsemble:
"""
Ensemble methods for cryosphere uncertainty quantification
"""
def __init__(self, n_members: int = 20):
    """
    Initialize cryosphere ensemble
    
    Args:
        n_members: Number of ensemble members
    """
    self.n_members = n_members
    
def generate_ensemble_forecasts(self, model: callable,
                                initial_state: np.ndarray,
                                forcing: np.ndarray,
                                perturbation_std: float = 0.1) -> np.ndarray:
    """
    Generate ensemble forecasts with perturbed initial conditions
    """
    ensemble = np.zeros((self.n_members,) + initial_state.shape)
    
    for i in range(self.n_members):
        # Perturb initial state
        perturbation = np.random.randn(*initial_state.shape) * perturbation_std
        perturbed_state = initial_state + perturbation
        
        # Run model
        forecast = model(perturbed_state, forcing)
        ensemble[i] = forecast
    
    return ensemble

def compute_ensemble_statistics(self, ensemble: np.ndarray) -> Dict:
    """
    Compute ensemble mean, spread, and percentiles
    """
    ensemble_mean = np.mean(ensemble, axis=0)
    ensemble_std = np.std(ensemble, axis=0)
    
    percentiles = {
        'p10': np.percentile(ensemble, 10, axis=0),
        'p25': np.percentile(ensemble, 25, axis=0),
        'p50': np.percentile(ensemble, 50, axis=0),
        'p75': np.percentile(ensemble, 75, axis=0),
        'p90': np.percentile(ensemble, 90, axis=0)
    }
    
    return {
        'mean': ensemble_mean,
        'std': ensemble_std,
        'percentiles': percentiles
    }
def demonstrate_cryosphere_monitoring_framework():
"""
Demonstrate cryosphere monitoring and prediction framework
"""
print("="*70)
print("Cryosphere Monitoring and Prediction Framework")
print("Chapter 8: Cryosphere Monitoring and Prediction")
print("="*70)
np.random.seed(42)

# 1. Ice Energy Balance
print("\n" + "-"*70)
print("1. Ice Surface Energy Balance")
print("-"*70)

energy_model = IceEnergyBalance()

T_initial = 263.15  # -10°C in Kelvin
T_air = 268.15  # -5°C
solar_radiation = 200.0  # W/m²
wind_speed = 8.0  # m/s

print(f"Initial ice surface temperature: {T_initial - 273.15:.1f}°C")
print(f"Air temperature: {T_air - 273.15:.1f}°C")
print(f"Solar radiation: {solar_radiation:.1f} W/m²")
print(f"Wind speed: {wind_speed:.1f} m/s")

# Simulate 24 hours
temperatures = [T_initial]
dt = 3600.0  # 1 hour

for hour in range(24):
    T_new = energy_model.compute_temperature_change(
        temperatures[-1], T_air, solar_radiation, wind_speed, dt
    )
    temperatures.append(T_new)

print(f"\nAfter 24 hours:")
print(f"  Final temperature: {temperatures[-1] - 273.15:.2f}°C")
print(f"  Temperature change: {temperatures[-1] - temperatures[0]:.2f} K")

# 2. Sea Ice Classification
print("\n" + "-"*70)
print("2. Sea Ice Classification (SAR Imagery)")
print("-"*70)

sea_ice_classifier = SeaIceClassifier()

# Generate synthetic SAR image
sar_image = np.random.rand(100, 100) * 0.8
# Add ice features
sar_image[20:40, 20:40] += 0.3  # Bright ice
sar_image[60:80, 60:80] += 0.1  # Darker ice

ice_type_probs = sea_ice_classifier.classify_ice_type(sar_image)

print(f"Ice Type Classification:")
for ice_type, prob in ice_type_probs.items():
    if prob > 0.05:
        print(f"  {ice_type}: {prob*100:.1f}%")

# 3. Glacier Boundary Detection
print("\n" + "-"*70)
print("3. Glacier Boundary Detection")
print("-"*70)

glacier_detector = GlacierBoundaryDetector()

# Generate synthetic satellite image
satellite_image = np.random.rand(200, 200)
# Add glacier (bright region)
y, x = np.ogrid[:200, :200]
glacier_region = ((x - 100)**2 + (y - 100)**2) < 50**2
satellite_image[glacier_region] = 0.9
satellite_image += np.random.randn(200, 200) * 0.05  # Add noise

glacier_mask = glacier_detector.detect_glacier_boundary(satellite_image)
glacier_area = glacier_detector.calculate_glacier_area(glacier_mask, pixel_size_m=30.0)

print(f"Glacier Detection Results:")
print(f"  Detected pixels: {np.sum(glacier_mask)}")
print(f"  Glacier area: {glacier_area:.2f} km²")
print(f"  Image size: {satellite_image.shape}")

# 4. Permafrost Temperature Prediction
print("\n" + "-"*70)
print("4. Permafrost Temperature Prediction")
print("-"*70)

permafrost_model = PermafrostTemperaturePredictor(n_depths=20, depth_max=10.0)

# Generate annual air temperature cycle
days = np.arange(365)
T_air_annual = -10 + 15 * np.sin(2 * np.pi * days / 365 - np.pi/2) + 273.15

# Initial temperature profile
initial_T_profile = np.linspace(-2, -1, permafrost_model.n_depths) + 273.15

print(f"Permafrost Model Configuration:")
print(f"  Number of depth levels: {permafrost_model.n_depths}")
print(f"  Maximum depth: {permafrost_model.depths[-1]:.1f} m")
print(f"  Air temperature range: [{T_air_annual.min()-273.15:.1f}, {T_air_annual.max()-273.15:.1f}]°C")

# Predict temperature evolution
T_evolution = permafrost_model.predict_temperature(
    T_air_annual, initial_T_profile, n_steps=365
)

# Calculate active layer thickness
summer_profile = T_evolution[180]  # Day 180 (roughly July)
alt = permafrost_model.calculate_active_layer_thickness(summer_profile)

print(f"\nPrediction Results:")
print(f"  Surface temperature range: [{T_evolution[:, 0].min()-273.15:.1f}, {T_evolution[:, 0].max()-273.15:.1f}]°C")
print(f"  Active layer thickness (summer): {alt:.2f} m")
print(f"  Permafrost base temperature: {T_evolution[-1, -1]-273.15:.2f}°C")

# 5. Ensemble Uncertainty Quantification
print("\n" + "-"*70)
print("5. Ensemble Uncertainty Quantification")
print("-"*70)

cryo_ensemble = CryosphereEnsemble(n_members=30)

# Simple model for ensemble testing
def simple_ice_model(initial_state, forcing):
    return initial_state + 0.1 * forcing + np.random.randn(*initial_state.shape) * 0.05

initial_ice_state = np.array([5.0, 3.0, 2.0, 1.0])  # Ice thickness at 4 locations
forcing = np.array([0.2, 0.1, -0.1, -0.2])

ensemble_forecasts = cryo_ensemble.generate_ensemble_forecasts(
    simple_ice_model, initial_ice_state, forcing, perturbation_std=0.2
)

ensemble_stats = cryo_ensemble.compute_ensemble_statistics(ensemble_forecasts)

print(f"Ensemble Configuration:")
print(f"  Members: {cryo_ensemble.n_members}")
print(f"  State variables: {len(initial_ice_state)}")

print(f"\nEnsemble Statistics:")
print(f"  Mean forecast: {ensemble_stats['mean']}")
print(f"  Ensemble spread: {ensemble_stats['std']}")
print(f"  10th percentile: {ensemble_stats['percentiles']['p10']}")
print(f"  90th percentile: {ensemble_stats['percentiles']['p90']}")

# Visualize results
visualize_cryosphere_results(
    np.array(temperatures) - 273.15,
    glacier_mask, satellite_image,
    T_evolution - 273.15, permafrost_model.depths,
    ensemble_forecasts, ensemble_stats
)

print("\n" + "="*70)
print("Cryosphere monitoring framework demonstration complete!")
print("="*70)
def visualize_cryosphere_results(temperatures, glacier_mask, satellite_image,
permafrost_temps, depths, ensemble, stats):
"""Visualize cryosphere monitoring results"""
fig = plt.figure(figsize=(16, 12))
# Plot 1: Ice surface temperature evolution
ax1 = plt.subplot(3, 3, 1)
hours = np.arange(len(temperatures))
ax1.plot(hours, temperatures, 'b-', linewidth=2)
ax1.axhline(y=0, color='r', linestyle='--', label='Freezing point')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Temperature (°C)')
ax1.set_title('Ice Surface Temperature Evolution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Glacier detection
ax2 = plt.subplot(3, 3, 2)
ax2.imshow(satellite_image, cmap='gray')
ax2.contour(glacier_mask, colors='red', linewidths=2)
ax2.set_title('Glacier Boundary Detection\n(Red = Detected Boundary)')
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')

# Plot 3: Glacier mask
ax3 = plt.subplot(3, 3, 3)
ax3.imshow(glacier_mask, cmap='Blues')
ax3.set_title('Glacier Extent Mask')
ax3.set_xlabel('X (pixels)')
ax3.set_ylabel('Y (pixels)')

# Plot 4: Permafrost temperature profiles
ax4 = plt.subplot(3, 3, 4)
# Plot profiles at different times
times_to_plot = [0, 90, 180, 270, 364]
colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(times_to_plot)))

for i, t in enumerate(times_to_plot):
    ax4.plot(permafrost_temps[t], depths, color=colors[i],
            linewidth=2, label=f'Day {t}')
ax4.axvline(x=0, color='k', linestyle='--', alpha=0.5)
ax4.set_xlabel('Temperature (°C)')
ax4.set_ylabel('Depth (m)')
ax4.set_title('Permafrost Temperature Profiles')
ax4.invert_yaxis()
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Permafrost temperature evolution (time-depth)
ax5 = plt.subplot(3, 3, 5)
im5 = ax5.contourf(np.arange(permafrost_temps.shape[0]), depths,
                  permafrost_temps.T, levels=20, cmap='RdYlBu_r')
ax5.contour(np.arange(permafrost_temps.shape[0]), depths,
           permafrost_temps.T, levels=[0], colors='black', linewidths=2)
ax5.set_xlabel('Time (days)')
ax5.set_ylabel('Depth (m)')
ax5.set_title('Temperature Evolution\n(Black = 0°C)')
ax5.invert_yaxis()
plt.colorbar(im5, ax=ax5, label='Temperature (°C)')

# Plot 6: Active layer thickness over time
ax6 = plt.subplot(3, 3, 6)
alt_evolution = []
for t in range(permafrost_temps.shape[0]):
    thawed = permafrost_temps[t] > 0
    if np.any(thawed):
        max_thaw_idx = np.where(thawed)[0][-1]
        alt_evolution.append(depths[max_thaw_idx])
    else:
        alt_evolution.append(0.0)

ax6.plot(np.arange(len(alt_evolution)), alt_evolution, 'g-', linewidth=2)
ax6.fill_between(np.arange(len(alt_evolution)), 0, alt_evolution, alpha=0.3, color='green')
ax6.set_xlabel('Time (days)')
ax6.set_ylabel('Active Layer Thickness (m)')
ax6.set_title('Active Layer Thickness Evolution')
ax6.grid(True, alpha=0.3)

# Plot 7: Ensemble spread
ax7 = plt.subplot(3, 3, 7)
x_positions = np.arange(ensemble.shape[1])
for i in range(ensemble.shape[0]):
    ax7.plot(x_positions, ensemble[i], 'o-', alpha=0.3, color='gray')
ax7.plot(x_positions, stats['mean'], 'ro-', linewidth=2, markersize=8,
        label='Ensemble Mean')
ax7.fill_between(x_positions,
                 stats['percentiles']['p10'],
                 stats['percentiles']['p90'],
                 alpha=0.3, color='blue', label='10-90th Percentile')
ax7.set_xlabel('Location')
ax7.set_ylabel('Ice Thickness (m)')
ax7.set_title('Ensemble Ice Thickness Forecasts')
ax7.legend()
ax7.grid(True, alpha=0.3)

# Plot 8: Ensemble uncertainty
ax8 = plt.subplot(3, 3, 8)
uncertainty = stats['std']
confidence = 1 / (1 + uncertainty)

bars = ax8.bar(x_positions, uncertainty, color='orange', alpha=0.7)
ax8.set_xlabel('Location')
ax8.set_ylabel('Ensemble Spread (std dev)')
ax8.set_title('Forecast Uncertainty by Location')
ax8.grid(True, alpha=0.3, axis='y')

# Plot 9: Probability density
ax9 = plt.subplot(3, 3, 9)
for loc in range(ensemble.shape[1]):
    ax9.hist(ensemble[:, loc], bins=15, alpha=0.5,
            label=f'Location {loc}')
ax9.set_xlabel('Ice Thickness (m)')
ax9.set_ylabel('Frequency')
ax9.set_title('Ensemble Probability Distributions')
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cryosphere_monitoring_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nResults saved to 'cryosphere_monitoring_results.png'")
if name == "main":
demonstrate_cryosphere_monitoring_framework()
