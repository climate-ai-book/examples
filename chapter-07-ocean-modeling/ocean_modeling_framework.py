"""
Ocean Modeling with Artificial Intelligence Framework
Implements ML methods for ocean dynamics and marine systems:

Physics-informed neural networks for ocean circulation
Graph neural networks for irregular ocean grids
Marine ecosystem and biogeochemical modeling
Ocean data assimilation with neural networks

Based on Chapter 7: Ocean Modeling with Artificial Intelligence
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
from scipy.ndimage import convolve
import warnings
warnings.filterwarnings('ignore')
class OceanPrimitiveEquations:
"""
Numerical solver for ocean primitive equations
Solves: ∂u/∂t + advection + Coriolis = pressure gradient + diffusion
"""

def __init__(self, grid_shape: Tuple[int, int, int], 
             dx: float, dy: float, dz: float,
             latitude: float = 30.0):
    """
    Initialize ocean dynamics model
    
    Args:
        grid_shape: (nx, ny, nz) grid dimensions
        dx, dy, dz: grid spacing
        latitude: Latitude for Coriolis parameter (degrees)
    """
    self.nx, self.ny, self.nz = grid_shape
    self.dx = dx
    self.dy = dy
    self.dz = dz
    
    # Coriolis parameter
    omega = 7.2921e-5  # Earth's angular velocity (rad/s)
    self.f = 2 * omega * np.sin(np.radians(latitude))
    
    # Physical constants
    self.rho0 = 1025.0  # Reference density (kg/m³)
    self.g = 9.81  # Gravitational acceleration (m/s²)
    
def coriolis_term(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Coriolis acceleration: -fv for u, fu for v
    """
    coriolis_u = -self.f * v
    coriolis_v = self.f * u
    
    return coriolis_u, coriolis_v

def pressure_gradient(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pressure gradient: -(1/ρ₀)∇p
    """
    # Central differences for pressure gradient
    dp_dx = np.gradient(p, self.dx, axis=1)
    dp_dy = np.gradient(p, self.dy, axis=0)
    
    return -dp_dx / self.rho0, -dp_dy / self.rho0

def advection_term(self, field: np.ndarray, u: np.ndarray, 
                  v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute advection: -(u∂/∂x + v∂/∂y + w∂/∂z)field
    """
    # Upwind scheme for stability
    dfield_dx = np.gradient(field, self.dx, axis=1)
    dfield_dy = np.gradient(field, self.dy, axis=0)
    dfield_dz = np.gradient(field, self.dz, axis=2) if field.ndim > 2 else 0
    
    advection = -(u * dfield_dx + v * dfield_dy + w * dfield_dz)
    
    return advection

def step(self, u: np.ndarray, v: np.ndarray, w: np.ndarray,
         p: np.ndarray, dt: float) -> Tuple[np.ndarray, ...]:
    """
    Single time step for ocean primitive equations
    """
    # Coriolis force
    cor_u, cor_v = self.coriolis_term(u, v)
    
    # Pressure gradient
    pg_u, pg_v = self.pressure_gradient(p)
    
    # Advection
    adv_u = self.advection_term(u, u, v, w)
    adv_v = self.advection_term(v, u, v, w)
    
    # Update velocities
    u_new = u + dt * (adv_u + cor_u + pg_u)
    v_new = v + dt * (adv_v + cor_v + pg_v)
    
    # Ensure continuity (simplified)
    # In practice, use more sophisticated pressure-velocity coupling
    
    return u_new, v_new, w
class PhysicsInformedOceanNet:
"""
Physics-Informed Neural Network for ocean circulation
Incorporates geostrophic balance and mass conservation
"""

def __init__(self, input_dim: int = 10, hidden_dim: int = 128):
    """
    Initialize PINN for ocean
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
    """
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    
    # Placeholder for neural network weights
    # In practice, use PyTorch or TensorFlow
    self.w1 = np.random.randn(input_dim, hidden_dim) * 0.01
    self.w2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
    self.w3 = np.random.randn(hidden_dim, 3) * 0.01  # Output: u, v, p
    
def forward(self, x: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Forward pass through network
    
    Returns:
        Dictionary with u, v, pressure predictions
    """
    # Simple feedforward (placeholder)
    h1 = np.tanh(np.dot(x, self.w1))
    h2 = np.tanh(np.dot(h1, self.w2))
    output = np.dot(h2, self.w3)
    
    return {
        'u': output[:, 0],
        'v': output[:, 1],
        'p': output[:, 2]
    }

def geostrophic_loss(self, u: np.ndarray, v: np.ndarray,
                    p: np.ndarray, f: float, rho0: float,
                    dx: float, dy: float) -> float:
    """
    Compute geostrophic balance residual
    
    fu + (1/ρ₀)∂p/∂y = 0
    fv - (1/ρ₀)∂p/∂x = 0
    """
    # Compute pressure gradients
    dp_dx = np.gradient(p, dx)
    dp_dy = np.gradient(p, dy)
    
    # Geostrophic residuals
    residual_u = f * u + (1 / rho0) * dp_dy
    residual_v = f * v - (1 / rho0) * dp_dx
    
    loss = np.mean(residual_u**2 + residual_v**2)
    
    return loss

def continuity_loss(self, u: np.ndarray, v: np.ndarray,
                   dx: float, dy: float) -> float:
    """
    Compute continuity equation residual
    
    ∂u/∂x + ∂v/∂y = 0
    """
    du_dx = np.gradient(u, dx)
    dv_dy = np.gradient(v, dy)
    
    residual = du_dx + dv_dy
    loss = np.mean(residual**2)
    
    return loss

def physics_informed_loss(self, predictions: Dict[str, np.ndarray],
                        targets: Dict[str, np.ndarray],
                        physics_params: Dict) -> Dict[str, float]:
    """
    Compute total physics-informed loss
    """
    # Data loss
    data_loss = np.mean((predictions['u'] - targets['u'])**2 +
                       (predictions['v'] - targets['v'])**2 +
                       (predictions['p'] - targets['p'])**2)
    
    # Physics losses
    geostrophic_loss = self.geostrophic_loss(
        predictions['u'], predictions['v'], predictions['p'],
        physics_params['f'], physics_params['rho0'],
        physics_params['dx'], physics_params['dy']
    )
    
    continuity_loss = self.continuity_loss(
        predictions['u'], predictions['v'],
        physics_params['dx'], physics_params['dy']
    )
    
    # Total loss
    total_loss = (data_loss + 
                 0.1 * geostrophic_loss + 
                 0.1 * continuity_loss)
    
    return {
        'total': total_loss,
        'data': data_loss,
        'geostrophic': geostrophic_loss,
        'continuity': continuity_loss
    }
class MarineBiogeochemistryModel:
"""
Neural network model for marine ecosystem and biogeochemistry
Predicts primary productivity and carbon cycling
"""

def __init__(self, n_species: int = 5):
    """
    Initialize marine biogeochemistry model
    
    Args:
        n_species: Number of biological/chemical species
    """
    self.n_species = n_species
    
    # Environmental limitation parameters
    self.optimal_temp = 20.0  # °C
    self.temp_range = 10.0
    
def temperature_limitation(self, sst: np.ndarray) -> np.ndarray:
    """
    Temperature limitation function for biological processes
    
    L_T = exp(-((T - T_opt)/σ)²)
    """
    return np.exp(-((sst - self.optimal_temp) / self.temp_range)**2)

def light_limitation(self, par: np.ndarray, k_par: float = 50.0) -> np.ndarray:
    """
    Light limitation following Michaelis-Menten kinetics
    
    L_PAR = PAR/(k_PAR + PAR)
    """
    return par / (k_par + par)

def nutrient_limitation(self, nutrient: np.ndarray, 
                       k_nutrient: float = 1.0) -> np.ndarray:
    """
    Nutrient limitation (Monod equation)
    
    L_N = [N]/(k_N + [N])
    """
    return nutrient / (k_nutrient + nutrient)

def primary_productivity(self, sst: np.ndarray, par: np.ndarray,
                        nitrate: np.ndarray, phosphate: np.ndarray,
                        mld: np.ndarray) -> np.ndarray:
    """
    Predict primary productivity from environmental variables
    
    P = P_max · L_T · L_PAR · L_N · L_P
    """
    P_max = 100.0  # Maximum productivity (mg C/m²/day)
    
    L_temp = self.temperature_limitation(sst)
    L_light = self.light_limitation(par)
    L_nitrate = self.nutrient_limitation(nitrate, k_nutrient=1.0)
    L_phosphate = self.nutrient_limitation(phosphate, k_nutrient=0.1)
    
    # Mixed layer depth effect (simplified)
    L_mld = np.exp(-mld / 50.0)  # Deeper MLD reduces productivity
    
    productivity = P_max * L_temp * L_light * L_nitrate * L_phosphate * L_mld
    
    return productivity

def carbon_cycle_step(self, carbon_pools: Dict[str, np.ndarray],
                     temperature: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
    """
    Simple carbon cycle dynamics
    
    dC_i/dt = production - respiration - export
    """
    # Temperature-dependent rates
    Q10 = 2.0  # Temperature sensitivity
    rate_factor = Q10**((temperature - 10.0) / 10.0)
    
    # Simple carbon transformations
    dissolved_organic = carbon_pools.get('dissolved_organic', np.zeros_like(temperature))
    particulate_organic = carbon_pools.get('particulate_organic', np.zeros_like(temperature))
    
    # Production from primary productivity (simplified)
    production = 0.5 * rate_factor
    
    # Respiration (temperature dependent)
    respiration = 0.1 * dissolved_organic * rate_factor
    
    # Export to deep ocean
    export = 0.05 * particulate_organic
    
    # Update pools
    dissolved_organic_new = dissolved_organic + dt * (production - respiration)
    particulate_organic_new = particulate_organic + dt * (0.3 * production - export)
    
    return {
        'dissolved_organic': dissolved_organic_new,
        'particulate_organic': particulate_organic_new,
        'export_flux': export
    }
class OceanDataAssimilation:
"""
Neural network-enhanced data assimilation for ocean modeling
Implements simplified ensemble Kalman filter
"""

def __init__(self, state_dim: int, n_ensemble: int = 20):
    """
    Initialize data assimilation system
    
    Args:
        state_dim: Dimension of ocean state vector
        n_ensemble: Number of ensemble members
    """
    self.state_dim = state_dim
    self.n_ensemble = n_ensemble
    
def generate_ensemble(self, initial_state: np.ndarray,
                     perturbation_std: float = 0.1) -> np.ndarray:
    """
    Generate ensemble from initial state
    """
    ensemble = np.zeros((self.n_ensemble, self.state_dim))
    
    for i in range(self.n_ensemble):
        perturbation = np.random.randn(self.state_dim) * perturbation_std
        ensemble[i] = initial_state + perturbation
    
    return ensemble

def observation_operator(self, state: np.ndarray,
                       obs_locations: np.ndarray) -> np.ndarray:
    """
    Map state to observation space
    
    In practice, this could be a neural network
    """
    # Simple extraction at observation locations
    observations = state[obs_locations]
    
    return observations

def ensemble_kalman_update(self, forecast_ensemble: np.ndarray,
                          observations: np.ndarray,
                          obs_locations: np.ndarray,
                          obs_error_std: float = 0.5) -> np.ndarray:
    """
    Ensemble Kalman filter update
    
    x_a = x_f + K(y - H(x_f))
    """
    n_obs = len(observations)
    
    # Ensemble mean
    forecast_mean = np.mean(forecast_ensemble, axis=0)
    
    # Ensemble perturbations
    forecast_pert = forecast_ensemble - forecast_mean
    
    # Predicted observations
    predicted_obs = np.zeros((self.n_ensemble, n_obs))
    for i in range(self.n_ensemble):
        predicted_obs[i] = self.observation_operator(
            forecast_ensemble[i], obs_locations
        )
    
    predicted_obs_mean = np.mean(predicted_obs, axis=0)
    predicted_obs_pert = predicted_obs - predicted_obs_mean
    
    # Kalman gain computation
    # K = P H^T (H P H^T + R)^{-1}
    PHt = forecast_pert.T @ predicted_obs_pert / (self.n_ensemble - 1)
    HPHt = predicted_obs_pert.T @ predicted_obs_pert / (self.n_ensemble - 1)
    R = obs_error_std**2 * np.eye(n_obs)
    
    K = PHt @ np.linalg.inv(HPHt + R)
    
    # Analysis ensemble
    analysis_ensemble = np.zeros_like(forecast_ensemble)
    for i in range(self.n_ensemble):
        obs_pert = np.random.randn(n_obs) * obs_error_std
        innovation = observations + obs_pert - predicted_obs[i]
        analysis_ensemble[i] = forecast_ensemble[i] + K @ innovation
    
    return analysis_ensemble
def demonstrate_ocean_modeling_framework():
"""
Demonstrate ocean modeling with AI framework
"""
print("="*70)
print("Ocean Modeling with Artificial Intelligence Framework")
print("Chapter 7: Ocean Modeling with Artificial Intelligence")
print("="*70)
# Configuration
np.random.seed(42)
grid_shape = (50, 50, 10)  # nx, ny, nz
n_species = 5

print(f"\nConfiguration:")
print(f"  Grid size: {grid_shape[0]}x{grid_shape[1]}x{grid_shape[2]}")
print(f"  Biological species: {n_species}")

# 1. Ocean Primitive Equations
print("\n" + "-"*70)
print("1. Ocean Primitive Equations Solver")
print("-"*70)

ocean_model = OceanPrimitiveEquations(
    grid_shape=grid_shape,
    dx=1000.0,  # 1 km
    dy=1000.0,
    dz=10.0,    # 10 m
    latitude=30.0
)

# Initialize velocity and pressure fields
u = np.random.randn(grid_shape[0], grid_shape[1]) * 0.1
v = np.random.randn(grid_shape[0], grid_shape[1]) * 0.1
w = np.zeros((grid_shape[0], grid_shape[1], grid_shape[2]))
p = np.random.randn(grid_shape[0], grid_shape[1]) * 100.0

print(f"Coriolis parameter (f): {ocean_model.f:.6f} s⁻¹")
print(f"Initial velocity magnitude: {np.sqrt(u**2 + v**2).mean():.4f} m/s")

# Simulate forward
dt = 100.0  # 100 seconds
n_steps = 50

for step in range(n_steps):
    u, v, w = ocean_model.step(u, v, w, p, dt)

print(f"\nAfter {n_steps} steps ({n_steps*dt/3600:.2f} hours):")
print(f"  Final velocity magnitude: {np.sqrt(u**2 + v**2).mean():.4f} m/s")
print(f"  Max velocity: {np.sqrt(u**2 + v**2).max():.4f} m/s")

# 2. Physics-Informed Ocean Network
print("\n" + "-"*70)
print("2. Physics-Informed Neural Network for Ocean")
print("-"*70)

pinn = PhysicsInformedOceanNet(input_dim=10, hidden_dim=128)

# Generate synthetic input
x_input = np.random.randn(100, 10)

# Forward pass
predictions = pinn.forward(x_input)

print(f"Predictions shape: u={predictions['u'].shape}, v={predictions['v'].shape}")
print(f"Predicted velocity range: [{predictions['u'].min():.2f}, {predictions['u'].max():.2f}]")

# Compute physics-informed loss
targets = {
    'u': np.random.randn(100) * 0.5,
    'v': np.random.randn(100) * 0.5,
    'p': np.random.randn(100) * 50.0
}

physics_params = {
    'f': ocean_model.f,
    'rho0': ocean_model.rho0,
    'dx': 1000.0,
    'dy': 1000.0
}

losses = pinn.physics_informed_loss(predictions, targets, physics_params)

print(f"\nPhysics-Informed Losses:")
print(f"  Data loss: {losses['data']:.6f}")
print(f"  Geostrophic balance: {losses['geostrophic']:.6f}")
print(f"  Continuity: {losses['continuity']:.6f}")
print(f"  Total: {losses['total']:.6f}")

# 3. Marine Biogeochemistry
print("\n" + "-"*70)
print("3. Marine Ecosystem and Biogeochemistry Modeling")
print("-"*70)

biogeo_model = MarineBiogeochemistryModel(n_species=n_species)

# Environmental conditions
sst = np.random.uniform(15, 25, size=100)  # °C
par = np.random.uniform(0, 400, size=100)  # W/m²
nitrate = np.random.uniform(0, 20, size=100)  # μmol/L
phosphate = np.random.uniform(0, 2, size=100)  # μmol/L
mld = np.random.uniform(20, 100, size=100)  # m

# Predict primary productivity
productivity = biogeo_model.primary_productivity(
    sst, par, nitrate, phosphate, mld
)

print(f"Environmental conditions:")
print(f"  SST: {sst.mean():.1f}±{sst.std():.1f} °C")
print(f"  PAR: {par.mean():.1f}±{par.std():.1f} W/m²")
print(f"  Nitrate: {nitrate.mean():.1f}±{nitrate.std():.1f} μmol/L")

print(f"\nPrimary Productivity:")
print(f"  Mean: {productivity.mean():.2f} mg C/m²/day")
print(f"  Range: [{productivity.min():.2f}, {productivity.max():.2f}]")

# Carbon cycle
carbon_pools = {
    'dissolved_organic': np.ones(100) * 10.0,
    'particulate_organic': np.ones(100) * 5.0
}

updated_pools = biogeo_model.carbon_cycle_step(
    carbon_pools, sst, dt=3600.0
)

print(f"\nCarbon Cycle (after 1 hour):")
print(f"  Dissolved organic: {updated_pools['dissolved_organic'].mean():.2f} mg C/L")
print(f"  Particulate organic: {updated_pools['particulate_organic'].mean():.2f} mg C/L")
print(f"  Export flux: {updated_pools['export_flux'].mean():.4f} mg C/m²/s")

# 4. Ocean Data Assimilation
print("\n" + "-"*70)
print("4. Ocean Data Assimilation (Ensemble Kalman Filter)")
print("-"*70)

state_dim = 500
da_system = OceanDataAssimilation(state_dim=state_dim, n_ensemble=20)

# Initial state
initial_state = np.random.randn(state_dim) * 10.0

# Generate ensemble
forecast_ensemble = da_system.generate_ensemble(
    initial_state, perturbation_std=2.0
)

print(f"Ensemble configuration:")
print(f"  State dimension: {state_dim}")
print(f"  Ensemble size: {da_system.n_ensemble}")
print(f"  Forecast spread: {forecast_ensemble.std():.3f}")

# Synthetic observations
n_obs = 50
obs_locations = np.random.choice(state_dim, n_obs, replace=False)
observations = initial_state[obs_locations] + np.random.randn(n_obs) * 0.5

print(f"\nObservations:")
print(f"  Number: {n_obs}")
print(f"  Locations: {n_obs} of {state_dim} grid points")

# Data assimilation update
analysis_ensemble = da_system.ensemble_kalman_update(
    forecast_ensemble, observations, obs_locations, obs_error_std=0.5
)

analysis_mean = np.mean(analysis_ensemble, axis=0)
analysis_spread = np.std(analysis_ensemble, axis=0).mean()

print(f"\nAssimilation results:")
print(f"  Forecast spread: {forecast_ensemble.std():.3f}")
print(f"  Analysis spread: {analysis_spread:.3f}")
print(f"  Spread reduction: {(1 - analysis_spread/forecast_ensemble.std())*100:.1f}%")

# Visualize results
visualize_ocean_modeling_results(
    u, v, productivity, updated_pools,
    forecast_ensemble, analysis_ensemble, observations, obs_locations
)

print("\n" + "="*70)
print("Ocean modeling framework demonstration complete!")
print("="*70)
def visualize_ocean_modeling_results(u, v, productivity, carbon_pools,
forecast_ens, analysis_ens, obs, obs_locs):
"""Visualize ocean modeling results"""
fig = plt.figure(figsize=(16, 12))
# Plot 1: Ocean velocity field
ax1 = plt.subplot(3, 3, 1)
speed = np.sqrt(u**2 + v**2)
im1 = ax1.imshow(speed, cmap='jet', origin='lower')
ax1.quiver(u[::5, ::5], v[::5, ::5], scale=2)
ax1.set_title('Ocean Surface Currents\n(Velocity Magnitude)')
ax1.set_xlabel('Longitude Grid')
ax1.set_ylabel('Latitude Grid')
plt.colorbar(im1, ax=ax1, label='Speed (m/s)')

# Plot 2: Primary productivity
ax2 = plt.subplot(3, 3, 2)
ax2.hist(productivity, bins=30, color='green', alpha=0.7, edgecolor='black')
ax2.axvline(productivity.mean(), color='red', linestyle='--', 
           linewidth=2, label=f'Mean: {productivity.mean():.1f}')
ax2.set_xlabel('Primary Productivity (mg C/m²/day)')
ax2.set_ylabel('Frequency')
ax2.set_title('Primary Productivity Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Carbon pools
ax3 = plt.subplot(3, 3, 3)
pools = ['Dissolved\nOrganic', 'Particulate\nOrganic']
values = [
    carbon_pools['dissolved_organic'].mean(),
    carbon_pools['particulate_organic'].mean()
]
colors = ['steelblue', 'darkorange']
bars = ax3.bar(pools, values, color=colors, alpha=0.7, edgecolor='black')
ax3.set_ylabel('Carbon (mg C/L)')
ax3.set_title('Carbon Pool Concentrations')
ax3.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}', ha='center', va='bottom')

# Plot 4: Vorticity (curl of velocity)
ax4 = plt.subplot(3, 3, 4)
# Simplified vorticity calculation
vorticity = np.gradient(v, axis=1) - np.gradient(u, axis=0)
im4 = ax4.imshow(vorticity, cmap='RdBu_r', origin='lower')
ax4.set_title('Relative Vorticity\n(∂v/∂x - ∂u/∂y)')
ax4.set_xlabel('Longitude Grid')
ax4.set_ylabel('Latitude Grid')
plt.colorbar(im4, ax=ax4, label='Vorticity (s⁻¹)')

# Plot 5: Data assimilation - state comparison
ax5 = plt.subplot(3, 3, 5)
state_indices = np.arange(len(forecast_ens.mean(axis=0)))
ax5.plot(state_indices, forecast_ens.mean(axis=0), 'b-', 
        label='Forecast Mean', alpha=0.7)
ax5.fill_between(state_indices,
                 forecast_ens.mean(axis=0) - forecast_ens.std(axis=0),
                 forecast_ens.mean(axis=0) + forecast_ens.std(axis=0),
                 alpha=0.3, color='blue')
ax5.plot(state_indices, analysis_ens.mean(axis=0), 'r-',
        label='Analysis Mean', alpha=0.7)
ax5.scatter(obs_locs, obs, c='green', marker='o', s=50,
           label='Observations', zorder=5)
ax5.set_xlabel('State Index')
ax5.set_ylabel('Value')
ax5.set_title('Data Assimilation\n(Forecast vs Analysis)')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.set_xlim([0, 100])  # Show first 100 points

# Plot 6: Ensemble spread reduction
ax6 = plt.subplot(3, 3, 6)
spread_forecast = forecast_ens.std(axis=0)
spread_analysis = analysis_ens.std(axis=0)

ax6.plot(state_indices, spread_forecast, 'b-', 
        label='Forecast Spread', linewidth=2)
ax6.plot(state_indices, spread_analysis, 'r-',
        label='Analysis Spread', linewidth=2)
ax6.set_xlabel('State Index')
ax6.set_ylabel('Ensemble Spread')
ax6.set_title('Ensemble Spread Reduction\n(Data Assimilation Impact)')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.set_xlim([0, 100])

# Plot 7: Streamlines
ax7 = plt.subplot(3, 3, 7)
Y, X = np.mgrid[0:u.shape[0], 0:u.shape[1]]
ax7.streamplot(X, Y, u, v, color=speed, cmap='viridis', density=1.5)
ax7.set_title('Ocean Circulation Streamlines')
ax7.set_xlabel('Longitude Grid')
ax7.set_ylabel('Latitude Grid')

# Plot 8: Export flux
ax8 = plt.subplot(3, 3, 8)
export_flux = carbon_pools['export_flux']
ax8.hist(export_flux, bins=25, color='brown', alpha=0.7, edgecolor='black')
ax8.axvline(export_flux.mean(), color='red', linestyle='--',
           linewidth=2, label=f'Mean: {export_flux.mean():.4f}')
ax8.set_xlabel('Export Flux (mg C/m²/s)')
ax8.set_ylabel('Frequency')
ax8.set_title('Carbon Export to Deep Ocean')
ax8.legend()
ax8.grid(True, alpha=0.3)

# Plot 9: Divergence field
ax9 = plt.subplot(3, 3, 9)
divergence = np.gradient(u, axis=1) + np.gradient(v, axis=0)
im9 = ax9.imshow(divergence, cmap='PuOr', origin='lower')
ax9.set_title('Velocity Divergence\n(∂u/∂x + ∂v/∂y)')
ax9.set_xlabel('Longitude Grid')
ax9.set_ylabel('Latitude Grid')
plt.colorbar(im9, ax=ax9, label='Divergence (s⁻¹)')

plt.tight_layout()
plt.savefig('ocean_modeling_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nResults saved to 'ocean_modeling_results.png'")
if name == "main":
demonstrate_ocean_modeling_framework()
