"""
Atmospheric Composition and Air Quality Prediction Framework
Implements ML methods for chemical transport modeling:

Graph neural networks for chemical species
Physics-informed chemical transport models
Source apportionment using VAE
Real-time air quality forecasting

Based on Chapter 6: Atmospheric Composition and Air Quality
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')
class ChemicalTransportModel:
"""
Numerical solver for atmospheric chemical transport equations
Solves: ∂c/∂t + ∇·(uc) = ∇·(K∇c) + R(c) + E - D
"""

def __init__(self, grid_size: Tuple[int, int], dx: float, dy: float):
    """
    Initialize chemical transport model
    
    Args:
        grid_size: (nx, ny) grid dimensions
        dx, dy: grid spacing in x, y directions
    """
    self.nx, self.ny = grid_size
    self.dx = dx
    self.dy = dy
    
def advection_term(self, c: np.ndarray, u: np.ndarray, 
                  v: np.ndarray) -> np.ndarray:
    """
    Compute advection term: -∇·(uc)
    
    Uses upwind differencing for stability
    """
    dc_dx = np.zeros_like(c)
    dc_dy = np.zeros_like(c)
    
    # Upwind scheme
    for i in range(1, self.nx-1):
        for j in range(1, self.ny-1):
            if u[i,j] > 0:
                dc_dx[i,j] = (c[i,j] - c[i-1,j]) / self.dx
            else:
                dc_dx[i,j] = (c[i+1,j] - c[i,j]) / self.dx
                
            if v[i,j] > 0:
                dc_dy[i,j] = (c[i,j] - c[i,j-1]) / self.dy
            else:
                dc_dy[i,j] = (c[i,j+1] - c[i,j]) / self.dy
    
    return -(u * dc_dx + v * dc_dy)

def diffusion_term(self, c: np.ndarray, K: float = 10.0) -> np.ndarray:
    """
    Compute diffusion term: ∇·(K∇c)
    
    Uses central differencing
    """
    d2c_dx2 = np.zeros_like(c)
    d2c_dy2 = np.zeros_like(c)
    
    # Central differences for second derivatives
    d2c_dx2[1:-1, 1:-1] = (c[2:, 1:-1] - 2*c[1:-1, 1:-1] + c[:-2, 1:-1]) / self.dx**2
    d2c_dy2[1:-1, 1:-1] = (c[1:-1, 2:] - 2*c[1:-1, 1:-1] + c[1:-1, :-2]) / self.dy**2
    
    return K * (d2c_dx2 + d2c_dy2)

def reaction_term(self, c: np.ndarray, k: float = 0.01) -> np.ndarray:
    """
    Simple first-order decay: R = -kc
    
    In practice, would include complex chemical mechanisms
    """
    return -k * c

def step(self, c: np.ndarray, u: np.ndarray, v: np.ndarray,
         emission: np.ndarray, dt: float, K: float = 10.0,
         k_decay: float = 0.01) -> np.ndarray:
    """
    Single time step using operator splitting
    """
    # Advection
    c = c + dt * self.advection_term(c, u, v)
    
    # Diffusion
    c = c + dt * self.diffusion_term(c, K)
    
    # Reaction
    c = c + dt * self.reaction_term(c, k_decay)
    
    # Emission
    c = c + dt * emission
    
    # Ensure non-negative concentrations
    c = np.maximum(c, 0)
    
    return c
class ChemicalGNN:
"""
Graph Neural Network for multi-species chemical modeling
Represents chemical species as nodes and reactions as edges
"""

def __init__(self, n_species: int, hidden_dim: int = 64):
    """
    Initialize chemical GNN
    
    Args:
        n_species: Number of chemical species
        hidden_dim: Hidden dimension size
    """
    self.n_species = n_species
    self.hidden_dim = hidden_dim
    
    # Placeholder for learnable parameters
    # In practice, use PyTorch or similar
    self.node_features = np.random.randn(n_species, hidden_dim)
    self.edge_weights = np.random.randn(n_species, n_species)
    
def message_passing(self, concentrations: np.ndarray,
                   adjacency: np.ndarray) -> np.ndarray:
    """
    Perform message passing based on chemical connectivity
    
    Args:
        concentrations: Current species concentrations
        adjacency: Chemical reaction adjacency matrix
        
    Returns:
        Updated concentrations
    """
    # Aggregate messages from neighboring species
    messages = np.zeros_like(concentrations)
    
    for i in range(self.n_species):
        for j in range(self.n_species):
            if adjacency[i, j] > 0:
                # Message from species j to species i
                messages[i] += (adjacency[i, j] * concentrations[j])
    
    # Update with aggregated messages
    updated = concentrations + 0.1 * messages
    
    return np.maximum(updated, 0)  # Non-negative

def forward(self, concentrations: np.ndarray,
           reaction_network: np.ndarray,
           n_steps: int = 5) -> np.ndarray:
    """
    Forward pass through chemical GNN
    
    Args:
        concentrations: Initial concentrations
        reaction_network: Adjacency matrix for reactions
        n_steps: Number of message passing steps
        
    Returns:
        Final concentrations after GNN processing
    """
    current = concentrations.copy()
    
    for _ in range(n_steps):
        current = self.message_passing(current, reaction_network)
    
    return current
class SourceApportionmentVAE:
"""
Variational Autoencoder for source apportionment
Learns latent representations of emission sources
"""

def __init__(self, n_species: int, n_sources: int, latent_dim: int = 10):
    """
    Initialize source apportionment VAE
    
    Args:
        n_species: Number of chemical species
        n_sources: Number of emission sources
        latent_dim: Latent space dimension
    """
    self.n_species = n_species
    self.n_sources = n_sources
    self.latent_dim = latent_dim
    
    # Simplified encoder/decoder (placeholder)
    self.encoder_w = np.random.randn(n_species, latent_dim) * 0.1
    self.decoder_w = np.random.randn(latent_dim, n_species) * 0.1
    
    # Source profiles (learned parameters)
    self.source_profiles = np.random.rand(n_sources, n_species)
    self.source_profiles /= self.source_profiles.sum(axis=1, keepdims=True)
    
def encode(self, concentrations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode observations to latent source contributions
    
    Returns:
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
    """
    # Simple linear encoding (placeholder)
    encoded = np.dot(concentrations, self.encoder_w)
    
    mu = encoded
    logvar = np.ones_like(encoded) * -2  # Small variance
    
    return mu, logvar

def reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
    """
    Reparameterization trick for VAE
    """
    std = np.exp(0.5 * logvar)
    eps = np.random.randn(*std.shape)
    return mu + eps * std

def decode(self, z: np.ndarray) -> np.ndarray:
    """
    Decode latent variables to concentration reconstruction
    """
    return np.dot(z, self.decoder_w)

def apportion_sources(self, concentrations: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Perform source apportionment
    
    Args:
        concentrations: Observed concentrations
        
    Returns:
        Dictionary with source contributions
    """
    # Encode to latent space
    mu, logvar = self.encode(concentrations)
    z = self.reparameterize(mu, logvar)
    
    # Compute source contributions (simplified)
    # In practice, this would involve more sophisticated decoding
    contributions = np.abs(z[:self.n_sources])
    contributions /= contributions.sum()
    
    # Reconstruct concentrations
    reconstructed = self.decode(z)
    
    return {
        'contributions': contributions,
        'reconstructed': reconstructed,
        'latent': z,
        'uncertainty': np.exp(0.5 * logvar)
    }
class RealTimeAQPredictor:
"""
Real-time air quality prediction system
Implements online learning and streaming data processing
"""

def __init__(self, n_species: int, buffer_size: int = 1440):
    """
    Initialize real-time predictor
    
    Args:
        n_species: Number of pollutant species
        buffer_size: Size of data buffer (e.g., 24 hours)
    """
    self.n_species = n_species
    self.buffer_size = buffer_size
    
    # Circular buffer for streaming data
    self.data_buffer = np.zeros((buffer_size, n_species))
    self.time_buffer = np.zeros(buffer_size)
    self.buffer_index = 0
    self.buffer_full = False
    
    # Simple linear model (placeholder)
    self.weights = np.random.randn(n_species, n_species) * 0.1
    
def add_observation(self, observation: np.ndarray, timestamp: float):
    """
    Add new observation to streaming buffer
    """
    self.data_buffer[self.buffer_index] = observation
    self.time_buffer[self.buffer_index] = timestamp
    
    self.buffer_index = (self.buffer_index + 1) % self.buffer_size
    if self.buffer_index == 0:
        self.buffer_full = True

def online_update(self, observation: np.ndarray, 
                 prediction: np.ndarray, learning_rate: float = 0.01):
    """
    Online learning update using gradient descent
    
    θ_{t+1} = θ_t - η∇L
    """
    error = observation - prediction
    
    # Simple gradient update (placeholder)
    gradient = -2 * error[:, np.newaxis] * self.get_recent_average()
    self.weights -= learning_rate * gradient
    
def get_recent_average(self) -> np.ndarray:
    """
    Get recent average for online learning
    """
    if self.buffer_full:
        return np.mean(self.data_buffer[-100:], axis=0)
    elif self.buffer_index > 10:
        return np.mean(self.data_buffer[:self.buffer_index], axis=0)
    else:
        return np.ones(self.n_species)

def predict(self, current_state: np.ndarray, 
           meteorology: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Generate real-time prediction
    
    Returns:
        Dictionary with predictions and uncertainty
    """
    # Simple linear prediction (placeholder)
    prediction = np.dot(current_state, self.weights)
    
    # Uncertainty based on recent prediction errors
    if self.buffer_full or self.buffer_index > 50:
        recent_data = self.data_buffer[-50:] if self.buffer_full else self.data_buffer[:self.buffer_index]
        uncertainty = np.std(recent_data, axis=0)
    else:
        uncertainty = np.ones(self.n_species) * 0.5
    
    return {
        'prediction': prediction,
        'uncertainty': uncertainty,
        'confidence': 1.0 / (1.0 + uncertainty)
    }
def demonstrate_atmospheric_composition_framework():
"""
Demonstrate atmospheric composition modeling framework
"""
print("="*70)
print("Atmospheric Composition and Air Quality Framework")
print("Chapter 6: Atmospheric Composition and Air Quality")
print("="*70)
# Setup parameters
np.random.seed(42)
grid_size = (50, 50)
n_species = 5
n_sources = 3

print(f"\nConfiguration:")
print(f"  Grid size: {grid_size[0]}x{grid_size[1]}")
print(f"  Number of species: {n_species}")
print(f"  Number of sources: {n_sources}")

# 1. Chemical Transport Modeling
print("\n" + "-"*70)
print("1. Chemical Transport Model (Advection-Diffusion-Reaction)")
print("-"*70)

transport_model = ChemicalTransportModel(grid_size, dx=1.0, dy=1.0)

# Initial concentration field
c_init = np.zeros(grid_size)
c_init[20:30, 20:30] = 100.0  # Initial pollution hotspot

# Wind field (simple uniform flow)
u = np.ones(grid_size) * 2.0  # 2 m/s eastward
v = np.ones(grid_size) * 1.0  # 1 m/s northward

# Emission source
emission = np.zeros(grid_size)
emission[10, 10] = 50.0  # Point source

print(f"Initial max concentration: {c_init.max():.2f} μg/m³")
print(f"Wind speed: u={u[0,0]:.1f}, v={v[0,0]:.1f} m/s")
print(f"Emission rate: {emission.max():.2f} μg/m³/s")

# Simulate forward in time
c_current = c_init.copy()
dt = 0.1
n_steps = 100

for step in range(n_steps):
    c_current = transport_model.step(c_current, u, v, emission, dt)

print(f"\nAfter {n_steps} steps:")
print(f"  Max concentration: {c_current.max():.2f} μg/m³")
print(f"  Mean concentration: {c_current.mean():.2f} μg/m³")
print(f"  Total mass: {c_current.sum():.2f}")

# 2. Graph Neural Network for Chemistry
print("\n" + "-"*70)
print("2. Graph Neural Network for Multi-Species Chemistry")
print("-"*70)

chem_gnn = ChemicalGNN(n_species=n_species, hidden_dim=32)

# Initial concentrations for multiple species
concentrations = np.array([100.0, 50.0, 30.0, 20.0, 10.0])

# Chemical reaction network (adjacency matrix)
reaction_network = np.array([
    [0, 1, 0, 0, 0],  # Species 0 reacts with species 1
    [1, 0, 1, 0, 0],  # Species 1 reacts with 0 and 2
    [0, 1, 0, 1, 0],  # Species 2 reacts with 1 and 3
    [0, 0, 1, 0, 1],  # Species 3 reacts with 2 and 4
    [0, 0, 0, 1, 0],  # Species 4 reacts with 3
])

print(f"Initial concentrations: {concentrations}")
print(f"Reaction network connections: {reaction_network.sum():.0f}")

# Process through GNN
updated_concentrations = chem_gnn.forward(
    concentrations, reaction_network, n_steps=5
)

print(f"\nAfter GNN processing:")
print(f"  Updated concentrations: {updated_concentrations}")
print(f"  Change: {updated_concentrations - concentrations}")

# 3. Source Apportionment
print("\n" + "-"*70)
print("3. Source Apportionment using VAE")
print("-"*70)

source_model = SourceApportionmentVAE(
    n_species=n_species,
    n_sources=n_sources,
    latent_dim=8
)

# Observed concentrations (mixture from multiple sources)
observed = np.array([75.0, 45.0, 60.0, 30.0, 25.0])

print(f"Observed concentrations: {observed}")
print(f"Source profiles shape: {source_model.source_profiles.shape}")

# Perform source apportionment
results = source_model.apportion_sources(observed)

print(f"\nSource Contributions:")
for i, contrib in enumerate(results['contributions']):
    print(f"  Source {i+1}: {contrib*100:.1f}%")

print(f"\nReconstruction error: {np.mean(np.abs(observed - results['reconstructed'])):.2f}")
print(f"Uncertainty: {np.mean(results['uncertainty']):.3f}")

# 4. Real-Time Prediction
print("\n" + "-"*70)
print("4. Real-Time Air Quality Forecasting")
print("-"*70)

rt_predictor = RealTimeAQPredictor(n_species=n_species, buffer_size=1440)

# Simulate streaming observations
print("Simulating 200 time steps of streaming data...")

for t in range(200):
    # Generate synthetic observation with trend and noise
    trend = 50 + 30 * np.sin(2 * np.pi * t / 100)
    noise = np.random.randn(n_species) * 5
    observation = np.maximum(0, trend + noise)
    
    rt_predictor.add_observation(observation, timestamp=t)
    
    # Make prediction
    if t > 10:
        current_state = observation
        forecast = rt_predictor.predict(current_state)
        
        # Online update
        if t % 10 == 0:
            rt_predictor.online_update(observation, forecast['prediction'])

print(f"Buffer filled: {rt_predictor.buffer_full}")
print(f"Current buffer index: {rt_predictor.buffer_index}")

# Final prediction
final_forecast = rt_predictor.predict(observation)
print(f"\nFinal Prediction:")
print(f"  Forecast: {final_forecast['prediction']}")
print(f"  Uncertainty: {final_forecast['uncertainty']}")
print(f"  Confidence: {final_forecast['confidence']}")

# Visualize results
visualize_atmospheric_composition_results(
    c_current, concentrations, updated_concentrations,
    results['contributions'], final_forecast
)

print("\n" + "="*70)
print("Atmospheric composition framework demonstration complete!")
print("="*70)
def visualize_atmospheric_composition_results(transport_field, init_conc,
updated_conc, source_contrib, forecast):
"""Visualize atmospheric composition modeling results"""
fig = plt.figure(figsize=(16, 10))
# Plot 1: Chemical transport field
ax1 = plt.subplot(2, 3, 1)
im1 = ax1.imshow(transport_field, cmap='YlOrRd', origin='lower')
ax1.set_title('Chemical Transport Field\n(After Advection-Diffusion-Reaction)')
ax1.set_xlabel('X (grid points)')
ax1.set_ylabel('Y (grid points)')
plt.colorbar(im1, ax=ax1, label='Concentration (μg/m³)')

# Plot 2: Species concentrations comparison
ax2 = plt.subplot(2, 3, 2)
species = np.arange(len(init_conc))
width = 0.35
ax2.bar(species - width/2, init_conc, width, label='Initial', alpha=0.8)
ax2.bar(species + width/2, updated_conc, width, label='After GNN', alpha=0.8)
ax2.set_xlabel('Chemical Species')
ax2.set_ylabel('Concentration (μg/m³)')
ax2.set_title('Multi-Species Chemical Evolution\n(Graph Neural Network)')
ax2.legend()
ax2.set_xticks(species)
ax2.set_xticklabels([f'S{i+1}' for i in species])
ax2.grid(True, alpha=0.3)

# Plot 3: Source apportionment
ax3 = plt.subplot(2, 3, 3)
sources = np.arange(len(source_contrib))
bars = ax3.bar(sources, source_contrib * 100, color='steelblue', alpha=0.8)
ax3.set_xlabel('Emission Source')
ax3.set_ylabel('Contribution (%)')
ax3.set_title('Source Apportionment\n(Variational Autoencoder)')
ax3.set_xticks(sources)
ax3.set_xticklabels([f'Source {i+1}' for i in sources])
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom')

# Plot 4: Real-time prediction with uncertainty
ax4 = plt.subplot(2, 3, 4)
species_idx = np.arange(len(forecast['prediction']))
pred = forecast['prediction']
unc = forecast['uncertainty']

ax4.errorbar(species_idx, pred, yerr=unc, fmt='o-', capsize=5,
            linewidth=2, markersize=8, label='Prediction ± Uncertainty')
ax4.set_xlabel('Pollutant Species')
ax4.set_ylabel('Concentration (μg/m³)')
ax4.set_title('Real-Time Forecast\n(with Uncertainty Bounds)')
ax4.set_xticks(species_idx)
ax4.set_xticklabels([f'P{i+1}' for i in species_idx])
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Confidence levels
ax5 = plt.subplot(2, 3, 5)
confidence = forecast['confidence']
colors = plt.cm.RdYlGn(confidence)
bars = ax5.bar(species_idx, confidence, color=colors, alpha=0.8)
ax5.set_xlabel('Pollutant Species')
ax5.set_ylabel('Confidence')
ax5.set_title('Forecast Confidence\n(Higher is Better)')
ax5.set_xticks(species_idx)
ax5.set_xticklabels([f'P{i+1}' for i in species_idx])
ax5.set_ylim([0, 1])
ax5.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Good confidence')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Transport field gradient
ax6 = plt.subplot(2, 3, 6)
gradient_magnitude = np.sqrt(
    np.gradient(transport_field, axis=0)**2 + 
    np.gradient(transport_field, axis=1)**2
)
im6 = ax6.imshow(gradient_magnitude, cmap='viridis', origin='lower')
ax6.set_title('Concentration Gradient\n(Transport Intensity)')
ax6.set_xlabel('X (grid points)')
ax6.set_ylabel('Y (grid points)')
plt.colorbar(im6, ax=ax6, label='Gradient Magnitude')

plt.tight_layout()
plt.savefig('atmospheric_composition_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nResults saved to 'atmospheric_composition_results.png'")
if name == "main":
demonstrate_atmospheric_composition_framework()
