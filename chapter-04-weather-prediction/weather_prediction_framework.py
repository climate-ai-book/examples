"""
AI-Enhanced Weather Prediction Framework
Implements fundamental components for weather forecasting using deep learning:

ConvLSTM for spatiotemporal precipitation nowcasting
Physics-informed loss functions
Ensemble uncertainty quantification
Forecast verification metrics

Based on Chapter 4: AI-Enhanced Weather Prediction
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')
class ConvLSTMCell:
"""
Convolutional LSTM cell for spatiotemporal weather modeling
Implements the ConvLSTM equations:
f_t = σ(W_xf * X_t + W_hf * H_{t-1} + b_f)  # Forget gate
i_t = σ(W_xi * X_t + W_hi * H_{t-1} + b_i)  # Input gate
o_t = σ(W_xo * X_t + W_ho * H_{t-1} + b_o)  # Output gate
g_t = tanh(W_xg * X_t + W_hg * H_{t-1} + b_g)  # Candidate
C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t  # Cell state
H_t = o_t ⊙ tanh(C_t)  # Hidden state
"""

def __init__(self, input_channels: int, hidden_channels: int, 
             kernel_size: int = 3):
    """
    Initialize ConvLSTM cell
    
    Args:
        input_channels: Number of input channels
        hidden_channels: Number of hidden channels
        kernel_size: Convolution kernel size
    """
    self.input_channels = input_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    
    # In practice, these would be learned parameters
    # Here we use placeholder initialization
    self.initialized = False
    
def _sigmoid(self, x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def _tanh(self, x: np.ndarray) -> np.ndarray:
    """Tanh activation function"""
    return np.tanh(x)

def _conv2d_simple(self, x: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Simplified 2D convolution (placeholder)
    In practice, use proper convolution implementation
    """
    # Simple averaging convolution as placeholder
    from scipy.ndimage import uniform_filter
    return uniform_filter(x, size=(1, 1, kernel_size, kernel_size))

def forward(self, x_t: np.ndarray, h_prev: np.ndarray, 
            c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward pass through ConvLSTM cell
    
    Args:
        x_t: Input at time t (batch, channels, height, width)
        h_prev: Previous hidden state
        c_prev: Previous cell state
        
    Returns:
        h_t: New hidden state
        c_t: New cell state
    """
    # Simplified implementation - in practice use proper convolutions
    # Placeholder gate computations
    f_t = self._sigmoid(x_t + h_prev)  # Forget gate
    i_t = self._sigmoid(x_t + h_prev)  # Input gate
    o_t = self._sigmoid(x_t + h_prev)  # Output gate
    g_t = self._tanh(x_t + h_prev)     # Candidate values
    
    # Update cell state
    c_t = f_t * c_prev + i_t * g_t
    
    # Compute hidden state
    h_t = o_t * self._tanh(c_t)
    
    return h_t, c_t
class WeatherNowcastingModel:
"""
ConvLSTM-based weather nowcasting model for precipitation prediction
"""
def __init__(self, input_shape: Tuple[int, int], 
             hidden_channels: int = 64,
             sequence_length: int = 5):
    """
    Initialize nowcasting model
    
    Args:
        input_shape: Spatial dimensions (height, width)
        hidden_channels: Number of hidden channels in ConvLSTM
        sequence_length: Input sequence length
    """
    self.input_shape = input_shape
    self.hidden_channels = hidden_channels
    self.sequence_length = sequence_length
    
    self.convlstm = ConvLSTMCell(1, hidden_channels)
    
def predict(self, x_sequence: np.ndarray, n_steps: int = 3) -> np.ndarray:
    """
    Predict future frames
    
    Args:
        x_sequence: Input sequence (time, height, width)
        n_steps: Number of future steps to predict
        
    Returns:
        predictions: Predicted frames (n_steps, height, width)
    """
    # Initialize hidden and cell states
    h, w = self.input_shape
    h_state = np.zeros((1, self.hidden_channels, h, w))
    c_state = np.zeros((1, self.hidden_channels, h, w))
    
    # Process input sequence
    for t in range(len(x_sequence)):
        x_t = x_sequence[t:t+1, np.newaxis, :, :]
        h_state, c_state = self.convlstm.forward(x_t, h_state, c_state)
    
    # Generate predictions
    predictions = []
    for _ in range(n_steps):
        # Use last hidden state to predict next frame
        # Simplified - in practice, add output projection layer
        pred = np.mean(h_state, axis=1, keepdims=True)
        predictions.append(pred[0, 0])
        
        # Update states for next prediction
        h_state, c_state = self.convlstm.forward(pred, h_state, c_state)
    
    return np.array(predictions)
class PhysicsInformedLoss:
"""
Physics-informed loss functions for atmospheric modeling
Implements conservation law constraints and physical consistency checks
"""

def __init__(self, lambda_pde: float = 0.1, 
             lambda_conservation: float = 0.1):
    """
    Initialize physics-informed loss
    
    Args:
        lambda_pde: Weight for PDE residual loss
        lambda_conservation: Weight for conservation law violations
    """
    self.lambda_pde = lambda_pde
    self.lambda_conservation = lambda_conservation
    
def continuity_residual(self, u: np.ndarray, v: np.ndarray, 
                       omega: np.ndarray, dx: float, dy: float, 
                       dp: float) -> np.ndarray:
    """
    Compute continuity equation residual
    
    Continuity: ∂u/∂x + ∂v/∂y + ∂ω/∂p = 0
    """
    # Compute spatial derivatives using finite differences
    du_dx = np.gradient(u, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)
    domega_dp = np.gradient(omega, dp, axis=2) if omega.ndim > 2 else 0
    
    # Residual
    residual = du_dx + dv_dy + domega_dp
    
    return np.mean(residual**2)

def mass_conservation_residual(self, q: np.ndarray, u: np.ndarray, 
                               v: np.ndarray, dt: float, 
                               dx: float, dy: float) -> float:
    """
    Compute mass conservation residual for moisture
    
    ∂q/∂t + ∂(qu)/∂x + ∂(qv)/∂y = 0 (simplified)
    """
    # Temporal derivative
    dq_dt = np.gradient(q, dt, axis=0) if q.shape[0] > 1 else 0
    
    # Spatial derivatives of fluxes
    d_qux = np.gradient(q * u, dx, axis=2 if q.ndim > 2 else 1)
    d_qvy = np.gradient(q * v, dy, axis=1 if q.ndim > 2 else 0)
    
    # Residual
    residual = dq_dt + d_qux + d_qvy
    
    return np.mean(residual**2)

def compute_total_loss(self, predictions: np.ndarray, 
                      targets: np.ndarray,
                      meteorological_fields: dict) -> float:
    """
    Compute total physics-informed loss
    
    Args:
        predictions: Model predictions
        targets: Ground truth
        meteorological_fields: Dictionary with u, v, omega, etc.
        
    Returns:
        Total loss value
    """
    # Data fitting loss (MSE)
    data_loss = np.mean((predictions - targets)**2)
    
    # Physics-informed losses
    pde_loss = 0.0
    if 'u' in meteorological_fields and 'v' in meteorological_fields:
        u = meteorological_fields['u']
        v = meteorological_fields['v']
        omega = meteorological_fields.get('omega', np.zeros_like(u))
        
        pde_loss = self.continuity_residual(
            u, v, omega, dx=1.0, dy=1.0, dp=1.0
        )
    
    conservation_loss = 0.0
    if 'q' in meteorological_fields:
        q = meteorological_fields['q']
        u = meteorological_fields.get('u', np.zeros_like(q))
        v = meteorological_fields.get('v', np.zeros_like(q))
        
        conservation_loss = self.mass_conservation_residual(
            q, u, v, dt=1.0, dx=1.0, dy=1.0
        )
    
    # Total loss
    total_loss = (data_loss + 
                 self.lambda_pde * pde_loss + 
                 self.lambda_conservation * conservation_loss)
    
    return total_loss
class EnsembleForecastSystem:
"""
Ensemble forecasting system with uncertainty quantification
Implements ensemble generation, spread calculation, and probabilistic metrics
"""

def __init__(self, n_members: int = 20):
    """
    Initialize ensemble system
    
    Args:
        n_members: Number of ensemble members
    """
    self.n_members = n_members
    
def generate_ensemble(self, initial_state: np.ndarray, 
                     perturbation_std: float = 0.1) -> np.ndarray:
    """
    Generate ensemble by perturbing initial conditions
    
    Args:
        initial_state: Base initial state
        perturbation_std: Standard deviation of perturbations
        
    Returns:
        Ensemble of initial states
    """
    ensemble = np.zeros((self.n_members,) + initial_state.shape)
    
    for i in range(self.n_members):
        perturbation = np.random.randn(*initial_state.shape) * perturbation_std
        ensemble[i] = initial_state + perturbation
    
    return ensemble

def compute_ensemble_statistics(self, 
                                ensemble_forecasts: np.ndarray) -> dict:
    """
    Compute ensemble mean and spread
    
    Args:
        ensemble_forecasts: Array of ensemble forecasts (members, time, ...)
        
    Returns:
        Dictionary with mean, spread, and percentiles
    """
    ensemble_mean = np.mean(ensemble_forecasts, axis=0)
    ensemble_spread = np.std(ensemble_forecasts, axis=0)
    
    # Compute percentiles
    percentiles = {
        'p10': np.percentile(ensemble_forecasts, 10, axis=0),
        'p25': np.percentile(ensemble_forecasts, 25, axis=0),
        'p50': np.percentile(ensemble_forecasts, 50, axis=0),
        'p75': np.percentile(ensemble_forecasts, 75, axis=0),
        'p90': np.percentile(ensemble_forecasts, 90, axis=0),
    }
    
    return {
        'mean': ensemble_mean,
        'spread': ensemble_spread,
        'percentiles': percentiles
    }

def crps(self, ensemble_forecast: np.ndarray, 
         observation: np.ndarray) -> float:
    """
    Compute Continuous Ranked Probability Score (CRPS)
    
    CRPS = (1/m)Σ|x_i - x_o| - (1/2m²)Σ_i Σ_j |x_i - x_j|
    """
    m = len(ensemble_forecast)
    
    # First term: mean absolute error
    term1 = np.mean(np.abs(ensemble_forecast - observation))
    
    # Second term: ensemble spread
    term2 = 0.0
    for i in range(m):
        for j in range(m):
            term2 += np.abs(ensemble_forecast[i] - ensemble_forecast[j])
    term2 /= (2 * m * m)
    
    crps = term1 - term2
    
    return crps
class ForecastVerification:
"""
Comprehensive forecast verification metrics
"""
@staticmethod
def mae(forecast: np.ndarray, observation: np.ndarray) -> float:
    """Mean Absolute Error"""
    return np.mean(np.abs(forecast - observation))

@staticmethod
def rmse(forecast: np.ndarray, observation: np.ndarray) -> float:
    """Root Mean Square Error"""
    return np.sqrt(np.mean((forecast - observation)**2))

@staticmethod
def correlation(forecast: np.ndarray, observation: np.ndarray) -> float:
    """Pearson correlation coefficient"""
    f_flat = forecast.flatten()
    o_flat = observation.flatten()
    
    return np.corrcoef(f_flat, o_flat)[0, 1]

@staticmethod
def anomaly_correlation(forecast: np.ndarray, observation: np.ndarray,
                       climatology: np.ndarray) -> float:
    """
    Anomaly Correlation Coefficient (ACC)
    
    ACC = Σ(f_i - c̄)(o_i - c̄)/√[Σ(f_i - c̄)² Σ(o_i - c̄)²]
    """
    f_anom = forecast - climatology
    o_anom = observation - climatology
    
    numerator = np.sum(f_anom * o_anom)
    denominator = np.sqrt(np.sum(f_anom**2) * np.sum(o_anom**2))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator

@staticmethod
def brier_score(forecast_prob: np.ndarray, 
               observation_binary: np.ndarray) -> float:
    """
    Brier Score for probabilistic forecasts
    
    BS = (1/n)Σ(p_i - o_i)²
    """
    return np.mean((forecast_prob - observation_binary)**2)
def demonstrate_weather_prediction_framework():
"""
Demonstrate AI-enhanced weather prediction framework
"""
print("="*70)
print("AI-Enhanced Weather Prediction Framework")
print("Chapter 4: AI-Enhanced Weather Prediction")
print("="*70)
# Generate synthetic radar/precipitation data
np.random.seed(42)
grid_size = 64
sequence_length = 10

# Create synthetic precipitation field with spatial structure
def generate_precipitation_field(t, size=64):
    """Generate synthetic precipitation with spatiotemporal evolution"""
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)
    
    # Moving precipitation system
    center_x = -2 + 0.3 * t
    center_y = 0 + 0.1 * t
    
    # Gaussian-shaped precipitation
    precip = 10 * np.exp(-((X - center_x)**2 + (Y - center_y)**2) / 2)
    precip += np.random.randn(size, size) * 0.5  # Add noise
    precip = np.maximum(precip, 0)  # Non-negative
    
    return precip

# Generate sequence
precip_sequence = np.array([
    generate_precipitation_field(t, grid_size) 
    for t in range(sequence_length)
])

print(f"\nSynthetic Data: {sequence_length} time steps, {grid_size}x{grid_size} grid")
print(f"Precipitation range: [{precip_sequence.min():.2f}, {precip_sequence.max():.2f}] mm/hr")

# 1. ConvLSTM Nowcasting
print("\n" + "-"*70)
print("1. ConvLSTM Nowcasting Model")
print("-"*70)

model = WeatherNowcastingModel(
    input_shape=(grid_size, grid_size),
    hidden_channels=32,
    sequence_length=5
)

# Use first 7 frames as input, predict next 3
input_seq = precip_sequence[:7]
target_seq = precip_sequence[7:]

print(f"Input sequence: {input_seq.shape}")
print(f"Predicting {len(target_seq)} future frames...")

predictions = model.predict(input_seq, n_steps=3)

print(f"Predictions shape: {predictions.shape}")
print(f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")

# 2. Physics-Informed Loss
print("\n" + "-"*70)
print("2. Physics-Informed Loss Computation")
print("-"*70)

physics_loss = PhysicsInformedLoss(lambda_pde=0.1, lambda_conservation=0.1)

# Create synthetic meteorological fields
u_field = np.random.randn(grid_size, grid_size) * 5  # Wind u-component
v_field = np.random.randn(grid_size, grid_size) * 5  # Wind v-component
q_field = precip_sequence[0] / 10  # Specific humidity proxy

met_fields = {
    'u': u_field,
    'v': v_field,
    'q': q_field
}

total_loss = physics_loss.compute_total_loss(
    predictions, target_seq, met_fields
)

print(f"Physics-informed total loss: {total_loss:.6f}")
print("Loss components:")
print(f"  - Data loss (MSE): {np.mean((predictions - target_seq)**2):.6f}")
print(f"  - PDE residual: Enforces continuity equation")
print(f"  - Conservation: Enforces mass conservation")

# 3. Ensemble Forecasting
print("\n" + "-"*70)
print("3. Ensemble Forecasting and Uncertainty Quantification")
print("-"*70)

ensemble_system = EnsembleForecastSystem(n_members=20)

# Generate ensemble of initial conditions
ensemble_ic = ensemble_system.generate_ensemble(
    precip_sequence[6], perturbation_std=0.5
)

print(f"Ensemble size: {ensemble_system.n_members} members")
print(f"IC perturbation std: 0.5 mm/hr")

# Simulate ensemble forecasts (simplified)
ensemble_forecasts = []
for member in ensemble_ic:
    # In practice, run full model for each member
    # Here, add simple evolution
    forecast = member + np.random.randn(*member.shape) * 0.3
    ensemble_forecasts.append(forecast)

ensemble_forecasts = np.array(ensemble_forecasts)

# Compute statistics
stats = ensemble_system.compute_ensemble_statistics(ensemble_forecasts)

print(f"\nEnsemble statistics:")
print(f"  Mean precipitation: {stats['mean'].mean():.2f} mm/hr")
print(f"  Ensemble spread: {stats['spread'].mean():.2f} mm/hr")
print(f"  10th percentile: {stats['percentiles']['p10'].mean():.2f} mm/hr")
print(f"  90th percentile: {stats['percentiles']['p90'].mean():.2f} mm/hr")

# CRPS
observation = precip_sequence[7]
crps_score = ensemble_system.crps(
    ensemble_forecasts.reshape(ensemble_system.n_members, -1).mean(axis=1),
    observation.flatten().mean()
)
print(f"\nCRPS: {crps_score:.4f}")

# 4. Forecast Verification
print("\n" + "-"*70)
print("4. Forecast Verification Metrics")
print("-"*70)

verifier = ForecastVerification()

mae = verifier.mae(predictions, target_seq)
rmse = verifier.rmse(predictions, target_seq)
corr = verifier.correlation(predictions, target_seq)

print(f"Deterministic verification:")
print(f"  MAE: {mae:.4f} mm/hr")
print(f"  RMSE: {rmse:.4f} mm/hr")
print(f"  Correlation: {corr:.4f}")

# Anomaly correlation (using climatology as reference)
climatology = np.mean(precip_sequence, axis=0)
acc = verifier.anomaly_correlation(predictions[0], target_seq[0], climatology)
print(f"  Anomaly Correlation: {acc:.4f}")

# Visualize results
visualize_weather_predictions(
    input_seq, predictions, target_seq, 
    ensemble_forecasts, stats
)

print("\n" + "="*70)
print("Weather prediction framework demonstration complete!")
print("="*70)
def visualize_weather_predictions(input_seq, predictions, targets,
ensemble, ensemble_stats):
"""Visualize weather prediction results"""
fig = plt.figure(figsize=(16, 10))
# Plot 1: Input sequence
ax1 = plt.subplot(2, 3, 1)
im1 = ax1.imshow(input_seq[-1], cmap='YlGnBu', vmin=0, vmax=10)
ax1.set_title('Last Input Frame (t=0)')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
plt.colorbar(im1, ax=ax1, label='Precipitation (mm/hr)')

# Plot 2: Prediction
ax2 = plt.subplot(2, 3, 2)
im2 = ax2.imshow(predictions[0], cmap='YlGnBu', vmin=0, vmax=10)
ax2.set_title('Prediction (t+1)')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
plt.colorbar(im2, ax=ax2, label='Precipitation (mm/hr)')

# Plot 3: Observation
ax3 = plt.subplot(2, 3, 3)
im3 = ax3.imshow(targets[0], cmap='YlGnBu', vmin=0, vmax=10)
ax3.set_title('Observation (t+1)')
ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')
plt.colorbar(im3, ax=ax3, label='Precipitation (mm/hr)')

# Plot 4: Ensemble mean
ax4 = plt.subplot(2, 3, 4)
im4 = ax4.imshow(ensemble_stats['mean'], cmap='YlGnBu', vmin=0, vmax=10)
ax4.set_title('Ensemble Mean')
ax4.set_xlabel('Longitude')
ax4.set_ylabel('Latitude')
plt.colorbar(im4, ax=ax4, label='Precipitation (mm/hr)')

# Plot 5: Ensemble spread
ax5 = plt.subplot(2, 3, 5)
im5 = ax5.imshow(ensemble_stats['spread'], cmap='Reds', vmin=0)
ax5.set_title('Ensemble Spread (Uncertainty)')
ax5.set_xlabel('Longitude')
ax5.set_ylabel('Latitude')
plt.colorbar(im5, ax=ax5, label='Std Dev (mm/hr)')

# Plot 6: Error
ax6 = plt.subplot(2, 3, 6)
error = np.abs(predictions[0] - targets[0])
im6 = ax6.imshow(error, cmap='Reds', vmin=0)
ax6.set_title('Prediction Error')
ax6.set_xlabel('Longitude')
ax6.set_ylabel('Latitude')
plt.colorbar(im6, ax=ax6, label='Absolute Error (mm/hr)')

plt.tight_layout()
plt.savefig('weather_prediction_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nResults saved to 'weather_prediction_results.png'")
if name == "main":
demonstrate_weather_prediction_framework()
