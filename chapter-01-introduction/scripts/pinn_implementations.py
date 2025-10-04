
"""
Physics-Informed Neural Networks for Climate Applications
This module implements PINNs that incorporate physical constraints
for atmospheric and oceanic modeling.
Based on Chapter 1: Introduction to AI in Climate Science
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Callable
import matplotlib.pyplot as plt
class PhysicsInformedNN:
"""
Physics-Informed Neural Network base class
Incorporates PDE constraints into the neural network training process.
"""

def __init__(self, layers: list, lambda_pde: float = 1.0, 
             lambda_bc: float = 1.0, lambda_ic: float = 1.0):
    """
    Initialize PINN
    
    Args:
        layers: List of layer sizes [input_dim, hidden1, hidden2, ..., output_dim]
        lambda_pde: Weight for PDE loss term
        lambda_bc: Weight for boundary condition loss
        lambda_ic: Weight for initial condition loss
    """
    self.layers = layers
    self.lambda_pde = lambda_pde
    self.lambda_bc = lambda_bc
    self.lambda_ic = lambda_ic
    
    self.model = self._build_network()
    self.optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
def _build_network(self) -> keras.Model:
    """Build the neural network architecture"""
    inputs = keras.Input(shape=(self.layers[0],))
    x = inputs
    
    for units in self.layers[1:-1]:
        x = keras.layers.Dense(units, activation='tanh',
                              kernel_initializer='glorot_normal')(x)
    
    outputs = keras.layers.Dense(self.layers[-1], 
                                 kernel_initializer='glorot_normal')(x)
    
    return keras.Model(inputs=inputs, outputs=outputs)

def compute_pde_residual(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
    """
    Compute PDE residual - to be implemented by subclasses
    
    Args:
        x: Spatial coordinates
        t: Time coordinates
        
    Returns:
        PDE residual
    """
    raise NotImplementedError("Subclasses must implement PDE residual")

def loss_function(self, x_data, t_data, u_data,
                 x_pde, t_pde,
                 x_bc, t_bc, u_bc,
                 x_ic, t_ic, u_ic):
    """
    Combined loss function: L = L_data + λ_pde*L_pde + λ_bc*L_bc + λ_ic*L_ic
    """
    # Data loss
    inputs_data = tf.concat([x_data, t_data], axis=1)
    u_pred_data = self.model(inputs_data)
    loss_data = tf.reduce_mean(tf.square(u_pred_data - u_data))
    
    # PDE loss
    with tf.GradientTape() as tape:
        tape.watch([x_pde, t_pde])
        residual = self.compute_pde_residual(x_pde, t_pde)
    loss_pde = tf.reduce_mean(tf.square(residual))
    
    # Boundary condition loss
    inputs_bc = tf.concat([x_bc, t_bc], axis=1)
    u_pred_bc = self.model(inputs_bc)
    loss_bc = tf.reduce_mean(tf.square(u_pred_bc - u_bc))
    
    # Initial condition loss
    inputs_ic = tf.concat([x_ic, t_ic], axis=1)
    u_pred_ic = self.model(inputs_ic)
    loss_ic = tf.reduce_mean(tf.square(u_pred_ic - u_ic))
    
    # Total loss
    total_loss = (loss_data + 
                 self.lambda_pde * loss_pde + 
                 self.lambda_bc * loss_bc + 
                 self.lambda_ic * loss_ic)
    
    return total_loss, loss_data, loss_pde, loss_bc, loss_ic

def train_step(self, x_data, t_data, u_data,
               x_pde, t_pde,
               x_bc, t_bc, u_bc,
               x_ic, t_ic, u_ic):
    """Single training step"""
    with tf.GradientTape() as tape:
        loss, l_data, l_pde, l_bc, l_ic = self.loss_function(
            x_data, t_data, u_data,
            x_pde, t_pde,
            x_bc, t_bc, u_bc,
            x_ic, t_ic, u_ic
        )
    
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    
    return loss, l_data, l_pde, l_bc, l_ic

def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Make predictions"""
    inputs = np.concatenate([x, t], axis=1)
    return self.model.predict(inputs)
class HeatEquationPINN(PhysicsInformedNN):
"""
PINN for 1D heat equation: ∂u/∂t = α∂²u/∂x²
Relevant for modeling temperature diffusion in atmosphere/ocean
"""

def __init__(self, alpha: float = 0.01, **kwargs):
    """
    Args:
        alpha: Thermal diffusivity coefficient
    """
    self.alpha = alpha
    super().__init__(**kwargs)

def compute_pde_residual(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
    """
    Compute heat equation residual: R = ∂u/∂t - α∂²u/∂x²
    """
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([x, t])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x, t])
            inputs = tf.concat([x, t], axis=1)
            u = self.model(inputs)
        
        u_x = tape1.gradient(u, x)
        u_t = tape1.gradient(u, t)
    
    u_xx = tape2.gradient(u_x, x)
    
    # Heat equation residual
    residual = u_t - self.alpha * u_xx
    
    return residual
class BurgersEquationPINN(PhysicsInformedNN):
"""
PINN for Burgers equation: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
Simplified model for fluid dynamics, relevant for atmospheric flows
"""

def __init__(self, nu: float = 0.01, **kwargs):
    """
    Args:
        nu: Kinematic viscosity
    """
    self.nu = nu
    super().__init__(**kwargs)

def compute_pde_residual(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
    """
    Compute Burgers equation residual: R = ∂u/∂t + u∂u/∂x - ν∂²u/∂x²
    """
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([x, t])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x, t])
            inputs = tf.concat([x, t], axis=1)
            u = self.model(inputs)
        
        u_x = tape1.gradient(u, x)
        u_t = tape1.gradient(u, t)
    
    u_xx = tape2.gradient(u_x, x)
    
    # Burgers equation residual
    residual = u_t + u * u_x - self.nu * u_xx
    
    return residual
def generate_heat_equation_data(nx: int = 100, nt: int = 100,
alpha: float = 0.01) -> Tuple:
"""
Generate synthetic data for heat equation
Args:
    nx: Number of spatial points
    nt: Number of temporal points
    alpha: Thermal diffusivity
    
Returns:
    x, t, u: Spatial coords, temporal coords, solution
"""
x = np.linspace(0, 1, nx)
t = np.linspace(0, 1, nt)
X, T = np.meshgrid(x, t)

# Analytical solution for simple initial condition
# u(x,t) = sin(πx) * exp(-π²αt)
U = np.sin(np.pi * X) * np.exp(-np.pi**2 * alpha * T)

x_flat = X.flatten()[:, None]
t_flat = T.flatten()[:, None]
u_flat = U.flatten()[:, None]

return x_flat, t_flat, u_flat
def train_heat_pinn_example():
"""
Example: Train PINN on heat equation
"""
print("Generating heat equation data...")
x_data, t_data, u_data = generate_heat_equation_data()
# Sample points for PDE evaluation
n_pde = 10000
x_pde = np.random.uniform(0, 1, (n_pde, 1))
t_pde = np.random.uniform(0, 1, (n_pde, 1))

# Boundary conditions: u(0,t) = u(1,t) = 0
n_bc = 200
x_bc0 = np.zeros((n_bc, 1))
x_bc1 = np.ones((n_bc, 1))
t_bc = np.random.uniform(0, 1, (n_bc, 1))
x_bc = np.vstack([x_bc0, x_bc1])
t_bc = np.vstack([t_bc, t_bc])
u_bc = np.zeros((2*n_bc, 1))

# Initial conditions: u(x,0) = sin(πx)
n_ic = 200
x_ic = np.random.uniform(0, 1, (n_ic, 1))
t_ic = np.zeros((n_ic, 1))
u_ic = np.sin(np.pi * x_ic)

# Convert to tensors
x_data_tf = tf.constant(x_data, dtype=tf.float32)
t_data_tf = tf.constant(t_data, dtype=tf.float32)
u_data_tf = tf.constant(u_data, dtype=tf.float32)

x_pde_tf = tf.constant(x_pde, dtype=tf.float32)
t_pde_tf = tf.constant(t_pde, dtype=tf.float32)

x_bc_tf = tf.constant(x_bc, dtype=tf.float32)
t_bc_tf = tf.constant(t_bc, dtype=tf.float32)
u_bc_tf = tf.constant(u_bc, dtype=tf.float32)

x_ic_tf = tf.constant(x_ic, dtype=tf.float32)
t_ic_tf = tf.constant(t_ic, dtype=tf.float32)
u_ic_tf = tf.constant(u_ic, dtype=tf.float32)

# Initialize PINN
print("Initializing PINN...")
layers = [2, 50, 50, 50, 1]
pinn = HeatEquationPINN(layers=layers, alpha=0.01, 
                       lambda_pde=1.0, lambda_bc=10.0, lambda_ic=10.0)

# Training loop
print("Training PINN...")
epochs = 1000
losses = []

for epoch in range(epochs):
    loss, l_data, l_pde, l_bc, l_ic = pinn.train_step(
        x_data_tf, t_data_tf, u_data_tf,
        x_pde_tf, t_pde_tf,
        x_bc_tf, t_bc_tf, u_bc_tf,
        x_ic_tf, t_ic_tf, u_ic_tf
    )
    
    losses.append(loss.numpy())
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.6f}, "
              f"L_data = {l_data:.6f}, L_pde = {l_pde:.6f}, "
              f"L_bc = {l_bc:.6f}, L_ic = {l_ic:.6f}")

# Visualize results
visualize_pinn_results(pinn, x_data, t_data, u_data, losses)

return pinn
def visualize_pinn_results(pinn, x_data, t_data, u_data, losses):
"""Visualize PINN training results"""
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# Plot 1: Training loss
axes[0, 0].plot(losses)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Total Loss')
axes[0, 0].set_title('Training Loss Curve')
axes[0, 0].set_yscale('log')
axes[0, 0].grid(True)

# Plot 2: True solution
u_true = u_data.reshape(100, 100)
im1 = axes[0, 1].contourf(u_true.T, levels=50, cmap='RdBu_r')
axes[0, 1].set_xlabel('Time step')
axes[0, 1].set_ylabel('Spatial point')
axes[0, 1].set_title('True Solution')
plt.colorbar(im1, ax=axes[0, 1])

# Plot 3: PINN prediction
u_pred = pinn.predict(x_data, t_data).reshape(100, 100)
im2 = axes[1, 0].contourf(u_pred.T, levels=50, cmap='RdBu_r')
axes[1, 0].set_xlabel('Time step')
axes[1, 0].set_ylabel('Spatial point')
axes[1, 0].set_title('PINN Prediction')
plt.colorbar(im2, ax=axes[1, 0])

# Plot 4: Error
error = np.abs(u_true - u_pred)
im3 = axes[1, 1].contourf(error.T, levels=50, cmap='Reds')
axes[1, 1].set_xlabel('Time step')
axes[1, 1].set_ylabel('Spatial point')
axes[1, 1].set_title(f'Absolute Error (Mean: {np.mean(error):.6f})')
plt.colorbar(im3, ax=axes[1, 1])

plt.tight_layout()
plt.savefig('pinn_heat_equation_results.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nResults:")
print(f"Mean Absolute Error: {np.mean(error):.6f}")
print(f"Max Absolute Error: {np.max(error):.6f}")
print(f"Relative L2 Error: {np.linalg.norm(u_true - u_pred) / np.linalg.norm(u_true):.6f}")
if name == "main":
print("="*60)
print("Physics-Informed Neural Network Example")
print("Chapter 1: Introduction to AI in Climate Science")
print("="*60)
# Train heat equation PINN
pinn = train_heat_pinn_example()

print("\nPINN training complete!")
print("Results saved to 'pinn_heat_equation_results.png'")
