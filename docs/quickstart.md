# Quick Start Guide
Get up and running with AI in Climate Science in minutes.
Your First Example
Let's run a simple climate data analysis using information theory concepts from Chapter 1.
1. Import Libraries
pythonimport numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Generate synthetic temperature data
np.random.seed(42)
temperature = 15 + 10 * np.sin(np.linspace(0, 4*np.pi, 365)) + np.random.randn(365) * 2
2. Calculate Entropy
pythondef calculate_entropy(data, bins=50):
    """Calculate entropy of data distribution"""
    hist, _ = np.histogram(data, bins=bins, density=True)
    hist = hist + 1e-10  # Avoid log(0)
    dx = (data.max() - data.min()) / bins
    return -np.sum(hist * np.log2(hist + 1e-10) * dx)

temp_entropy = calculate_entropy(temperature)
print(f"Temperature Entropy: {temp_entropy:.4f} bits")
3. Visualize
pythonplt.figure(figsize=(12, 4))

# Time series
plt.subplot(1, 2, 1)
plt.plot(temperature)
plt.xlabel('Day of Year')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Time Series')
plt.grid(True)

# Distribution
plt.subplot(1, 2, 2)
plt.hist(temperature, bins=30, density=True, alpha=0.7)
plt.xlabel('Temperature (°C)')
plt.ylabel('Probability Density')
plt.title(f'Temperature Distribution (H={temp_entropy:.2f} bits)')
plt.grid(True)

plt.tight_layout()
plt.show()
Output:
Temperature Entropy: 4.2341 bits
Running Chapter Examples
Each chapter has organized examples:
bash# Navigate to chapter
cd chapter-01-introduction

# Explore notebooks
jupyter lab notebooks/01_information_theory_climate.ipynb

# Or run scripts
python scripts/pinn_implementations.py
Common Workflows
Loading Climate Data (NetCDF)
pythonimport xarray as xr

# Load ERA5 temperature data
ds = xr.open_dataset('data/temperature.nc')
temp = ds['t2m']  # 2-meter temperature

# Basic operations
mean_temp = temp.mean(dim='time')
seasonal_cycle = temp.groupby('time.month').mean()
Building a Simple Neural Network
pythonimport tensorflow as tf
from tensorflow import keras

# Simple feedforward network
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, validation_split=0.2)
Physics-Informed Neural Network
python# Define PDE loss (example: heat equation)
def pde_loss(x, t, model):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, t])
        u = model(tf.concat([x, t], axis=1))
        u_x = tape.gradient(u, x)
        u_t = tape.gradient(u, t)
    u_xx = tape.gradient(u_x, x)
    
    # Heat equation: du/dt = alpha * d2u/dx2
    residual = u_t - 0.01 * u_xx
    return tf.reduce_mean(tf.square(residual))
Interactive Notebooks
Try examples online without installation:
Show Image
Exploring the Repository
examples/
├── chapter-01-introduction/
│   ├── notebooks/          # Jupyter notebooks
│   ├── scripts/            # Python scripts
│   └── data/               # Sample data
├── chapter-02-preprocessing/
├── chapter-03-ml-fundamentals/
└── ...
Best Practices

Always activate the environment before running code
Start with notebooks for interactive learning
Check data paths - adjust relative paths as needed
Monitor memory usage with large climate datasets
Use GPU for deep learning examples when available

Common Patterns
Pattern 1: Data Loading
pythonimport xarray as xr

# Load multiple files
ds = xr.open_mfdataset('data/*.nc', combine='by_coords')
Pattern 2: Train-Test Split
pythonfrom sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
Pattern 3: Model Evaluation
pythonfrom sklearn.metrics import mean_squared_error, r2_score

predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

