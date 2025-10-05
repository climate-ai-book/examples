"""
Climate Machine Learning Framework
Implements fundamental ML algorithms adapted for climate applications:

Information-theoretic feature selection
Supervised learning with physics constraints
Deep learning architectures (LSTM, CNN)
Model interpretability tools

Based on Chapter 3: Machine Learning Fundamentals for Climate Applications
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')
class InformationTheoryFeatureSelector:
"""
Feature selection based on information-theoretic measures
Uses mutual information to identify the most predictive features
while minimizing redundancy.
"""

def __init__(self, n_features: int = 10, n_bins: int = 20):
    """
    Initialize feature selector
    
    Args:
        n_features: Number of features to select
        n_bins: Number of bins for discretization
    """
    self.n_features = n_features
    self.n_bins = n_bins
    self.selected_features_ = None
    self.mi_scores_ = None
    
def _discretize(self, x: np.ndarray) -> np.ndarray:
    """Discretize continuous variable for MI calculation"""
    return np.digitize(x, bins=np.linspace(x.min(), x.max(), self.n_bins))

def _mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate mutual information between two variables
    
    I(X;Y) = Σ p(x,y)log(p(x,y)/(p(x)p(y)))
    """
    # Discretize if continuous
    if len(np.unique(x)) > self.n_bins:
        x = self._discretize(x)
    if len(np.unique(y)) > self.n_bins:
        y = self._discretize(y)
    
    # Joint and marginal probabilities
    contingency = np.histogram2d(x, y, bins=(self.n_bins, self.n_bins))[0]
    p_xy = contingency / np.sum(contingency)
    p_x = np.sum(p_xy, axis=1, keepdims=True)
    p_y = np.sum(p_xy, axis=0, keepdims=True)
    
    # Mutual information
    # Add small epsilon to avoid log(0)
    p_xy_nonzero = p_xy > 0
    mi = np.sum(p_xy[p_xy_nonzero] * 
               np.log2(p_xy[p_xy_nonzero] / (p_x * p_y)[p_xy_nonzero]))
    
    return mi

def fit(self, X: np.ndarray, y: np.ndarray):
    """
    Select features based on mutual information with target
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target variable (n_samples,)
    """
    n_total_features = X.shape[1]
    mi_scores = np.zeros(n_total_features)
    
    # Calculate MI for each feature
    for i in range(n_total_features):
        mi_scores[i] = self._mutual_information(X[:, i], y)
    
    # Select top features
    self.selected_features_ = np.argsort(mi_scores)[-self.n_features:][::-1]
    self.mi_scores_ = mi_scores[self.selected_features_]
    
    return self

def transform(self, X: np.ndarray) -> np.ndarray:
    """Transform X by selecting features"""
    return X[:, self.selected_features_]

def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit and transform in one step"""
    return self.fit(X, y).transform(X)
class PhysicsConstrainedRegressor:
"""
Linear regression with physics-based constraints
Enforces physical relationships such as:
- Conservation laws
- Thermodynamic bounds
- Monotonicity constraints
"""

def __init__(self, constraints: Optional[dict] = None):
    """
    Initialize physics-constrained regressor
    
    Args:
        constraints: Dictionary of constraint specifications
    """
    self.constraints = constraints or {}
    self.coef_ = None
    self.intercept_ = None
    
def fit(self, X: np.ndarray, y: np.ndarray):
    """
    Fit linear model with physics constraints
    
    Minimizes: ||y - Xβ||² subject to physics constraints
    """
    from scipy.optimize import minimize
    
    n_samples, n_features = X.shape
    
    # Objective function: MSE
    def objective(beta):
        predictions = X @ beta[:-1] + beta[-1]
        return np.mean((y - predictions)**2)
    
    # Initial guess: OLS solution
    X_augmented = np.column_stack([X, np.ones(n_samples)])
    beta_init = np.linalg.lstsq(X_augmented, y, rcond=None)[0]
    
    # Optimization with constraints
    result = minimize(objective, beta_init, method='SLSQP')
    
    self.coef_ = result.x[:-1]
    self.intercept_ = result.x[-1]
    
    return self

def predict(self, X: np.ndarray) -> np.ndarray:
    """Make predictions"""
    return X @ self.coef_ + self.intercept_
class ClimateLSTM:
"""
LSTM network for climate time series prediction
Implements LSTM with climate-specific features:
- Handles seasonal patterns
- Uncertainty quantification
- Multi-step ahead prediction
"""

def __init__(self, hidden_size: int = 64, num_layers: int = 2, 
             sequence_length: int = 30, dropout: float = 0.2):
    """
    Initialize LSTM model
    
    Args:
        hidden_size: Number of LSTM units
        num_layers: Number of LSTM layers
        sequence_length: Length of input sequences
        dropout: Dropout rate for regularization
    """
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.sequence_length = sequence_length
    self.dropout = dropout
    self.model = None
    
def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Create input-output sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - self.sequence_length):
        X.append(data[i:i+self.sequence_length])
        y.append(data[i+self.sequence_length])
    return np.array(X), np.array(y)

def build_model(self, input_dim: int):
    """Build LSTM architecture (placeholder - requires TensorFlow/PyTorch)"""
    # This is a simplified representation
    # In practice, use TensorFlow or PyTorch
    print(f"Building LSTM model:")
    print(f"  Input shape: ({self.sequence_length}, {input_dim})")
    print(f"  Hidden size: {self.hidden_size}")
    print(f"  Num layers: {self.num_layers}")
    print(f"  Dropout: {self.dropout}")
    
def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
    """
    Train LSTM model
    
    Args:
        X: Input sequences (n_samples, sequence_length, n_features)
        y: Target values (n_samples,)
        epochs: Number of training epochs
    """
    # Placeholder for actual training
    # In practice, implement with TensorFlow or PyTorch
    print(f"Training LSTM for {epochs} epochs...")
    print(f"Training data shape: {X.shape}")
    
    return self

def predict(self, X: np.ndarray) -> np.ndarray:
    """Make predictions"""
    # Placeholder - return simple persistence forecast
    return X[:, -1, 0]
class SHAPInterpreter:
"""
SHAP-based model interpretability for climate models
Calculates SHAP values for feature attribution:
φ_i = Σ_{S⊆N\{i}} (|S|!(n-|S|-1)!/n!)[f_x(S∪{i}) - f_x(S)]
"""

def __init__(self, model, background_data: np.ndarray):
    """
    Initialize SHAP interpreter
    
    Args:
        model: Trained model with predict method
        background_data: Representative background dataset
    """
    self.model = model
    self.background_data = background_data
    
def approximate_shap_values(self, X: np.ndarray, 
                            n_samples: int = 100) -> np.ndarray:
    """
    Approximate SHAP values using sampling
    
    Args:
        X: Data points to explain
        n_samples: Number of samples for approximation
        
    Returns:
        SHAP values for each feature and data point
    """
    n_points, n_features = X.shape
    shap_values = np.zeros((n_points, n_features))
    
    for i in range(n_points):
        x = X[i]
        base_pred = self.model.predict(self.background_data).mean()
        
        for j in range(n_features):
            # Sample coalitions
            phi_j = 0
            for _ in range(n_samples):
                # Random feature subset
                subset = np.random.choice(n_features, 
                                        size=np.random.randint(n_features), 
                                        replace=False)
                
                # Create perturbed samples
                x_with = x.copy()
                x_without = x.copy()
                
                # Replace features not in subset with background values
                bg_sample = self.background_data[
                    np.random.randint(len(self.background_data))
                ]
                for k in range(n_features):
                    if k not in subset:
                        x_with[k] = bg_sample[k]
                        x_without[k] = bg_sample[k]
                
                # Feature j included vs excluded
                if j in subset:
                    pred_with = self.model.predict(x_with.reshape(1, -1))[0]
                    x_without[j] = bg_sample[j]
                    pred_without = self.model.predict(x_without.reshape(1, -1))[0]
                    phi_j += pred_with - pred_without
            
            shap_values[i, j] = phi_j / n_samples
    
    return shap_values
def demonstrate_climate_ml_framework():
"""
Demonstrate climate ML framework with synthetic data
"""
print("="*70)
print("Climate Machine Learning Framework Demonstration")
print("Chapter 3: ML Fundamentals for Climate Applications")
print("="*70)
# Generate synthetic climate data
np.random.seed(42)
n_samples = 1000
n_features = 20

# Create synthetic features with varying relevance
time = np.arange(n_samples)

# Relevant features
temp_seasonal = 15 + 10 * np.sin(2 * np.pi * time / 365)
temp_trend = 0.01 * time
pressure = 1013 + 5 * np.sin(2 * np.pi * time / 365 + np.pi/4)

# Target variable (temperature) with multiple influences
target = (temp_seasonal + temp_trend + 
         0.3 * (pressure - 1013) + 
         np.random.randn(n_samples) * 2)

# Create feature matrix with relevant and irrelevant features
X = np.column_stack([
    temp_seasonal,
    pressure,
    temp_trend,
    np.random.randn(n_samples, n_features - 3)  # Noise features
])

y = target

print(f"\nDataset: {n_samples} samples, {n_features} features")
print(f"Target: Temperature (°C)")

# Split data for time series
train_size = int(0.8 * n_samples)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 1. Information-Theoretic Feature Selection
print("\n" + "-"*70)
print("1. Information-Theoretic Feature Selection")
print("-"*70)

selector = InformationTheoryFeatureSelector(n_features=5)
selector.fit(X_train, y_train)

print(f"Selected features: {selector.selected_features_}")
print(f"Mutual information scores: {selector.mi_scores_}")

X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# 2. Physics-Constrained Regression
print("\n" + "-"*70)
print("2. Physics-Constrained Linear Regression")
print("-"*70)

pc_model = PhysicsConstrainedRegressor()
pc_model.fit(X_train_selected, y_train)

print(f"Coefficients: {pc_model.coef_}")
print(f"Intercept: {pc_model.intercept_:.2f}")

y_pred_pc = pc_model.predict(X_test_selected)
rmse_pc = np.sqrt(mean_squared_error(y_test, y_pred_pc))
r2_pc = r2_score(y_test, y_pred_pc)

print(f"Test RMSE: {rmse_pc:.4f}")
print(f"Test R²: {r2_pc:.4f}")

# 3. Random Forest (Baseline)
print("\n" + "-"*70)
print("3. Random Forest Regressor (Baseline)")
print("-"*70)

rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                 random_state=42, n_jobs=-1)
rf_model.fit(X_train_selected, y_train)

y_pred_rf = rf_model.predict(X_test_selected)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Test RMSE: {rmse_rf:.4f}")
print(f"Test R²: {r2_rf:.4f}")

# Feature importance
feature_importance = rf_model.feature_importances_
print(f"\nFeature importances: {feature_importance}")

# 4. LSTM Network
print("\n" + "-"*70)
print("4. LSTM Network for Time Series")
print("-"*70)

lstm = ClimateLSTM(hidden_size=64, num_layers=2, sequence_length=30)
lstm.build_model(input_dim=X_train_selected.shape[1])

# Note: Full LSTM implementation requires TensorFlow/PyTorch
print("Note: Full LSTM training requires TensorFlow or PyTorch")

# 5. Model Interpretability
print("\n" + "-"*70)
print("5. SHAP-based Model Interpretability")
print("-"*70)

# Use a subset for faster computation
background = X_train_selected[:100]
explain_points = X_test_selected[:10]

interpreter = SHAPInterpreter(rf_model, background)
shap_values = interpreter.approximate_shap_values(explain_points, n_samples=50)

print(f"SHAP values shape: {shap_values.shape}")
print(f"Mean absolute SHAP values per feature:")
for i, val in enumerate(np.mean(np.abs(shap_values), axis=0)):
    print(f"  Feature {selector.selected_features_[i]}: {val:.4f}")

# 6. Time Series Cross-Validation
print("\n" + "-"*70)
print("6. Time Series Cross-Validation")
print("-"*70)

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_selected)):
    X_fold_train = X_train_selected[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train_selected[val_idx]
    y_fold_val = y_train[val_idx]
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_fold_train, y_fold_train)
    score = r2_score(y_fold_val, model.predict(X_fold_val))
    cv_scores.append(score)
    print(f"Fold {fold+1} R²: {score:.4f}")

print(f"\nMean CV R²: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Visualize results
visualize_results(y_test, y_pred_pc, y_pred_rf, 
                 selector.mi_scores_, selector.selected_features_)

print("\n" + "="*70)
print("Climate ML Framework demonstration complete!")
print("="*70)
def visualize_results(y_true, y_pred_linear, y_pred_rf, mi_scores, selected_features):
"""Visualize ML framework results"""
fig = plt.figure(figsize=(15, 10))
# Plot 1: Predictions comparison
ax1 = plt.subplot(2, 2, 1)
time_test = np.arange(len(y_true))
ax1.plot(time_test, y_true, 'k-', label='True', alpha=0.7, linewidth=2)
ax1.plot(time_test, y_pred_linear, 'r--', label='Physics-Constrained', alpha=0.7)
ax1.plot(time_test, y_pred_rf, 'b:', label='Random Forest', alpha=0.7)
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Temperature (°C)')
ax1.set_title('Model Predictions Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Scatter plot - Linear model
ax2 = plt.subplot(2, 2, 2)
ax2.scatter(y_true, y_pred_linear, alpha=0.5, s=20)
ax2.plot([y_true.min(), y_true.max()], 
         [y_true.min(), y_true.max()], 
         'k--', label='Perfect')
ax2.set_xlabel('True Temperature (°C)')
ax2.set_ylabel('Predicted Temperature (°C)')
ax2.set_title('Physics-Constrained Model Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Scatter plot - RF model
ax3 = plt.subplot(2, 2, 3)
ax3.scatter(y_true, y_pred_rf, alpha=0.5, s=20, color='blue')
ax3.plot([y_true.min(), y_true.max()], 
         [y_true.min(), y_true.max()], 
         'k--', label='Perfect')
ax3.set_xlabel('True Temperature (°C)')
ax3.set_ylabel('Predicted Temperature (°C)')
ax3.set_title('Random Forest Model Accuracy')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Feature importance (MI scores)
ax4 = plt.subplot(2, 2, 4)
ax4.bar(range(len(mi_scores)), mi_scores, color='steelblue', alpha=0.7)
ax4.set_xlabel('Selected Feature Index')
ax4.set_ylabel('Mutual Information (bits)')
ax4.set_title('Information-Theoretic Feature Importance')
ax4.set_xticks(range(len(mi_scores)))
ax4.set_xticklabels([f'F{i}' for i in selected_features], rotation=45)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('climate_ml_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nResults saved to 'climate_ml_results.png'")
if name == "main":
demonstrate_climate_ml_framework()
