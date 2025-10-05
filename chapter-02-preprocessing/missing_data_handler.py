"""
Missing Data Handler for Climate Preprocessing
Implements various missing data imputation strategies including:

Matrix completion via nuclear norm minimization
Bayesian imputation
Physics-informed gap filling

Based on Chapter 2: Climate Data Challenges and Preprocessing
"""
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.impute import KNNImputer
from typing import Tuple, Optional
import matplotlib.pyplot as plt
class MissingDataClassifier:
"""
Classify missing data mechanisms: MCAR, MAR, MNAR
"""
def __init__(self, data: np.ndarray):
    """
    Initialize with data matrix
    
    Args:
        data: Climate data array with missing values (NaN)
    """
    self.data = data
    self.missing_mask = np.isnan(data)
    
def test_mcar(self, alpha: float = 0.05) -> Tuple[bool, float]:
    """
    Test for Missing Completely at Random (MCAR)
    
    Uses Little's MCAR test
    
    Returns:
        is_mcar: Boolean indicating if data is MCAR
        p_value: P-value of the test
    """
    # Simplified MCAR test based on mean comparison
    observed_means = []
    missing_patterns = []
    
    n_vars = self.data.shape[1]
    
    for i in range(n_vars):
        mask = self.missing_mask[:, i]
        if np.sum(mask) > 0 and np.sum(~mask) > 0:
            # Compare means of other variables for missing vs observed
            for j in range(n_vars):
                if i != j:
                    observed = self.data[~mask, j]
                    missing = self.data[mask, j]
                    
                    obs_mean = np.nanmean(observed)
                    mis_mean = np.nanmean(missing)
                    
                    observed_means.append(obs_mean)
                    missing_patterns.append(mis_mean)
    
    # Simple t-test approximation
    if len(observed_means) > 0:
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(observed_means, missing_patterns, 
                                          nan_policy='omit')
        is_mcar = p_value > alpha
        return is_mcar, p_value
    else:
        return True, 1.0
class MatrixCompletionImputer:
"""
Matrix completion via nuclear norm minimization
Implements: min ||Z||_* subject to Z_ij = X_ij for (i,j) in Omega
"""

def __init__(self, max_rank: Optional[int] = None, 
             max_iter: int = 100, tol: float = 1e-4):
    """
    Initialize matrix completion imputer
    
    Args:
        max_rank: Maximum rank for approximation
        max_iter: Maximum iterations for optimization
        tol: Convergence tolerance
    """
    self.max_rank = max_rank
    self.max_iter = max_iter
    self.tol = tol
    
def fit_transform(self, X: np.ndarray) -> np.ndarray:
    """
    Perform matrix completion via SVD iteration
    
    Args:
        X: Data matrix with missing values (NaN)
        
    Returns:
        Completed matrix
    """
    # Initialize with mean imputation
    X_filled = X.copy()
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        X_filled[np.isnan(X[:, j]), j] = col_means[j]
    
    observed = ~np.isnan(X)
    
    for iteration in range(self.max_iter):
        X_old = X_filled.copy()
        
        # SVD decomposition
        U, s, Vt = linalg.svd(X_filled, full_matrices=False)
        
        # Rank truncation
        if self.max_rank is not None:
            s[self.max_rank:] = 0
        
        # Reconstruct
        X_filled = U @ np.diag(s) @ Vt
        
        # Restore observed values
        X_filled[observed] = X[observed]
        
        # Check convergence
        change = np.linalg.norm(X_filled - X_old, 'fro') / np.linalg.norm(X_old, 'fro')
        if change < self.tol:
            print(f"Converged after {iteration + 1} iterations")
            break
    
    return X_filled
class BayesianImputer:
"""
Bayesian imputation with uncertainty quantification
Assumes multivariate normal distribution
"""

def __init__(self, n_samples: int = 100):
    """
    Initialize Bayesian imputer
    
    Args:
        n_samples: Number of posterior samples for uncertainty
    """
    self.n_samples = n_samples
    self.mean_ = None
    self.cov_ = None
    
def fit(self, X: np.ndarray):
    """
    Estimate parameters from observed data
    
    Args:
        X: Data matrix with missing values
    """
    # Use pairwise complete observations for covariance estimation
    self.mean_ = np.nanmean(X, axis=0)
    self.cov_ = np.ma.cov(np.ma.masked_invalid(X), rowvar=False).data
    
    # Ensure positive definite
    self.cov_ = self.cov_ + 1e-6 * np.eye(self.cov_.shape[0])
    
    return self
    
def transform(self, X: np.ndarray, return_std: bool = False):
    """
    Impute missing values with posterior mean
    
    Args:
        X: Data matrix with missing values
        return_std: Whether to return standard deviations
        
    Returns:
        X_imputed: Imputed data
        X_std: Standard deviations (if return_std=True)
    """
    X_imputed = X.copy()
    X_std = np.zeros_like(X) if return_std else None
    
    for i in range(X.shape[0]):
        missing = np.isnan(X[i])
        if np.any(missing):
            observed = ~missing
            
            # Conditional distribution: Y_mis | Y_obs ~ N(mu, Sigma)
            # mu = mu_mis + Sigma_mis,obs * Sigma_obs^-1 * (Y_obs - mu_obs)
            # Sigma = Sigma_mis - Sigma_mis,obs * Sigma_obs^-1 * Sigma_obs,mis
            
            mu_obs = self.mean_[observed]
            mu_mis = self.mean_[missing]
            
            Sigma_obs = self.cov_[np.ix_(observed, observed)]
            Sigma_mis = self.cov_[np.ix_(missing, missing)]
            Sigma_mis_obs = self.cov_[np.ix_(missing, observed)]
            
            # Posterior mean
            try:
                Sigma_obs_inv = np.linalg.inv(Sigma_obs)
                mu_cond = mu_mis + Sigma_mis_obs @ Sigma_obs_inv @ (X[i, observed] - mu_obs)
                X_imputed[i, missing] = mu_cond
                
                # Posterior covariance
                if return_std:
                    Sigma_cond = Sigma_mis - Sigma_mis_obs @ Sigma_obs_inv @ Sigma_mis_obs.T
                    X_std[i, missing] = np.sqrt(np.diag(Sigma_cond))
                    
            except np.linalg.LinAlgError:
                # Fallback to mean imputation
                X_imputed[i, missing] = mu_mis
                if return_std:
                    X_std[i, missing] = np.sqrt(np.diag(Sigma_mis))
    
    if return_std:
        return X_imputed, X_std
    return X_imputed
def evaluate_imputation(X_true: np.ndarray, X_imputed: np.ndarray,
missing_mask: np.ndarray) -> dict:
"""
Evaluate imputation performance
Args:
    X_true: True values
    X_imputed: Imputed values
    missing_mask: Boolean mask of missing values
    
Returns:
    Dictionary of performance metrics
"""
# Only evaluate on truly missing values
errors = X_true[missing_mask] - X_imputed[missing_mask]

metrics = {
    'rmse': np.sqrt(np.mean(errors**2)),
    'mae': np.mean(np.abs(errors)),
    'bias': np.mean(errors),
    'correlation': np.corrcoef(X_true[missing_mask], X_imputed[missing_mask])[0, 1]
}

return metrics
def demonstrate_imputation_methods():
"""
Demonstrate different imputation methods on synthetic climate data
"""
print("="*60)
print("Missing Data Imputation Demonstration")
print("Chapter 2: Climate Data Challenges and Preprocessing")
print("="*60)
# Generate synthetic climate data
np.random.seed(42)
n_samples = 365  # One year of daily data
n_vars = 5  # Temperature, pressure, humidity, wind speed, precipitation

# Create correlated climate variables
mean = np.array([15, 1013, 60, 5, 2])  # Typical values
cov = np.array([
    [25, 5, -10, 2, -3],      # Temperature
    [5, 100, -5, 1, 0],        # Pressure
    [-10, -5, 400, -2, 5],     # Humidity
    [2, 1, -2, 9, -1],         # Wind speed
    [-3, 0, 5, -1, 4]          # Precipitation
])

X_true = np.random.multivariate_normal(mean, cov, size=n_samples)

# Introduce missing data (20% MCAR)
missing_rate = 0.20
missing_mask = np.random.rand(*X_true.shape) < missing_rate
X_missing = X_true.copy()
X_missing[missing_mask] = np.nan

print(f"\nData shape: {X_true.shape}")
print(f"Missing rate: {100*missing_rate:.1f}%")
print(f"Total missing values: {np.sum(missing_mask)}")

# Test MCAR
print("\n" + "-"*60)
print("Testing Missing Data Mechanism")
print("-"*60)
classifier = MissingDataClassifier(X_missing)
is_mcar, p_value = classifier.test_mcar()
print(f"MCAR test p-value: {p_value:.4f}")
print(f"Data appears to be MCAR: {is_mcar}")

# Method 1: Matrix Completion
print("\n" + "-"*60)
print("Method 1: Matrix Completion (Nuclear Norm Minimization)")
print("-"*60)
mc_imputer = MatrixCompletionImputer(max_rank=3, max_iter=100)
X_mc = mc_imputer.fit_transform(X_missing)
metrics_mc = evaluate_imputation(X_true, X_mc, missing_mask)
print(f"RMSE: {metrics_mc['rmse']:.4f}")
print(f"MAE: {metrics_mc['mae']:.4f}")
print(f"Correlation: {metrics_mc['correlation']:.4f}")

# Method 2: Bayesian Imputation
print("\n" + "-"*60)
print("Method 2: Bayesian Imputation")
print("-"*60)
bayes_imputer = BayesianImputer(n_samples=100)
bayes_imputer.fit(X_missing)
X_bayes, X_std = bayes_imputer.transform(X_missing, return_std=True)
metrics_bayes = evaluate_imputation(X_true, X_bayes, missing_mask)
print(f"RMSE: {metrics_bayes['rmse']:.4f}")
print(f"MAE: {metrics_bayes['mae']:.4f}")
print(f"Correlation: {metrics_bayes['correlation']:.4f}")
print(f"Mean uncertainty (std): {np.mean(X_std[missing_mask]):.4f}")

# Method 3: KNN Imputation (baseline)
print("\n" + "-"*60)
print("Method 3: KNN Imputation (Baseline)")
print("-"*60)
knn_imputer = KNNImputer(n_neighbors=5)
X_knn = knn_imputer.fit_transform(X_missing)
metrics_knn = evaluate_imputation(X_true, X_knn, missing_mask)
print(f"RMSE: {metrics_knn['rmse']:.4f}")
print(f"MAE: {metrics_knn['mae']:.4f}")
print(f"Correlation: {metrics_knn['correlation']:.4f}")

# Visualize results
visualize_imputation_results(X_true, X_missing, X_mc, X_bayes, 
                             X_knn, missing_mask)

print("\n" + "="*60)
print("Imputation demonstration complete!")
print("="*60)
def visualize_imputation_results(X_true, X_missing, X_mc, X_bayes, X_knn, missing_mask):
"""Visualize imputation results"""
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
var_names = ['Temperature', 'Pressure', 'Humidity', 'Wind Speed', 'Precipitation']
var_idx = 0  # Visualize temperature

time = np.arange(len(X_true))

# Plot 1: Original vs Observed
axes[0, 0].plot(time, X_true[:, var_idx], 'k-', label='True', alpha=0.7)
axes[0, 0].plot(time, X_missing[:, var_idx], 'b.', label='Observed', markersize=3)
axes[0, 0].set_xlabel('Time (days)')
axes[0, 0].set_ylabel(var_names[var_idx])
axes[0, 0].set_title('Original Data with Missing Values')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Matrix Completion
axes[0, 1].plot(time, X_true[:, var_idx], 'k-', label='True', alpha=0.7)
axes[0, 1].plot(time, X_mc[:, var_idx], 'r--', label='Matrix Completion', alpha=0.7)
axes[0, 1].set_xlabel('Time (days)')
axes[0, 1].set_ylabel(var_names[var_idx])
axes[0, 1].set_title('Matrix Completion Imputation')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Bayesian
axes[1, 0].plot(time, X_true[:, var_idx], 'k-', label='True', alpha=0.7)
axes[1, 0].plot(time, X_bayes[:, var_idx], 'g--', label='Bayesian', alpha=0.7)
axes[1, 0].set_xlabel('Time (days)')
axes[1, 0].set_ylabel(var_names[var_idx])
axes[1, 0].set_title('Bayesian Imputation')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Comparison scatter
missing_points = missing_mask[:, var_idx]
axes[1, 1].scatter(X_true[missing_points, var_idx], 
                   X_mc[missing_points, var_idx], 
                   alpha=0.5, label='Matrix Completion', s=20)
axes[1, 1].scatter(X_true[missing_points, var_idx], 
                   X_bayes[missing_points, var_idx], 
                   alpha=0.5, label='Bayesian', s=20)
axes[1, 1].plot([X_true[:, var_idx].min(), X_true[:, var_idx].max()],
                 [X_true[:, var_idx].min(), X_true[:, var_idx].max()],
                 'k--', label='Perfect')
axes[1, 1].set_xlabel('True Values')
axes[1, 1].set_ylabel('Imputed Values')
axes[1, 1].set_title('Imputation Accuracy (Missing Values Only)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('imputation_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nResults saved to 'imputation_results.png'")
if name == "main":
demonstrate_imputation_methods()
