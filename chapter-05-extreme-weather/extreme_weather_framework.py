"""
Extreme Weather Event Prediction Framework
Implements specialized ML methods for atmospheric hazard forecasting:

Extreme value theory (GEV/GPD) fitting
Focal loss for imbalanced learning
Enhanced ensemble methods for rare events
Specialized evaluation metrics for extremes

Based on Chapter 5: Extreme Weather Event Prediction
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')
class ExtremeValueTheory:
"""
Extreme Value Theory implementation for climate extremes
Provides GEV and GPD fitting, return level calculation,
and tail behavior analysis
"""

def __init__(self):
    self.gev_params = None
    self.gpd_params = None
    
def fit_gev(self, data: np.ndarray, block_size: int = 365) -> Dict[str, float]:
    """
    Fit Generalized Extreme Value distribution to block maxima
    
    F(x;μ,σ,ξ) = exp{-[1 + ξ(x-μ)/σ]^{-1/ξ}}
    
    Args:
        data: Time series data
        block_size: Size of blocks for maxima extraction
        
    Returns:
        Dictionary with location, scale, and shape parameters
    """
    # Extract block maxima
    n_blocks = len(data) // block_size
    block_maxima = []
    
    for i in range(n_blocks):
        block = data[i*block_size:(i+1)*block_size]
        if len(block) > 0:
            block_maxima.append(np.max(block))
    
    block_maxima = np.array(block_maxima)
    
    # Fit GEV distribution using scipy
    shape, loc, scale = stats.genextreme.fit(block_maxima)
    
    self.gev_params = {
        'location': loc,
        'scale': scale,
        'shape': -shape  # Note: scipy uses negative convention
    }
    
    return self.gev_params

def fit_gpd(self, data: np.ndarray, threshold: Optional[float] = None,
            threshold_quantile: float = 0.95) -> Dict[str, float]:
    """
    Fit Generalized Pareto Distribution to threshold exceedances
    
    F(x;σ,ξ) = 1 - (1 + ξx/σ)^{-1/ξ}
    
    Args:
        data: Time series data
        threshold: Threshold value (if None, use quantile)
        threshold_quantile: Quantile for threshold selection
        
    Returns:
        Dictionary with threshold, scale, and shape parameters
    """
    if threshold is None:
        threshold = np.quantile(data, threshold_quantile)
    
    # Extract exceedances
    exceedances = data[data > threshold] - threshold
    
    if len(exceedances) == 0:
        raise ValueError("No exceedances above threshold")
    
    # Fit GPD
    shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
    
    self.gpd_params = {
        'threshold': threshold,
        'scale': scale,
        'shape': shape
    }
    
    return self.gpd_params

def calculate_return_level(self, return_period: float, 
                           method: str = 'gev') -> float:
    """
    Calculate return level for given return period
    
    For GEV: x_T = μ + (σ/ξ)[(-ln(1-1/T))^{-ξ} - 1]
    
    Args:
        return_period: Return period in years
        method: 'gev' or 'gpd'
        
    Returns:
        Return level value
    """
    if method == 'gev':
        if self.gev_params is None:
            raise ValueError("GEV parameters not fitted")
        
        mu = self.gev_params['location']
        sigma = self.gev_params['scale']
        xi = self.gev_params['shape']
        
        if abs(xi) < 1e-6:  # Gumbel case
            return_level = mu - sigma * np.log(-np.log(1 - 1/return_period))
        else:
            return_level = mu + (sigma/xi) * ((-np.log(1 - 1/return_period))**(-xi) - 1)
            
    elif method == 'gpd':
        if self.gpd_params is None:
            raise ValueError("GPD parameters not fitted")
        
        u = self.gpd_params['threshold']
        sigma = self.gpd_params['scale']
        xi = self.gpd_params['shape']
        
        # Return level for GPD
        return_level = u + (sigma/xi) * ((return_period)**xi - 1)
    else:
        raise ValueError("Method must be 'gev' or 'gpd'")
    
    return return_level

def diagnostic_plots(self, data: np.ndarray, block_size: int = 365):
    """Generate diagnostic plots for GEV fit"""
    # Fit if not already done
    if self.gev_params is None:
        self.fit_gev(data, block_size)
    
    # Extract block maxima
    n_blocks = len(data) // block_size
    block_maxima = [np.max(data[i*block_size:(i+1)*block_size]) 
                   for i in range(n_blocks)]
    block_maxima = np.array(block_maxima)
    
    # Theoretical GEV quantiles
    shape_param = -self.gev_params['shape']  # scipy convention
    theoretical = stats.genextreme.rvs(
        shape_param,
        loc=self.gev_params['location'],
        scale=self.gev_params['scale'],
        size=len(block_maxima)
    )
    
    return block_maxima, np.sort(theoretical)
class FocalLossTrainer:
"""
Focal Loss implementation for imbalanced extreme event prediction
L_focal(p_t) = -α_t(1-p_t)^γ log(p_t)
"""

def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
    """
    Initialize Focal Loss
    
    Args:
        alpha: Weighting factor for class balance
        gamma: Focusing parameter for hard examples
    """
    self.alpha = alpha
    self.gamma = gamma
    
def focal_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute focal loss
    
    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted probabilities
        
    Returns:
        Focal loss value
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    
    # Compute focal loss
    p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
    alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
    
    loss = -alpha_t * (1 - p_t)**self.gamma * np.log(p_t)
    
    return np.mean(loss)

def focal_loss_gradient(self, y_true: np.ndarray, 
                       y_pred: np.ndarray) -> np.ndarray:
    """
    Compute gradient of focal loss w.r.t. predictions
    
    Returns:
        Gradient array
    """
    # Clip predictions
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    
    # Compute gradient
    p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
    alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
    
    # Gradient computation
    grad = alpha_t * (1 - p_t)**(self.gamma - 1) * (
        self.gamma * p_t * np.log(p_t) + p_t - 1
    )
    
    grad = np.where(y_true == 1, -grad, grad)
    
    return grad
class RareEventEnsemble:
"""
Enhanced ensemble methods for rare event prediction
Implements importance sampling and enhanced perturbations
"""

def __init__(self, n_members: int = 50):
    """
    Initialize rare event ensemble
    
    Args:
        n_members: Number of ensemble members
    """
    self.n_members = n_members
    
def generate_enhanced_perturbations(self, 
                                   initial_state: np.ndarray,
                                   extreme_indicator: float,
                                   base_std: float = 1.0,
                                   alpha_extreme: float = 2.0,
                                   threshold: float = 0.5) -> np.ndarray:
    """
    Generate ensemble with enhanced perturbations for extreme conditions
    
    σ_pert^extreme = σ_pert^standard · (1 + α·max(0, I-threshold))
    
    Args:
        initial_state: Base state vector
        extreme_indicator: Extreme weather index
        base_std: Base perturbation standard deviation
        alpha_extreme: Amplification factor for extremes
        threshold: Detection threshold
        
    Returns:
        Array of perturbed initial states
    """
    # Calculate enhanced perturbation magnitude
    enhancement = 1 + alpha_extreme * max(0, extreme_indicator - threshold)
    perturb_std = base_std * enhancement
    
    # Generate ensemble
    ensemble = np.zeros((self.n_members,) + initial_state.shape)
    
    for i in range(self.n_members):
        perturbation = np.random.randn(*initial_state.shape) * perturb_std
        ensemble[i] = initial_state + perturbation
    
    return ensemble

def importance_sampling_estimate(self, 
                                 samples: np.ndarray,
                                 target_density: callable,
                                 proposal_density: callable,
                                 extreme_region: callable) -> float:
    """
    Estimate extreme event probability using importance sampling
    
    P̂ = (1/N)Σ I(X_i∈A) p(X_i)/q(X_i)
    
    Args:
        samples: Sample array
        target_density: Target probability density function
        proposal_density: Proposal density function
        extreme_region: Function indicating extreme event region
        
    Returns:
        Estimated probability
    """
    n_samples = len(samples)
    
    # Compute importance weights
    weights = target_density(samples) / (proposal_density(samples) + 1e-10)
    
    # Indicator for extreme region
    indicators = extreme_region(samples)
    
    # Importance sampling estimate
    prob_estimate = np.mean(indicators * weights)
    
    return prob_estimate
class ExtremeEventVerification:
"""
Specialized verification metrics for extreme weather prediction
"""
@staticmethod
def compute_roc_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                       thresholds: Optional[np.ndarray] = None) -> Dict:
    """
    Compute ROC curve metrics
    
    Returns:
        Dictionary with TPR, FPR, and thresholds
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)
    
    tpr_list = []
    fpr_list = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    # Calculate AUC
    auc = np.trapz(tpr_list, fpr_list)
    
    return {
        'tpr': np.array(tpr_list),
        'fpr': np.array(fpr_list),
        'thresholds': thresholds,
        'auc': abs(auc)  # Take absolute value
    }

@staticmethod
def compute_precision_recall(y_true: np.ndarray, y_pred_proba: np.ndarray,
                             thresholds: Optional[np.ndarray] = None) -> Dict:
    """
    Compute precision-recall curve
    
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)
    
    precision_list = []
    recall_list = []
    f1_list = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
    # Calculate AUC-PR
    auc_pr = np.trapz(precision_list, recall_list)
    
    return {
        'precision': np.array(precision_list),
        'recall': np.array(recall_list),
        'f1': np.array(f1_list),
        'thresholds': thresholds,
        'auc_pr': abs(auc_pr)
    }

@staticmethod
def critical_success_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Critical Success Index (CSI)
    
    CSI = TP/(TP + FP + FN)
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    return csi
def demonstrate_extreme_weather_framework():
"""
Demonstrate extreme weather prediction framework
"""
print("="*70)
print("Extreme Weather Event Prediction Framework")
print("Chapter 5: Extreme Weather Event Prediction")
print("="*70)
# Generate synthetic extreme event data
np.random.seed(42)
n_samples = 10000

# Base climate variable (e.g., temperature)
base_data = np.random.gamma(shape=2, scale=15, size=n_samples)

# Add extreme events (5% of data)
n_extremes = int(0.05 * n_samples)
extreme_indices = np.random.choice(n_samples, n_extremes, replace=False)
base_data[extreme_indices] += np.random.gamma(shape=3, scale=20, size=n_extremes)

print(f"\nSynthetic Data: {n_samples} samples")
print(f"Extreme events: {n_extremes} ({100*n_extremes/n_samples:.1f}%)")
print(f"Data range: [{base_data.min():.2f}, {base_data.max():.2f}]")

# 1. Extreme Value Theory Analysis
print("\n" + "-"*70)
print("1. Extreme Value Theory (GEV and GPD)")
print("-"*70)

evt = ExtremeValueTheory()

# Fit GEV to block maxima
gev_params = evt.fit_gev(base_data, block_size=365)
print(f"\nGEV Parameters:")
print(f"  Location (μ): {gev_params['location']:.2f}")
print(f"  Scale (σ): {gev_params['scale']:.2f}")
print(f"  Shape (ξ): {gev_params['shape']:.4f}")

if gev_params['shape'] > 0:
    print(f"  Distribution type: Fréchet (heavy-tailed)")
elif abs(gev_params['shape']) < 0.01:
    print(f"  Distribution type: Gumbel (exponential tails)")
else:
    print(f"  Distribution type: Weibull (bounded)")

# Fit GPD to threshold exceedances
gpd_params = evt.fit_gpd(base_data, threshold_quantile=0.95)
print(f"\nGPD Parameters:")
print(f"  Threshold (u): {gpd_params['threshold']:.2f}")
print(f"  Scale (σ): {gpd_params['scale']:.2f}")
print(f"  Shape (ξ): {gpd_params['shape']:.4f}")

# Calculate return levels
return_periods = [10, 50, 100, 500]
print(f"\nReturn Levels:")
for T in return_periods:
    level = evt.calculate_return_level(T, method='gev')
    print(f"  {T}-year return level: {level:.2f}")

# 2. Focal Loss for Imbalanced Learning
print("\n" + "-"*70)
print("2. Focal Loss for Imbalanced Data")
print("-"*70)

# Create binary labels (extreme vs normal)
threshold_95 = np.quantile(base_data, 0.95)
y_true = (base_data > threshold_95).astype(int)

# Simulate predictions
y_pred_proba = np.random.beta(2, 5, size=n_samples)
y_pred_proba[y_true == 1] += 0.3  # Better predictions for extremes
y_pred_proba = np.clip(y_pred_proba, 0, 1)

focal_loss_trainer = FocalLossTrainer(alpha=0.25, gamma=2.0)

# Compute focal loss
fl = focal_loss_trainer.focal_loss(y_true, y_pred_proba)

# Compare with standard cross-entropy
ce = -np.mean(y_true * np.log(y_pred_proba + 1e-7) + 
              (1 - y_true) * np.log(1 - y_pred_proba + 1e-7))

print(f"Focal Loss: {fl:.4f}")
print(f"Cross-Entropy Loss: {ce:.4f}")
print(f"Class distribution: {np.sum(y_true)}/{len(y_true)} positive")

# 3. Rare Event Ensemble Methods
print("\n" + "-"*70)
print("3. Enhanced Ensemble for Rare Events")
print("-"*70)

rare_event_ensemble = RareEventEnsemble(n_members=50)

# Initial state
initial_state = np.array([threshold_95])
extreme_indicator = 0.8  # High extreme weather index

# Generate ensemble
ensemble = rare_event_ensemble.generate_enhanced_perturbations(
    initial_state,
    extreme_indicator=extreme_indicator,
    base_std=5.0,
    alpha_extreme=2.0,
    threshold=0.5
)

print(f"Ensemble size: {rare_event_ensemble.n_members} members")
print(f"Extreme indicator: {extreme_indicator:.2f}")
print(f"Ensemble mean: {ensemble.mean():.2f}")
print(f"Ensemble spread: {ensemble.std():.2f}")
print(f"Enhanced perturbation factor: {1 + 2.0 * (extreme_indicator - 0.5):.2f}")

# 4. Extreme Event Verification
print("\n" + "-"*70)
print("4. Specialized Verification Metrics")
print("-"*70)

verifier = ExtremeEventVerification()

# ROC metrics
roc_metrics = verifier.compute_roc_metrics(y_true, y_pred_proba)
print(f"\nROC Analysis:")
print(f"  AUC-ROC: {roc_metrics['auc']:.4f}")

# Precision-Recall
pr_metrics = verifier.compute_precision_recall(y_true, y_pred_proba)
print(f"\nPrecision-Recall Analysis:")
print(f"  AUC-PR: {pr_metrics['auc_pr']:.4f}")
print(f"  Max F1 Score: {np.max(pr_metrics['f1']):.4f}")

# Critical Success Index
y_pred_binary = (y_pred_proba > 0.5).astype(int)
csi = verifier.critical_success_index(y_true, y_pred_binary)
print(f"\nCritical Success Index: {csi:.4f}")

# Visualize results
visualize_extreme_weather_results(
    base_data, evt, y_true, y_pred_proba, 
    roc_metrics, pr_metrics, ensemble
)

print("\n" + "="*70)
print("Extreme weather framework demonstration complete!")
print("="*70)
def visualize_extreme_weather_results(data, evt, y_true, y_pred_proba,
roc_metrics, pr_metrics, ensemble):
"""Visualize extreme weather prediction results"""
fig = plt.figure(figsize=(16, 12))
# Plot 1: Data distribution with GEV fit
ax1 = plt.subplot(3, 3, 1)
ax1.hist(data, bins=50, density=True, alpha=0.7, label='Data')

# GEV PDF
x = np.linspace(data.min(), data.max(), 200)
shape_param = -evt.gev_params['shape']
gev_pdf = stats.genextreme.pdf(
    x, shape_param,
    loc=evt.gev_params['location'],
    scale=evt.gev_params['scale']
)
ax1.plot(x, gev_pdf, 'r-', linewidth=2, label='GEV Fit')
ax1.set_xlabel('Value')
ax1.set_ylabel('Density')
ax1.set_title('Data Distribution and GEV Fit')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Q-Q plot
ax2 = plt.subplot(3, 3, 2)
block_maxima, theoretical = evt.diagnostic_plots(data)
ax2.scatter(np.sort(theoretical), np.sort(block_maxima), alpha=0.6)
lims = [min(theoretical.min(), block_maxima.min()),
        max(theoretical.max(), block_maxima.max())]
ax2.plot(lims, lims, 'r--', linewidth=2)
ax2.set_xlabel('Theoretical Quantiles')
ax2.set_ylabel('Sample Quantiles')
ax2.set_title('Q-Q Plot (GEV)')
ax2.grid(True, alpha=0.3)

# Plot 3: Return levels
ax3 = plt.subplot(3, 3, 3)
return_periods = np.array([2, 5, 10, 20, 50, 100, 200, 500])
return_levels = [evt.calculate_return_level(T, 'gev') for T in return_periods]
ax3.semilogx(return_periods, return_levels, 'bo-', linewidth=2, markersize=8)
ax3.set_xlabel('Return Period (years)')
ax3.set_ylabel('Return Level')
ax3.set_title('Return Level Plot')
ax3.grid(True, alpha=0.3, which='both')

# Plot 4: ROC Curve
ax4 = plt.subplot(3, 3, 4)
ax4.plot(roc_metrics['fpr'], roc_metrics['tpr'], 'b-', linewidth=2,
        label=f"AUC = {roc_metrics['auc']:.3f}")
ax4.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
ax4.set_xlabel('False Positive Rate')
ax4.set_ylabel('True Positive Rate')
ax4.set_title('ROC Curve')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Precision-Recall Curve
ax5 = plt.subplot(3, 3, 5)
ax5.plot(pr_metrics['recall'], pr_metrics['precision'], 'g-', linewidth=2,
        label=f"AUC-PR = {pr_metrics['auc_pr']:.3f}")
baseline = np.sum(y_true) / len(y_true)
ax5.axhline(baseline, color='r', linestyle='--', linewidth=2, label='Baseline')
ax5.set_xlabel('Recall')
ax5.set_ylabel('Precision')
ax5.set_title('Precision-Recall Curve')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: F1 Score vs Threshold
ax6 = plt.subplot(3, 3, 6)
ax6.plot(pr_metrics['thresholds'], pr_metrics['f1'], 'purple', linewidth=2)
best_idx = np.argmax(pr_metrics['f1'])
ax6.plot(pr_metrics['thresholds'][best_idx], pr_metrics['f1'][best_idx],
        'ro', markersize=10, label=f"Max F1 = {pr_metrics['f1'][best_idx]:.3f}")
ax6.set_xlabel('Threshold')
ax6.set_ylabel('F1 Score')
ax6.set_title('F1 Score vs Classification Threshold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Plot 7: Ensemble spread
ax7 = plt.subplot(3, 3, 7)
ax7.hist(ensemble.flatten(), bins=30, alpha=0.7, color='orange')
ax7.axvline(ensemble.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean = {ensemble.mean():.2f}')
ax7.set_xlabel('Ensemble Value')
ax7.set_ylabel('Frequency')
ax7.set_title('Ensemble Distribution (Enhanced Perturbations)')
ax7.legend()
ax7.grid(True, alpha=0.3)

# Plot 8: Prediction reliability
ax8 = plt.subplot(3, 3, 8)
n_bins = 10
bins = np.linspace(0, 1, n_bins + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Calculate observed frequencies for each predicted probability bin
observed_freq = []
for i in range(n_bins):
    mask = (y_pred_proba >= bins[i]) & (y_pred_proba < bins[i+1])
    if np.sum(mask) > 0:
        observed_freq.append(np.mean(y_true[mask]))
    else:
        observed_freq.append(0)

ax8.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
ax8.plot(bin_centers, observed_freq, 'bo-', linewidth=2, markersize=8,
        label='Model')
ax8.set_xlabel('Predicted Probability')
ax8.set_ylabel('Observed Frequency')
ax8.set_title('Reliability Diagram')
ax8.legend()
ax8.grid(True, alpha=0.3)

# Plot 9: Confusion matrix
ax9 = plt.subplot(3, 3, 9)
y_pred_binary = (y_pred_proba > 0.5).astype(int)
cm = np.zeros((2, 2))
cm[0, 0] = np.sum((y_true == 0) & (y_pred_binary == 0))  # TN
cm[0, 1] = np.sum((y_true == 0) & (y_pred_binary == 1))  # FP
cm[1, 0] = np.sum((y_true == 1) & (y_pred_binary == 0))  # FN
cm[1, 1] = np.sum((y_true == 1) & (y_pred_binary == 1))  # TP

im = ax9.imshow(cm, cmap='Blues')
ax9.set_xticks([0, 1])
ax9.set_yticks([0, 1])
ax9.set_xticklabels(['Normal', 'Extreme'])
ax9.set_yticklabels(['Normal', 'Extreme'])
ax9.set_xlabel('Predicted')
ax9.set_ylabel('True')
ax9.set_title('Confusion Matrix')

# Add text annotations
for i in range(2):
    for j in range(2):
        text = ax9.text(j, i, int(cm[i, j]),
                      ha="center", va="center", color="black", fontsize=14)

plt.colorbar(im, ax=ax9)

plt.tight_layout()
plt.savefig('extreme_weather_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nResults saved to 'extreme_weather_results.png'")
if name == "main":
demonstrate_extreme_weather_framework()
