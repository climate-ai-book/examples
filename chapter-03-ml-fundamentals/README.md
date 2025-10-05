# Chapter 3: Machine Learning Fundamentals for Climate Applications - This folder contains code examples, notebooks, and datasets accompanying Chapter 3 of "AI in Climate Science: Machine Learning for Environmental Modeling and Prediction."Chapter OverviewThis chapter establishes rigorous mathematical frameworks for applying machine learning to climate science, including:
Information-theoretic foundations for climate data analysis
Stochastic modeling of climate variability
Supervised learning algorithms for climate prediction
Deep learning architectures for spatiotemporal data
Physics-informed machine learning
Unsupervised learning for pattern discovery
Spectral methods and multi-scale analysis
Bayesian methods and uncertainty quantification
Model interpretability and explainability
Contentschapter-03-ml-fundamentals/
├── README.md (this file)
├── notebooks/
│   ├── 01_information_theory_ml.ipynb
│   ├── 02_supervised_learning_climate.ipynb
│   ├── 03_deep_learning_architectures.ipynb
│   ├── 04_physics_informed_networks.ipynb
│   ├── 05_unsupervised_pattern_discovery.ipynb
│   ├── 06_bayesian_uncertainty.ipynb
│   └── 07_model_interpretability.ipynb
├── scripts/
│   ├── climate_ml_framework.py
│   ├── pinn_implementation.py
│   ├── lstm_climate_model.py
│   └── interpretability_tools.py
└── data/
    ├── sample_climate_timeseries.nc
    ├── spatial_climate_fields.nc
    └── README.mdKey Concepts and Mathematical Foundations1. Information Theory for Climate MLDifferential Entropy:
H(X) = -∫ p(x)log p(x) dxMutual Information:
I(X;Y) = ∫∫ p(x,y)log(p(x,y)/(p(x)p(y))) dx dyKL Divergence:
D_KL(P||Q) = ∫ p(x)log(p(x)/q(x)) dx2. Stochastic Climate ModelingStochastic Differential Equation:
dX_t = f(X_t,t,θ)dt + g(X_t,t,θ)dW_tFokker-Planck Equation:
∂p/∂t = -Σ_i ∂/∂x_i[f_i(x,t)p] + (1/2)Σ_ij ∂²/∂x_i∂x_j[G_ij(x,t)p]VAR Model:
X_t = c + Σ_{i=1}^p A_i X_{t-i} + ε_tState-Space Model:
x_t = F_t x_{t-1} + G_t u_t + w_t  (state equation)
y_t = H_t x_t + v_t                 (observation equation)3. Supervised LearningGeneral Framework:
f̂ = argmin_{f∈F} (1/n)Σ_i L(y_i, f(x_i)) + λR(f)Kernel SVR:
f*(x) = Σ_i α_i k(x_i, x)Random Forest:
f_RF(x) = (1/B)Σ_{b=1}^B T_b(x)Gradient Boosting:
F_m(x) = F_{m-1}(x) + γ_m h_m(x)4. Deep Learning ArchitecturesUniversal Approximation:
f(x) = Σ_{i=1}^N w_i σ(v_i^T x + b_i) + w_0Convolution (Circular for Global Data):
(X *_circ W)_{i,j} = Σ_{m,n} X_{(i+m) mod H,(j+n) mod W} W_{m,n}LSTM Gates:
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)      # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)      # Input gate
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  # Candidate
C_t = f_t * C_{t-1} + i_t * C̃_t         # Cell state
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)      # Output gate
h_t = o_t * tanh(C_t)                     # Hidden state5. Physics-Informed Neural Networks (PINNs)PINN Loss Function:
L = L_data + λ_pde L_pde + λ_bc L_bcConservation Constraints:
L_momentum = Σ_i |∂v/∂t + (v·∇)v + fk×v + (1/ρ)∇p|²
L_continuity = Σ_i |∂ρ/∂t + ∇·(ρv)|²
L_thermodynamic = Σ_i |∂T/∂t + v·∇T - Q/c_p|²6. Unsupervised LearningPCA:
Cv_i = λ_i v_i
EVR_i = λ_i / Σ_j λ_jICA (FastICA):
x = As
w_new = E[xg(w^T x)] - E[g'(w^T x)]wK-means:
J = Σ_{i=1}^k Σ_{x∈C_i} ||x - μ_i||²Gaussian Mixture Model:
p(x) = Σ_{k=1}^K π_k N(x|μ_k, Σ_k)7. Spectral AnalysisDiscrete Fourier Transform:
X_k = Σ_{n=0}^{N-1} x_n e^{-2πikn/N}Continuous Wavelet Transform:
W(a,b) = (1/√a) ∫ x(t)ψ*((t-b)/a) dtEmpirical Mode Decomposition:
x(t) = Σ_{i=1}^n c_i(t) + r_n(t)8. Bayesian MethodsBayesian Linear Regression:
p(β|y,X,σ²) = N(μ_n, Σ_n)
p(y*|x*,y,X) = N(x*^T μ_n, σ² + x*^T Σ_n x*)Gaussian Process:
f(x) ~ GP(m(x), k(x,x'))
f̄* = k^T(K + σ_n²I)^{-1} y
var[f*] = k* - k^T(K + σ_n²I)^{-1} kMonte Carlo Dropout:
p(y*|x*,D) ≈ (1/T)Σ_{t=1}^T f(x*; w_t)9. OptimizationAdam Optimizer:
m_t = β₁m_{t-1} + (1-β₁)∇L
v_t = β₂v_{t-1} + (1-β₂)(∇L)²
θ_{t+1} = θ_t - η/(√v̂_t + ε) m̂_tLearning Rate Schedules:
Cosine: η_t = η_min + (1/2)(η_max - η_min)(1 + cos(T_cur/T_max π))
Exponential: η_t = η_0 e^{-λt}10. InterpretabilitySHAP Values:
φ_i = Σ_{S⊆N\{i}} (|S|!(n-|S|-1)!/n!)[f_x(S∪{i}) - f_x(S)]Integrated Gradients:
IG_i(x) = (x_i - x'_i)∫_{α=0}^1 ∂f(x'+α(x-x'))/∂x_i dαPartial Dependence:
PD_{x_j}(z) = E_{x_{-j}}[f(z, x_{-j})]Running the ExamplesPrerequisites
bashpip install numpy pandas scipy xarray netCDF4 scikit-learn tensorflow torch matplotlib seaborn pywavelets shapJupyter Notebooks
bashcd chapter-03-ml-fundamentals/notebooks
jupyter labPython Scripts
bashcd chapter-03-ml-fundamentals/scripts
python climate_ml_framework.pyNotebook Descriptions01_information_theory_ml.ipynb

Calculate entropy and mutual information
Feature selection using information-theoretic measures
Teleconnection detection via mutual information
KL divergence for distribution comparison
02_supervised_learning_climate.ipynb

Linear regression with physics constraints
Kernel SVR for temperature prediction
Random Forests for precipitation modeling
Gradient boosting for extreme events
03_deep_learning_architectures.ipynb

Feedforward networks with universal approximation
CNNs for spatial climate patterns
LSTMs for temporal climate sequences
Hybrid spatiotemporal architectures
04_physics_informed_networks.ipynb

PINNs with conservation law constraints
Atmospheric dynamics equations
Thermodynamic consistency enforcement
Validation against physical bounds
05_unsupervised_pattern_discovery.ipynb

PCA for EOF analysis
ICA for signal separation
K-means for weather regime classification
GMM for probabilistic clustering
06_bayesian_uncertainty.ipynb

Bayesian linear regression
Gaussian Process regression
Bayesian neural networks
Uncertainty propagation
07_model_interpretability.ipynb

SHAP values for feature importance
Integrated gradients analysis
Partial dependence plots
LIME for local explanations
Performance MetricsClimate-Specific Metrics

Climate Skill Score: CSS = 1 - MSE_model/MSE_climatology
Anomaly Correlation: Skill in predicting departures from climatology
Brier Score: For probabilistic forecasts
Extremal Dependence: For extreme event prediction
Standard ML Metrics

RMSE, MAE, R²
Precision, Recall, F1 (for classification)
Log-likelihood
AIC, BIC (for model selection)
Validation StrategiesTemporal Cross-Validation

Expanding window CV
Sliding window CV
Walk-forward validation
Spatial Cross-Validation

Spatial block CV
Leave-location-out CV
Distance-based buffer zones
Key Figures from Chapter
Figure 3.1: Information-Theoretic Framework
Figure 3.2: Stochastic Modeling for Climate Systems
Figure 3.3: Deep Learning Architectures
Figure 3.4: Spectral Methods
Figure 3.5: Computational Framework
Figure 3.6: Interpretability Framework
ReferencesKey papers discussed in this chapter:

Chen et al. (2023) - Machine learning for climate
Bochenek et al. (2022) - Stochastic climate modeling
Krasnopolsky (2024) - Deep learning applications
Kashinath et al. (2021) - Physics-informed ML
Schultz et al. (2021) - Model validation strategies
Advanced TopicsTransfer Learning

Domain adaptation theory
Maximum Mean Discrepancy (MMD)
CORAL alignment
Adversarial domain adaptation
Computational Efficiency

Data parallelism
Model parallelism
Gradient checkpointing
Mixed precision training
Emerging Paradigms

Foundation models for climate
Quantum machine learning
Federated learning for climate data
Further Reading
Full chapter text in the book
Additional tutorials: https://climate-ai-book.readthedocs.io
Discussion forum: https://github.com/climate-ai-book/examples/discussions
ContributingFound an error or have an improvement? Please open an issue or submit a pull request.LicenseMIT License - See main repository LICENSE fileContact
GitHub Issues: https://github.com/climate-ai-book/examples/issues
Email: contact@climate-ai-book.org
Part of "AI in Climate Science: Machine Learning for Environmental Modeling and Prediction"
by Gupta, Kanwer, and Ahmad (2025)
