# Chapter 5: Extreme Weather Event Prediction - This folder contains code examples, notebooks, and datasets accompanying Chapter 5 of "AI in Climate Science: Machine Learning for Environmental Modeling and Prediction."Chapter OverviewThis chapter develops comprehensive mathematical frameworks for AI-enhanced extreme weather prediction, including:
Mathematical foundations of extreme weather prediction
Deep learning architectures for extreme event detection
Physics-informed neural networks for extreme weather dynamics
Advanced ensemble methods for rare event uncertainty quantification
Spectral analysis and multi-scale extreme weather modeling
Advanced optimization techniques for imbalanced datasets
Probabilistic forecasting and extreme value analysis
Real-time processing and operational deployment
Advanced evaluation frameworks for extreme weather models
Contentschapter-05-extreme-weather/
├── README.md (this file)
├── notebooks/
│   ├── 01_extreme_value_theory.ipynb
│   ├── 02_imbalanced_learning_extreme_events.ipynb
│   ├── 03_attention_mechanisms_extremes.ipynb
│   ├── 04_rare_event_ensemble_methods.ipynb
│   ├── 05_multi_scale_extreme_detection.ipynb
│   ├── 06_probabilistic_extreme_forecasting.ipynb
│   └── 07_operational_warning_systems.ipynb
├── scripts/
│   ├── extreme_weather_framework.py
│   ├── evt_neural_networks.py
│   ├── focal_loss_training.py
│   └── extreme_event_verification.py
└── data/
    ├── extreme_events_catalog.csv
    ├── hurricane_tracks.nc
    └── README.mdKey Concepts and Mathematical Foundations1. Extreme Value TheoryGeneralized Extreme Value (GEV) Distribution:
F(x;μ,σ,ξ) = exp{-[1 + ξ(x-μ)/σ]^{-1/ξ}}

ξ > 0: Fréchet domain (heavy tails)
ξ = 0: Gumbel domain (exponential tails)
ξ < 0: Weibull domain (bounded)Generalized Pareto Distribution (GPD):
F(x;σ,ξ) = 1 - (1 + ξx/σ)^{-1/ξ}Return Level:
x_T = μ + (σ/ξ)[(-ln(1-1/T))^{-ξ} - 1]2. Lyapunov Exponent and PredictabilityDivergence Rate:
λ = lim_{t→∞} (1/t)ln(|δ(t)|/|δ(0)|)Predictability Timescale:
τ_pred = (1/λ)ln(Δ_saturation/Δ_initial)3. Extreme Value Loss FunctionL_extreme = L_mse + λ_tail L_tail + λ_evt L_evt4. Deep Learning for ExtremesScale-Adaptive Convolution:
(X *_adaptive W)_{i,j} = Σ_{m,n} X_{i+d_m·m, j+d_n·n} · W_{m,n}Extreme-Aware Attention:
Attention_extreme(Q,K,V) = softmax(QK^T/√d_k + βE_extreme)VExtreme-LSTM Gates:
f_t = σ(W_f·[h_{t-1},x_t] + b_f + γ_f I_extreme)
i_t = σ(W_i·[h_{t-1},x_t] + b_i + γ_i I_extreme)
o_t = σ(W_o·[h_{t-1},x_t] + b_o + γ_o I_extreme)Graph Neural Networks for Extremes:
m_ij^(l) = Message(h_i^(l), h_j^(l), e_ij^meteor)
h_i^(l+1) = Update(h_i^(l), Σ_{j∈N(i)} m_ij^(l))5. Physics-Informed ConstraintsCAPE (Convective Available Potential Energy):
CAPE = g ∫_{z_LFC}^{z_EL} (T_{v,parcel} - T_{v,env})/T_{v,env} dzObukhov Length:
L = -u_*³ T_v/(κg w'θ_v')PINN Loss for Extremes:
L_PINN-extreme = L_data + λ_pde L_pde + λ_bc L_bc + λ_conserve L_conserve + λ_extreme L_extreme6. Ensemble Methods for Rare EventsEnhanced Perturbation Scaling:
σ_pert^extreme = σ_pert^standard · (1 + α_extreme · max(0, I_extreme - I_threshold))Importance Sampling:
P̂_extreme = (1/N)Σ I(X_i∈A) p(X_i)/q(X_i)Bayesian Model Averaging:
p(y*|x*,D) = Σ_k p(y*|x*,M_k,D) p(M_k|D)7. Optimization for Imbalanced DataFocal Loss:
L_focal(p_t) = -α_t(1-p_t)^γ log(p_t)Adam with Momentum:
v_{t+1} = β₁v_t + (1-β₁)∇L
s_{t+1} = β₂s_t + (1-β₂)(∇L)²
θ_{t+1} = θ_t - η/(√s_{t+1} + ε) ⊙ v_{t+1}8. Probabilistic ForecastingBayesian Posterior:
p(θ|D_extreme) ∝ p(D_extreme|θ) p(θ)Evidence Lower Bound (ELBO):
L_ELBO = E_{q_φ(θ)}[log p(D_extreme|θ)] - KL[q_φ(θ)||p(θ)]Conformal Prediction:
C(x) = {y : s(x,y) ≤ Q_{1-α}({s(x_i,y_i)})}9. Evaluation MetricsROC Metrics:
TPR(τ) = TP/(TP + FN)
FPR(τ) = FP/(FP + TN)Precision-Recall:
Precision = TP/(TP + FP)
Recall = TP/(TP + FN)
F1 = 2·Precision·Recall/(Precision + Recall)Extreme Dependency Score:
EDS = 2log(F_o(p))/log(F_{f,o}(p,p)) - 1Economic Value Score:
EVS = (H - F·C/L)/(1-s) - C/L10. Real-Time Processing4D-Var Data Assimilation:
J(x) = (1/2)(x-x_b)^T B^{-1}(x-x_b) + (1/2)Σ_i(y_i-H_i(x_i))^T R_i^{-1}(y_i-H_i(x_i))Load Balancing:
w_p = α·C_p + β·L_p + γ·Q_pFederated Learning:
w_global^{t+1} = Σ_k (n_k/n)w_k^{t+1}Running the ExamplesPrerequisites
bashpip install numpy pandas scipy xarray netCDF4 tensorflow torch matplotlib scikit-learn imbalanced-learnJupyter Notebooks
bashcd chapter-05-extreme-weather/notebooks
jupyter labPython Scripts
bashcd chapter-05-extreme-weather/scripts
python extreme_weather_framework.pyNotebook Descriptions01_extreme_value_theory.ipynb

GEV and GPD fitting
Return level calculation
Tail behavior analysis
Block maxima and peaks-over-threshold methods
02_imbalanced_learning_extreme_events.ipynb

Focal loss implementation
SMOTE for rare events
Class weighting strategies
Evaluation metrics for imbalanced data
03_attention_mechanisms_extremes.ipynb

Extreme-aware attention
Teleconnection detection
Multi-head attention for scale interactions
Spatial attention for hazard localization
04_rare_event_ensemble_methods.ipynb

Enhanced perturbation strategies
Importance sampling
Multilevel Monte Carlo
Bayesian model averaging
05_multi_scale_extreme_detection.ipynb

Wavelet decomposition
Scale-adaptive convolutions
Energy cascade analysis
Multi-resolution neural networks
06_probabilistic_extreme_forecasting.ipynb

Bayesian neural networks
Conformal prediction
Quantile regression
Uncertainty calibration
07_operational_warning_systems.ipynb

Real-time data processing
Alert generation algorithms
Performance under operational constraints
Decision support systems
Extreme Event Types CoveredTropical Cyclones

Track prediction
Intensity forecasting
Rapid intensification detection
Landfall timing
Severe Convective Storms

Tornado prediction
Hail forecasting
Damaging wind events
Flash flood warnings
Heat Waves

Duration prediction
Spatial extent forecasting
Health impact assessment
Urban heat island effects
Cold Waves and Winter Storms

Extreme cold prediction
Blizzard forecasting
Ice storm warnings
Wind chill modeling
Extreme Precipitation

Flash flood prediction
Heavy rainfall forecasting
Atmospheric river detection
Orographic enhancement
Performance MetricsClassification Metrics

Accuracy: Overall correctness
Precision: Positive predictive value
Recall (Sensitivity): True positive rate
F1 Score: Harmonic mean of precision/recall
AUC-ROC: Area under ROC curve
AUC-PR: Area under precision-recall curve
Probabilistic Metrics

Brier Score: Probability forecast accuracy
Reliability Diagram: Calibration assessment
Sharpness: Forecast resolution
CRPS: Continuous ranked probability score
Extreme-Specific Metrics

EDS: Extreme dependency score
EVS: Economic value score
POD: Probability of detection
FAR: False alarm ratio
CSI: Critical success index
Key Figures from Chapter
Figure 5.1: Mathematical Foundations of Extreme Weather Prediction
Figure 5.2: Deep Learning Architectures for Extreme Events
Figure 5.5: Probabilistic Forecasting Framework
Figure 5.6: Real-Time Processing Architecture
ReferencesKey papers discussed in this chapter:

Camps et al. (2024) - Artificial intelligence for extreme events
Dewitte et al. (2021) - AI in climate extremes
Mu et al. (2025) - Predictability studies
Li et al. (2021) - Robust prediction methods
Singh et al. (2023) - Enhanced forecasting
Advanced TopicsFoundation Models for Extremes

Pre-training on extreme event catalogs
Transfer learning across hazard types
Few-shot learning for rare events
Explainable AI for Extremes

Feature attribution for warnings
Physical interpretation of predictions
Uncertainty communication
Multi-Hazard Prediction

Compound extreme events
Cascading hazards
Multi-variate extremes
ApplicationsEarly Warning Systems

Multi-hazard monitoring
Alert dissemination
Impact-based forecasting
Decision support
Risk Assessment

Return period estimation
Vulnerability mapping
Climate change impacts
Infrastructure planning
Emergency Management

Resource allocation
Evacuation planning
Damage assessment
Recovery optimization
Further Reading
Full chapter text in the book
Additional tutorials: https://climate-ai-book.readthedocs.io
Discussion forum: https://github.com/climate-ai-book/examples/discussions
ContributingFound an error or have an improvement? Please open an issue or submit a pull request.LicenseMIT License - See main repository LICENSE fileContact
GitHub Issues: https://github.com/climate-ai-book/examples/issues
Email: contact@climate-ai-book.org
Part of "AI in Climate Science: Machine Learning for Environmental Modeling and Prediction"
by Gupta, Kanwer, and Ahmad (2025)
