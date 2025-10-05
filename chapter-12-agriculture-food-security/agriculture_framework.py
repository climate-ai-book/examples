"""
Agricultural and Food Security Applications Framework
Implements ML methods for agriculture and food systems:

Random forest for crop yield prediction
Transfer learning for food security monitoring
CNN for food quality detection
Ensemble methods for disease identification

Based on Chapter 12: Agricultural and Food Security Applications
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')
@dataclass
class CropData:
"""Data structure for crop information"""
biomass: float
lai: float  # Leaf area index
ndvi: float
evi: float
temperature: float
precipitation: float
soil_moisture: float
gdd: float  # Growing degree days
class CropGrowthModel:
"""
Crop growth modeling with biomass dynamics
Simulates crop development based on environmental conditions
"""

def __init__(self, crop_type: str = 'maize'):
    """Initialize crop growth model"""
    self.crop_type = crop_type
    
    # Growth parameters (crop-specific)
    if crop_type == 'maize':
        self.mu_max = 0.15  # Maximum specific growth rate (1/day)
        self.respiration_rate = 0.02  # Respiration coefficient
        self.senescence_rate = 0.01  # Senescence rate
        self.temp_opt = 28  # Optimal temperature (°C)
        self.temp_min = 10
        self.temp_max = 40
    else:
        # Default parameters
        self.mu_max = 0.12
        self.respiration_rate = 0.02
        self.senescence_rate = 0.01
        self.temp_opt = 25
        self.temp_min = 8
        self.temp_max = 38

def temperature_response(self, temp: float) -> float:
    """
    Temperature response function
    
    Cardinal temperature model
    """
    if temp < self.temp_min or temp > self.temp_max:
        return 0
    
    # Simplified cardinal temperature response
    if temp <= self.temp_opt:
        response = (temp - self.temp_min) / (self.temp_opt - self.temp_min)
    else:
        response = (self.temp_max - temp) / (self.temp_max - self.temp_opt)
    
    return max(0, min(1, response))

def water_response(self, soil_moisture: float) -> float:
    """
    Soil moisture stress response
    
    Optimal around field capacity (0.3-0.4)
    """
    # Simplified moisture response
    if soil_moisture < 0.15:
        return soil_moisture / 0.15
    elif soil_moisture > 0.6:
        return np.exp(-(soil_moisture - 0.6) / 0.2)
    else:
        return 1.0

def growth_rate(self, temp: float, soil_moisture: float,
               light: float = 1.0, nutrients: float = 1.0) -> float:
    """
    Specific growth rate μ(I,T,W,N)
    
    Multiplicative stress model
    """
    temp_factor = self.temperature_response(temp)
    water_factor = self.water_response(soil_moisture)
    
    mu = self.mu_max * temp_factor * water_factor * light * nutrients
    
    return mu

def simulate_growth(self, initial_biomass: float,
                   days: int, weather_data: Dict) -> np.ndarray:
    """
    Simulate crop growth over time
    
    dB/dt = μ(I,T,W,N)·B - R(T)·B - L_senescence
    """
    biomass = np.zeros(days + 1)
    biomass[0] = initial_biomass
    
    for day in range(days):
        # Current biomass
        B = biomass[day]
        
        # Environmental conditions
        temp = weather_data['temperature'][day]
        sm = weather_data['soil_moisture'][day]
        
        # Growth rate
        mu = self.growth_rate(temp, sm)
        
        # Respiration
        R = self.respiration_rate * self.temperature_response(temp)
        
        # Senescence (increases over time)
        L_sen = self.senescence_rate * (day / days)
        
        # Biomass change
        dB_dt = mu * B - R * B - L_sen * B
        
        # Update biomass (simple Euler integration)
        biomass[day + 1] = B + dB_dt
        biomass[day + 1] = max(biomass[day + 1], 0)
    
    return biomass
class CropYieldPredictor:
"""
Machine learning-based crop yield prediction
Uses vegetation indices and weather data
"""

def __init__(self):
    """Initialize yield predictor"""
    # Simplified linear model (in practice, use Random Forest/NN)
    self.weights = None
    self.trained = False
    
def extract_features(self, crop_data: List[CropData]) -> np.ndarray:
    """
    Extract features from crop data
    """
    n_samples = len(crop_data)
    features = np.zeros((n_samples, 10))
    
    for i, data in enumerate(crop_data):
        features[i, 0] = data.ndvi
        features[i, 1] = data.evi
        features[i, 2] = data.lai
        features[i, 3] = data.gdd
        features[i, 4] = data.precipitation
        features[i, 5] = data.temperature
        features[i, 6] = data.soil_moisture
        features[i, 7] = data.ndvi * data.gdd  # Interaction
        features[i, 8] = data.precipitation * data.temperature
        features[i, 9] = data.lai ** 2
    
    return features

def train(self, X: np.ndarray, y: np.ndarray):
    """
    Train yield prediction model
    
    Simple linear regression as placeholder
    """
    # Add bias term
    X_with_bias = np.column_stack([np.ones(len(X)), X])
    
    # Least squares solution
    self.weights = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
    self.trained = True

def predict(self, X: np.ndarray) -> np.ndarray:
    """
    Predict crop yield
    
    Y = f_neural(W_weather, S_soil, M_management, V_vegetation, P_phenology)
    """
    if not self.trained:
        raise ValueError("Model must be trained first")
    
    # Add bias term
    X_with_bias = np.column_stack([np.ones(len(X)), X])
    
    predictions = X_with_bias @ self.weights
    
    return np.maximum(predictions, 0)  # Non-negative yields

def predict_with_uncertainty(self, X: np.ndarray,
                             noise_std: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict with uncertainty estimation
    """
    predictions = self.predict(X)
    
    # Simple uncertainty (in practice, use ensemble)
    uncertainty = noise_std * np.ones_like(predictions)
    
    return predictions, uncertainty
class FoodSecurityClassifier:
"""
Food security status classification
Multi-class prediction of food security states
"""

def __init__(self):
    """Initialize food security classifier"""
    self.states = ['Secure', 'Moderately Insecure', 'Severely Insecure']
    self.n_states = len(self.states)
    self.weights = None
    
def extract_features(self, data: Dict) -> np.ndarray:
    """
    Extract food security features
    
    FSI(r,t) = g_ML(P_production, A_access, U_utilization, S_stability)
    """
    features = []
    
    # Production dimension
    features.append(data.get('crop_production', 100))
    features.append(data.get('livestock_production', 50))
    features.append(data.get('food_imports', 30))
    
    # Access dimension
    features.append(data.get('household_income', 1000))
    features.append(data.get('food_prices', 1.0))
    features.append(data.get('market_access', 0.8))
    
    # Utilization dimension
    features.append(data.get('dietary_diversity', 5))
    features.append(data.get('nutrition_score', 0.7))
    features.append(data.get('water_access', 0.9))
    
    # Stability dimension
    features.append(data.get('production_variability', 0.2))
    features.append(data.get('conflict_index', 0.1))
    features.append(data.get('climate_shocks', 0))
    
    return np.array(features)

def predict_state(self, features: np.ndarray) -> Tuple[str, np.ndarray]:
    """
    Predict food security state
    
    Returns state and probability distribution
    """
    # Simplified scoring (in practice, use trained classifier)
    production_score = features[0:3].mean()
    access_score = features[3:6].mean()
    utilization_score = features[6:9].mean()
    stability_score = 1 - features[9:12].mean()
    
    # Overall food security score
    fs_score = (production_score * 0.3 +
               access_score * 0.3 +
               utilization_score * 0.2 +
               stability_score * 0.2)
    
    # Normalize to 0-100
    fs_score = fs_score / 10
    
    # Classify into states
    if fs_score > 70:
        state = 'Secure'
        probs = [0.8, 0.15, 0.05]
    elif fs_score > 40:
        state = 'Moderately Insecure'
        probs = [0.15, 0.7, 0.15]
    else:
        state = 'Severely Insecure'
        probs = [0.05, 0.25, 0.7]
    
    return state, np.array(probs)
class FoodQualityDetector:
"""
Computer vision for food quality assessment
Detects defects and quality attributes
"""

def __init__(self):
    """Initialize food quality detector"""
    self.quality_classes = ['Excellent', 'Good', 'Fair', 'Poor']
    
def extract_visual_features(self, image: np.ndarray) -> np.ndarray:
    """
    Extract features from food image
    
    Q_quality = f_CNN(I_visual, F_texture, S_spectral, G_geometry)
    """
    # Simplified feature extraction (in practice, use CNN)
    
    # Color statistics (RGB)
    color_mean = image.mean(axis=(0, 1))
    color_std = image.std(axis=(0, 1))
    
    # Texture features (simplified)
    gray = image.mean(axis=2)
    texture_std = gray.std()
    
    # Shape features (simplified)
    area = image.shape[0] * image.shape[1]
    
    features = np.concatenate([
        color_mean,
        color_std,
        [texture_std, area]
    ])
    
    return features

def detect_defects(self, image: np.ndarray,
                  threshold: float = 0.3) -> Dict:
    """
    Detect defects in food product
    
    Returns defect locations and types
    """
    # Simplified defect detection
    # In practice, use object detection (YOLO, Faster R-CNN)
    
    # Simulate defect detection
    defect_score = np.random.rand()
    
    if defect_score > threshold:
        defects = {
            'has_defects': True,
            'defect_type': 'bruise',
            'severity': defect_score,
            'location': (np.random.randint(0, image.shape[0]),
                       np.random.randint(0, image.shape[1]))
        }
    else:
        defects = {
            'has_defects': False,
            'defect_type': None,
            'severity': 0,
            'location': None
        }
    
    return defects

def classify_quality(self, features: np.ndarray) -> Tuple[str, float]:
    """
    Classify overall food quality
    """
    # Simplified quality scoring
    quality_score = np.random.rand()
    
    if quality_score > 0.8:
        quality_class = 'Excellent'
    elif quality_score > 0.6:
        quality_class = 'Good'
    elif quality_score > 0.4:
        quality_class = 'Fair'
    else:
        quality_class = 'Poor'
    
    return quality_class, quality_score
class DiseaseDetector:
"""
Crop disease detection using deep learning
Identifies plant diseases from leaf images
"""

def __init__(self):
    """Initialize disease detector"""
    self.diseases = [
        'Healthy',
        'Leaf Blight',
        'Powdery Mildew',
        'Rust',
        'Bacterial Spot'
    ]
    self.n_classes = len(self.diseases)
    
def preprocess_image(self, image: np.ndarray) -> np.ndarray:
    """
    Preprocess leaf image for disease detection
    """
    # Normalize to [0, 1]
    image_norm = image.astype(np.float32) / 255.0
    
    # Simple augmentation could go here
    
    return image_norm

def extract_disease_features(self, image: np.ndarray) -> np.ndarray:
    """
    Extract features for disease classification
    
    p(Disease | I) = Softmax(CNN_deep(I_leaf))
    """
    # Simplified feature extraction
    # In practice, use pre-trained CNN (ResNet, EfficientNet)
    
    # Color distribution
    color_hist = [np.histogram(image[:,:,i], bins=10)[0] 
                 for i in range(3)]
    color_features = np.concatenate(color_hist)
    
    # Texture (variance in local patches)
    patch_size = 16
    texture_features = []
    for i in range(0, image.shape[0] - patch_size, patch_size):
        for j in range(0, image.shape[1] - patch_size, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            texture_features.append(patch.std())
    
    texture_features = np.array(texture_features[:10])  # Limit size
    
    features = np.concatenate([
        color_features / color_features.sum(),
        texture_features
    ])
    
    return features

def predict_disease(self, image: np.ndarray) -> Tuple[str, np.ndarray, float]:
    """
    Predict disease class with confidence
    
    Returns: (disease_name, probabilities, confidence)
    """
    # Preprocess
    image_prep = self.preprocess_image(image)
    
    # Extract features
    features = self.extract_disease_features(image_prep)
    
    # Simplified prediction (random for demo)
    # In practice, use trained deep learning model
    probabilities = np.random.dirichlet(np.ones(self.n_classes))
    
    # Get prediction
    disease_idx = np.argmax(probabilities)
    disease_name = self.diseases[disease_idx]
    confidence = probabilities[disease_idx]
    
    return disease_name, probabilities, confidence
def demonstrate_agriculture_framework():
"""
Demonstrate agricultural and food security framework
"""
print("="*70)
print("Agricultural and Food Security Applications Framework")
print("Chapter 12: Agricultural and Food Security Applications")
print("="*70)
np.random.seed(42)

# 1. Crop Growth Simulation
print("\n" + "-"*70)
print("1. Crop Growth Modeling")
print("-"*70)

crop_model = CropGrowthModel(crop_type='maize')

print(f"Crop Type: {crop_model.crop_type.title()}")
print(f"Maximum growth rate: {crop_model.mu_max:.3f} day⁻¹")
print(f"Optimal temperature: {crop_model.temp_opt}°C")

# Simulate growing season
days = 120
weather_data = {
    'temperature': 20 + 10 * np.sin(np.linspace(0, np.pi, days)),
    'soil_moisture': 0.3 + 0.1 * np.random.randn(days)
}
weather_data['soil_moisture'] = np.clip(weather_data['soil_moisture'], 0.1, 0.6)

initial_biomass = 0.1  # kg/m²
biomass_trajectory = crop_model.simulate_growth(
    initial_biomass, days, weather_data
)

print(f"\nGrowing Season Simulation ({days} days):")
print(f"  Initial biomass: {initial_biomass:.2f} kg/m²")
print(f"  Final biomass: {biomass_trajectory[-1]:.2f} kg/m²")
print(f"  Total growth: {biomass_trajectory[-1] - initial_biomass:.2f} kg/m²")
print(f"  Harvest index estimate: 0.5")
print(f"  Estimated yield: {biomass_trajectory[-1] * 0.5:.2f} kg/m²")

# 2. Crop Yield Prediction
print("\n" + "-"*70)
print("2. Crop Yield Prediction from Remote Sensing")
print("-"*70)

yield_predictor = CropYieldPredictor()

# Generate synthetic training data
n_samples = 100
training_data = []
for _ in range(n_samples):
    data = CropData(
        biomass=np.random.uniform(5, 15),
        lai=np.random.uniform(2, 6),
        ndvi=np.random.uniform(0.5, 0.9),
        evi=np.random.uniform(0.3, 0.7),
        temperature=np.random.uniform(20, 30),
        precipitation=np.random.uniform(300, 800),
        soil_moisture=np.random.uniform(0.2, 0.5),
        gdd=np.random.uniform(1000, 2000)
    )
    training_data.append(data)

# Extract features
X_train = yield_predictor.extract_features(training_data)

# Generate synthetic yields (simplified relationship)
y_train = (2.0 + 
          5.0 * X_train[:, 0] +  # NDVI
          0.005 * X_train[:, 3] +  # GDD
          0.002 * X_train[:, 4] +  # Precipitation
          np.random.randn(n_samples) * 0.5)

# Train model
yield_predictor.train(X_train, y_train)

# Make predictions
test_data = CropData(
    biomass=10.0, lai=4.5, ndvi=0.75, evi=0.55,
    temperature=25, precipitation=600, soil_moisture=0.35, gdd=1500
)
X_test = yield_predictor.extract_features([test_data])

yield_pred, yield_unc = yield_predictor.predict_with_uncertainty(X_test)

print(f"Test Case Vegetation Indices:")
print(f"  NDVI: {test_data.ndvi:.2f}")
print(f"  EVI: {test_data.evi:.2f}")
print(f"  LAI: {test_data.lai:.2f}")
print(f"\nYield Prediction:")
print(f"  Predicted yield: {yield_pred[0]:.2f} ± {yield_unc[0]:.2f} Mg/ha")

# 3. Food Security Classification
print("\n" + "-"*70)
print("3. Food Security Monitoring")
print("-"*70)

fs_classifier = FoodSecurityClassifier()

# Scenario 1: Food secure region
secure_data = {
    'crop_production': 150,
    'livestock_production': 80,
    'food_imports': 40,
    'household_income': 1500,
    'food_prices': 0.8,
    'market_access': 0.9,
    'dietary_diversity': 7,
    'nutrition_score': 0.85,
    'water_access': 0.95,
    'production_variability': 0.1,
    'conflict_index': 0.05,
    'climate_shocks': 0
}

features_secure = fs_classifier.extract_features(secure_data)
state_secure, probs_secure = fs_classifier.predict_state(features_secure)

print(f"Scenario 1: Food Secure Region")
print(f"  Production: High")
print(f"  Access: Good")
print(f"  Predicted state: {state_secure}")
print(f"  Confidence: {probs_secure[fs_classifier.states.index(state_secure)]*100:.1f}%")

# Scenario 2: Food insecure region
insecure_data = {
    'crop_production': 40,
    'livestock_production': 20,
    'food_imports': 10,
    'household_income': 300,
    'food_prices': 2.5,
    'market_access': 0.3,
    'dietary_diversity': 3,
    'nutrition_score': 0.4,
    'water_access': 0.5,
    'production_variability': 0.6,
    'conflict_index': 0.7,
    'climate_shocks': 2
}

features_insecure = fs_classifier.extract_features(insecure_data)
state_insecure, probs_insecure = fs_classifier.predict_state(features_insecure)

print(f"\nScenario 2: Food Insecure Region")
print(f"  Production: Low")
print(f"  Access: Poor")
print(f"  Predicted state: {state_insecure}")
print(f"  Confidence: {probs_insecure[fs_classifier.states.index(state_insecure)]*100:.1f}%")

# 4. Food Quality Detection
print("\n" + "-"*70)
print("4. Computer Vision for Food Quality")
print("-"*70)

quality_detector = FoodQualityDetector()

# Simulate food image (RGB)
food_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# Extract features
visual_features = quality_detector.extract_visual_features(food_image)

# Detect defects
defects = quality_detector.detect_defects(food_image)

# Classify quality
quality_class, quality_score = quality_detector.classify_quality(visual_features)

print(f"Food Product Analysis:")
print(f"  Image size: {food_image.shape[0]}x{food_image.shape[1]}")
print(f"  Quality class: {quality_class}")
print(f"  Quality score: {quality_score:.2f}")
print(f"  Defects detected: {'Yes' if defects['has_defects'] else 'No'}")
if defects['has_defects']:
    print(f"    Type: {defects['defect_type']}")
    print(f"    Severity: {defects['severity']:.2f}")

# 5. Disease Detection
print("\n" + "-"*70)
print("5. Crop Disease Detection")
print("-"*70)

disease_detector = DiseaseDetector()

# Simulate leaf image
leaf_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

# Predict disease
disease_name, disease_probs, confidence = disease_detector.predict_disease(leaf_image)

print(f"Leaf Image Analysis:")
print(f"  Image size: {leaf_image.shape[0]}x{leaf_image.shape[1]}")
print(f"  Predicted disease: {disease_name}")
print(f"  Confidence: {confidence*100:.1f}%")
print(f"\nDisease Probabilities:")
for i, disease in enumerate(disease_detector.diseases):
    print(f"  {disease:20s}: {disease_probs[i]*100:5.1f}%")

# Visualize results
visualize_agriculture_results(
    biomass_trajectory, weather_data,
    [state_secure, state_insecure],
    [probs_secure, probs_insecure],
    disease_probs, disease_detector.diseases
)

print("\n" + "="*70)
print("Agricultural framework demonstration complete!")
print("="*70)
def visualize_agriculture_results(biomass, weather, fs_states, fs_probs,
disease_probs, disease_names):
"""Visualize agricultural modeling results"""
fig = plt.figure(figsize=(16, 10))
days = len(biomass) - 1
time = np.arange(days + 1)

# Plot 1: Crop biomass growth
ax1 = plt.subplot(2, 3, 1)
ax1.plot(time, biomass, 'g-', linewidth=2)
ax1.fill_between(time, 0, biomass, alpha=0.3, color='green')
ax1.set_xlabel('Days After Planting')
ax1.set_ylabel('Biomass (kg/m²)')
ax1.set_title('Crop Growth Trajectory')
ax1.grid(True, alpha=0.3)

# Plot 2: Environmental conditions
ax2 = plt.subplot(2, 3, 2)
ax2_twin = ax2.twinx()

l1 = ax2.plot(weather['temperature'], 'r-', linewidth=2, label='Temperature')
ax2.set_xlabel('Day')
ax2.set_ylabel('Temperature (°C)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

l2 = ax2_twin.plot(weather['soil_moisture'], 'b-', linewidth=2, label='Soil Moisture')
ax2_twin.set_ylabel('Soil Moisture', color='b')
ax2_twin.tick_params(axis='y', labelcolor='b')

ax2.set_title('Environmental Conditions')
ax2.grid(True, alpha=0.3)

# Plot 3: Food security states
ax3 = plt.subplot(2, 3, 3)
scenarios = ['Scenario 1\n(Secure)', 'Scenario 2\n(Insecure)']
colors_fs = ['green', 'red']

bars = ax3.bar(scenarios, [fs_probs[0][0], fs_probs[1][2]],
               color=colors_fs, alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_ylabel('Probability')
ax3.set_title('Food Security Classification')
ax3.set_ylim([0, 1])
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height*100:.1f}%', ha='center', va='bottom', fontweight='bold')

# Plot 4: Food security probability distribution (Scenario 1)
ax4 = plt.subplot(2, 3, 4)
states = ['Secure', 'Mod.\nInsecure', 'Sev.\nInsecure']
colors_states = ['green', 'orange', 'red']

bars4 = ax4.bar(states, fs_probs[0], color=colors_states,
                alpha=0.7, edgecolor='black', linewidth=2)
ax4.set_ylabel('Probability')
ax4.set_title('Food Security States (Scenario 1)')
ax4.set_ylim([0, 1])
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Disease detection probabilities
ax5 = plt.subplot(2, 3, 5)
colors_disease = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(disease_names)))

bars5 = ax5.barh(disease_names, disease_probs, color=colors_disease,
                 edgecolor='black', linewidth=1)
ax5.set_xlabel('Probability')
ax5.set_title('Disease Detection Results')
ax5.set_xlim([0, max(disease_probs) * 1.2])
ax5.grid(True, alpha=0.3, axis='x')

# Add value labels
for bar, prob in zip(bars5, disease_probs):
    width = bar.get_width()
    ax5.text(width, bar.get_y() + bar.get_height()/2.,
            f'{prob*100:.1f}%', ha='left', va='center',
            fontweight='bold', fontsize=9)

# Plot 6: Yield components
ax6 = plt.subplot(2, 3, 6)
components = ['Biomass\n(Final)', 'Harvest\nIndex', 'Estimated\nYield']
values = [biomass[-1], 0.5, biomass[-1] * 0.5]
colors_comp = ['darkgreen', 'gold', 'brown']

bars6 = ax6.bar(components, values, color=colors_comp,
                alpha=0.7, edgecolor='black', linewidth=2)
ax6.set_ylabel('kg/m²')
ax6.set_title('Yield Components')
ax6.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars6, values):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('agriculture_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nResults saved to 'agriculture_results.png'")
if name == "main":
demonstrate_agriculture_framework()
