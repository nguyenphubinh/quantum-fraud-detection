"""
Global configuration for Quantum ML vs Classical ML benchmark project.
All hyperparameters, paths, and experiment settings are centralized here.
"""
import os
import numpy as np

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

# Create directories if they don't exist
for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

DATASET_PATH = os.path.join(DATA_DIR, "creditcard.csv")

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================================
# QUANTUM CIRCUIT PARAMETERS
# ============================================================================
# Number of qubits = number of features fed into quantum circuit
N_QUBITS = 6          # Use top-6 PCA features for quantum models
N_LAYERS_QSVM = 2     # Depth of quantum kernel feature map
N_LAYERS_VQC = 4       # Depth of variational circuit
N_LAYERS_HYBRID = 3    # Depth of hybrid QNN quantum layer

# Quantum Device Backend (use "lightning.gpu" on GPU-enabled Kaggle, otherwise "default.qubit")
QML_DEVICE = "default.qubit"

# VQC Training
VQC_LEARNING_RATE = 0.01
VQC_EPOCHS = 80
VQC_BATCH_SIZE = 16

# Hybrid QNN
HYBRID_HIDDEN_DIM = 16  # Classical hidden layer size
HYBRID_LEARNING_RATE = 0.005
HYBRID_EPOCHS = 80

# ============================================================================
# EXPERIMENT SETTINGS
# ============================================================================
# Sample sizes for the main experiment (quantum advantage at small N)
SAMPLE_SIZES = [100, 200, 300, 500, 750, 1000, 1500, 2000]

# Quick test uses smaller sample sizes
QUICK_SAMPLE_SIZES = [100, 200, 500]

# Imbalance ratios for secondary experiment
IMBALANCE_RATIOS = [0.01, 0.05, 0.10, 0.20, 0.50]  # Fraction of fraud

# Fixed sample size for imbalance experiment
IMBALANCE_FIXED_N = 500

# Number of repetitions per configuration (for statistical significance)
N_REPETITIONS = 5
QUICK_N_REPETITIONS = 2

# Cross-validation folds
CV_FOLDS = 5

# Number of top features to select for quantum models
TOP_N_FEATURES = N_QUBITS  # Must match number of qubits

# ============================================================================
# CLASSICAL MODEL HYPERPARAMETER GRIDS
# ============================================================================
SVM_RBF_PARAMS = {
    "C": [0.1, 1.0, 10.0],
    "gamma": ["scale", "auto"],
}

SVM_POLY_PARAMS = {
    "C": [0.1, 1.0, 10.0],
    "degree": [2, 3],
    "gamma": ["scale"],
}

RF_PARAMS = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
}

XGB_PARAMS = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.3],
    "subsample": [0.8, 1.0],
}

LGBM_PARAMS = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.3],
    "num_leaves": [15, 31],
}

MLP_PARAMS = {
    "hidden_layer_sizes": [(32,), (64,), (32, 16)],
    "alpha": [0.0001, 0.001],
    "learning_rate_init": [0.001, 0.01],
}

# ============================================================================
# VISUALIZATION
# ============================================================================
QUANTUM_COLOR = "#7C3AED"        # Purple for quantum models
CLASSICAL_COLOR = "#059669"      # Green for classical models
HYBRID_COLOR = "#D97706"         # Amber for hybrid models
ACCENT_COLOR = "#DC2626"         # Red for highlights

QUANTUM_MODELS_LIST = ["QSVM", "VQC", "Hybrid_QNN"]
CLASSICAL_MODELS_LIST = ["SVM_RBF", "SVM_Poly", "Random_Forest", "XGBoost", "LightGBM", "MLP"]

MODEL_COLORS = {
    "QSVM": "#7C3AED",
    "VQC": "#8B5CF6",
    "Hybrid_QNN": "#D97706",
    "SVM_RBF": "#059669",
    "SVM_Poly": "#10B981",
    "Random_Forest": "#2563EB",
    "XGBoost": "#DC2626",
    "LightGBM": "#16A34A",
    "MLP": "#64748B",
}
