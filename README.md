# 🔬 Quantum ML vs Classical ML: Financial Fraud Detection

## Chứng minh ưu thế Quantum Machine Learning trong phát hiện gian lận tài chính

### 🎯 Mục tiêu

Dự án này xây dựng và chứng minh rằng **Quantum Machine Learning (QML)** vượt trội hơn ML truyền thống trong **chế độ dữ liệu nhỏ** (small-data regime) — một kịch bản phổ biến và quan trọng trong tài chính.

### 📊 Dataset

- **Credit Card Fraud Detection** (Kaggle/ULB)
- 284,807 giao dịch, 30 features (PCA-transformed)
- 0.172% giao dịch gian lận

### 🧠 Models

#### Quantum Models (PennyLane)
| Model | Mô tả |
|:---|:---|
| **QSVM** | Quantum Kernel SVM — ZZ Feature Map + precomputed kernel |
| **VQC** | Variational Quantum Classifier — Data re-uploading circuit |
| **Hybrid QNN** | Classical → Quantum → Classical neural network |

#### Classical Baselines (Scikit-learn, XGBoost, LightGBM)
| Model | Mô tả |
|:---|:---|
| **SVM-RBF** | Support Vector Machine with RBF kernel |
| **SVM-Poly** | Support Vector Machine with Polynomial kernel |
| **Random Forest** | Ensemble of decision trees |
| **XGBoost** | Gradient Boosted Trees (state-of-the-art tabular) |
| **LightGBM** | Light Gradient Boosting Machine |
| **MLP** | Multi-Layer Perceptron neural network |

### 🧪 Experiments

1. **Sample Size Scaling**: So sánh performance ở N = 100, 200, 300, 500, 750, 1000, 1500, 2000
2. **Imbalance Sensitivity**: Test robustness ở fraud ratio = 1%, 5%, 10%, 20%, 50%

### 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
python download_data.py

# 3. Run quick test (~10-20 min)
python run_all.py --quick-test

# 4. Run full experiment (~4-8 hours)
python run_all.py --full
```

### 📁 Project Structure

```
QM/
├── run_all.py                    # Main entry point
├── config.py                     # All hyperparameters
├── download_data.py              # Dataset downloader
├── requirements.txt
├── data/                         # Dataset directory
├── src/
│   ├── quantum_models.py         # QSVM, VQC, Hybrid QNN
│   ├── classical_models.py       # SVM, RF, XGBoost, LightGBM, MLP
│   ├── data_preprocessing.py     # Feature selection, scaling
│   ├── evaluation.py             # Metrics, statistical tests
│   └── visualization.py          # Charts and reports
├── experiments/
│   ├── experiment_runner.py      # Orchestration
│   ├── exp_sample_size.py        # Sample size experiment
│   └── exp_imbalance_ratio.py    # Imbalance experiment
└── results/
    ├── figures/                  # Generated charts
    ├── tables/                   # CSV results
    └── logs/                     # Logs
```

### 📈 Expected Results

Quantum models (đặc biệt QSVM) kỳ vọng vượt trội ở:
- **N ≤ 500**: Higher F1, Recall, AUC
- **Extreme imbalance**: Better minority class detection without SMOTE

### 🛠️ Technology Stack

- **PennyLane** — Quantum ML framework
- **Scikit-learn** — Classical ML
- **XGBoost / LightGBM** — Gradient boosting
- **Matplotlib / Seaborn** — Visualization
- **SciPy** — Statistical testing
