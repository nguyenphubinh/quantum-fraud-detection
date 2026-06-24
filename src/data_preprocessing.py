"""
Data preprocessing pipeline for the Credit Card Fraud Detection dataset.

Handles:
- Loading and validating the dataset
- Feature selection (top-N by importance for quantum models)
- Stratified sampling at various sizes
- Balanced subsampling for quantum experiments
- Feature scaling for quantum encoding (bounded to [-pi, pi])
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATASET_PATH, RANDOM_SEED, TOP_N_FEATURES, DATA_DIR
)


def load_dataset(path=None):
    """
    Load the Credit Card Fraud Detection dataset.

    Returns:
        pd.DataFrame: Full dataset with columns V1-V28, Amount, Time, Class.
    """
    if path is None:
        path = DATASET_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            f"Run 'python download_data.py' first."
        )

    df = pd.read_csv(path)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Fraud: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
    print(f"   Legit: {(df['Class'] == 0).sum()}")
    return df


def select_top_features(df, n_features=TOP_N_FEATURES, method="mutual_info"):
    """
    Select the top-N most informative features for quantum models.

    Uses mutual information to rank features, since quantum circuits
    have limited qubit count and we need the most discriminative features.

    Args:
        df: Full dataframe.
        n_features: Number of features to select.
        method: 'mutual_info' or 'rf_importance'.

    Returns:
        list: Names of selected features.
    """
    feature_cols = [c for c in df.columns if c not in ["Class", "Time"]]
    X = df[feature_cols].values
    y = df["Class"].values

    if method == "mutual_info":
        scores = mutual_info_classif(X, y, random_state=RANDOM_SEED)
    elif method == "rf_importance":
        rf = RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1
        )
        rf.fit(X, y)
        scores = rf.feature_importances_
    else:
        raise ValueError(f"Unknown method: {method}")

    # Rank features
    feature_ranking = sorted(
        zip(feature_cols, scores), key=lambda x: x[1], reverse=True
    )

    selected = [f for f, _ in feature_ranking[:n_features]]
    print(f"✅ Selected top-{n_features} features: {selected}")
    for name, score in feature_ranking[:n_features]:
        print(f"   {name}: {score:.4f}")

    return selected


def create_balanced_subset(df, n_samples, fraud_ratio=0.5, seed=RANDOM_SEED):
    """
    Create a balanced subset with specified fraud ratio.

    For quantum experiments, we use balanced subsampling (50/50) at small sizes
    to maximize the signal for quantum kernels.

    Args:
        df: Full dataframe.
        n_samples: Total number of samples.
        fraud_ratio: Fraction of samples that should be fraud.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame: Balanced subset.
    """
    rng = np.random.RandomState(seed)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    fraud_df = df[df["Class"] == 1]
    legit_df = df[df["Class"] == 0]

    # Sample fraud instances (with replacement if needed)
    if n_fraud <= len(fraud_df):
        fraud_sample = fraud_df.sample(n=n_fraud, random_state=rng)
    else:
        fraud_sample = fraud_df.sample(
            n=n_fraud, replace=True, random_state=rng
        )

    # Sample legitimate instances
    legit_sample = legit_df.sample(n=n_legit, random_state=rng)

    subset = pd.concat([fraud_sample, legit_sample]).sample(
        frac=1, random_state=rng
    ).reset_index(drop=True)

    return subset


def prepare_quantum_data(X, scaler_type="minmax_pi"):
    """
    Scale features for quantum circuit encoding.

    Quantum feature maps typically require inputs bounded in [-pi, pi] or [0, 2*pi].
    We scale the classical features to this range.

    Args:
        X: Feature array (n_samples, n_features).
        scaler_type: 'minmax_pi' maps to [-pi, pi], 'standard' uses StandardScaler.

    Returns:
        X_scaled: Scaled feature array.
        scaler: Fitted scaler object (for transforming test data).
    """
    if scaler_type == "minmax_pi":
        scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}")

    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def prepare_experiment_data(
    df,
    n_samples,
    selected_features,
    fraud_ratio=0.5,
    test_size=0.2,
    seed=RANDOM_SEED,
):
    """
    Complete data preparation pipeline for one experiment run.

    Steps:
    1. Create balanced subset of specified size
    2. Select features
    3. Split into train/test
    4. Scale for quantum encoding (train fit, test transform)

    Args:
        df: Full dataframe.
        n_samples: Total samples for this experiment.
        selected_features: List of feature column names.
        fraud_ratio: Fraction of fraud in the subset.
        test_size: Fraction for test set.
        seed: Random seed.

    Returns:
        dict with keys:
            X_train, X_test, y_train, y_test: Quantum-scaled data
            X_train_raw, X_test_raw: StandardScaled data (for classical)
            scaler_quantum, scaler_classical: Fitted scalers
    """
    # Step 1: Balanced subset
    subset = create_balanced_subset(df, n_samples, fraud_ratio, seed)

    # Step 2: Select features
    X = subset[selected_features].values
    y = subset["Class"].values

    # Step 3: Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Step 4a: Quantum scaling ([-pi, pi])
    scaler_q = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_train_q = scaler_q.fit_transform(X_train)
    X_test_q = scaler_q.transform(X_test)

    # Step 4b: Classical scaling (StandardScaler)
    scaler_c = StandardScaler()
    X_train_c = scaler_c.fit_transform(X_train)
    X_test_c = scaler_c.transform(X_test)

    return {
        "X_train_quantum": X_train_q,
        "X_test_quantum": X_test_q,
        "X_train_classical": X_train_c,
        "X_test_classical": X_test_c,
        "y_train": y_train,
        "y_test": y_test,
        "scaler_quantum": scaler_q,
        "scaler_classical": scaler_c,
        "n_samples": n_samples,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "fraud_ratio": y_train.mean(),
    }


def get_feature_cache_path():
    """Return path to cached feature selection results."""
    return os.path.join(DATA_DIR, "selected_features.npy")


def get_or_compute_features(df, n_features=TOP_N_FEATURES):
    """
    Get selected features from cache or compute them.

    Feature selection is expensive, so we cache the results.
    """
    cache_path = get_feature_cache_path()
    if os.path.exists(cache_path):
        features = list(np.load(cache_path, allow_pickle=True))
        print(f"✅ Loaded cached features: {features}")
        return features

    features = select_top_features(df, n_features)
    np.save(cache_path, np.array(features))
    return features
