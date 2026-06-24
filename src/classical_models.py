"""
Classical ML baseline models for fraud detection.

Provides a suite of classical models for fair comparison with quantum models:
- SVM with RBF kernel (direct comparison with QSVM)
- SVM with Polynomial kernel
- Random Forest
- XGBoost
- LightGBM
- MLP Neural Network (direct comparison with Hybrid QNN)

All models are wrapped with consistent sklearn interface and support
hyperparameter tuning via GridSearchCV.
"""
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RANDOM_SEED, CV_FOLDS,
    SVM_RBF_PARAMS, SVM_POLY_PARAMS, RF_PARAMS,
    XGB_PARAMS, LGBM_PARAMS, MLP_PARAMS,
)


class TunedClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper that performs GridSearchCV on initialization parameters,
    then exposes the best model with a consistent interface.
    """

    def __init__(self, base_estimator, param_grid, name="Model",
                 cv=CV_FOLDS, scoring="f1"):
        self.base_estimator = base_estimator
        self.param_grid = param_grid
        self.name = name
        self.cv = cv
        self.scoring = scoring
        self.best_model_ = None
        self.best_params_ = None
        self.train_time_ = 0

    def fit(self, X, y):
        """Fit with hyperparameter tuning via GridSearchCV."""
        start_time = time.time()

        # For very small datasets, reduce CV folds
        n_samples = len(X)
        actual_cv = min(self.cv, max(2, n_samples // 10))

        # Check if we have enough samples for both classes in each fold
        min_class_count = min(np.sum(y == 0), np.sum(y == 1))
        if min_class_count < actual_cv:
            actual_cv = max(2, min_class_count)

        grid = GridSearchCV(
            self.base_estimator,
            self.param_grid,
            cv=actual_cv,
            scoring=self.scoring,
            n_jobs=-1,
            refit=True,
            error_score=0.0,  # Return 0 for failed fits instead of raising
        )
        grid.fit(X, y)

        self.best_model_ = grid.best_estimator_
        self.best_params_ = grid.best_params_
        self.train_time_ = time.time() - start_time

        print(f"   {self.name} best params: {self.best_params_} "
              f"(CV score: {grid.best_score_:.4f}, time: {self.train_time_:.1f}s)")
        return self

    def predict(self, X):
        return self.best_model_.predict(X)

    def predict_proba(self, X):
        if hasattr(self.best_model_, "predict_proba"):
            return self.best_model_.predict_proba(X)
        else:
            # For SVM without probability=True, use decision function
            decision = self.best_model_.decision_function(X)
            prob_pos = 1 / (1 + np.exp(-decision))
            return np.column_stack([1 - prob_pos, prob_pos])


def _get_xgb_estimator():
    """Lazy import of XGBoost to avoid mandatory dependency."""
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
        )
    except ImportError:
        print("⚠️  XGBoost not installed. Skipping XGBoost model.")
        return None


def _get_lgbm_estimator():
    """Lazy import of LightGBM to avoid mandatory dependency."""
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            random_state=RANDOM_SEED,
            verbose=-1,
        )
    except ImportError:
        print("⚠️  LightGBM not installed. Skipping LightGBM model.")
        return None


def get_classical_models():
    """
    Return a dictionary of classical models ready for training.

    Each model is wrapped in TunedClassifier for automatic
    hyperparameter optimization.

    Returns:
        dict: {model_name: TunedClassifier_instance}
    """
    models = {}

    # SVM with RBF kernel - direct comparison with QSVM
    models["SVM_RBF"] = TunedClassifier(
        base_estimator=SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_SEED,
        ),
        param_grid=SVM_RBF_PARAMS,
        name="SVM_RBF",
    )

    # SVM with Polynomial kernel
    models["SVM_Poly"] = TunedClassifier(
        base_estimator=SVC(
            kernel="poly",
            probability=True,
            random_state=RANDOM_SEED,
        ),
        param_grid=SVM_POLY_PARAMS,
        name="SVM_Poly",
    )

    # Random Forest
    models["Random_Forest"] = TunedClassifier(
        base_estimator=RandomForestClassifier(
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        param_grid=RF_PARAMS,
        name="Random_Forest",
    )

    # XGBoost
    xgb_est = _get_xgb_estimator()
    if xgb_est is not None:
        models["XGBoost"] = TunedClassifier(
            base_estimator=xgb_est,
            param_grid=XGB_PARAMS,
            name="XGBoost",
        )

    # LightGBM
    lgbm_est = _get_lgbm_estimator()
    if lgbm_est is not None:
        models["LightGBM"] = TunedClassifier(
            base_estimator=lgbm_est,
            param_grid=LGBM_PARAMS,
            name="LightGBM",
        )

    # MLP Neural Network - comparison with Hybrid QNN
    models["MLP"] = TunedClassifier(
        base_estimator=MLPClassifier(
            max_iter=500,
            random_state=RANDOM_SEED,
            early_stopping=True,
            validation_fraction=0.15,
        ),
        param_grid=MLP_PARAMS,
        name="MLP",
    )

    return models
