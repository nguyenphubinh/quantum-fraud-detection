"""
Main experiment runner that orchestrates Quantum vs Classical ML benchmarks.

Handles:
- Running all models on a given dataset split
- Collecting metrics and training times
- Aggregating results across repetitions
- Saving results to CSV and generating visualizations
"""
import os
import sys
import time
import json
import traceback
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RANDOM_SEED, N_REPETITIONS, TABLES_DIR, LOGS_DIR,
    QUANTUM_MODELS_LIST, CLASSICAL_MODELS_LIST,
)
from src.quantum_models import get_quantum_models
from src.classical_models import get_classical_models
from src.evaluation import compute_metrics, aggregate_results, compare_models
from src.visualization import (
    plot_metric_comparison, plot_multi_metric_comparison,
    plot_training_time_comparison, plot_significance_heatmap,
    plot_confusion_matrices,
)


def run_single_experiment(data_dict, quantum=True, classical=True):
    """
    Run all models on a single train/test split.

    Args:
        data_dict: Output from prepare_experiment_data().
        quantum: Whether to run quantum models.
        classical: Whether to run classical models.

    Returns:
        dict: {model_name: {"metrics": {...}, "train_time": float,
               "y_pred": array, "y_prob": array}}
    """
    results = {}

    X_train_q = data_dict["X_train_quantum"]
    X_test_q = data_dict["X_test_quantum"]
    X_train_c = data_dict["X_train_classical"]
    X_test_c = data_dict["X_test_classical"]
    y_train = data_dict["y_train"]
    y_test = data_dict["y_test"]

    # ---- Quantum Models ----
    if quantum:
        q_models = get_quantum_models()
        for name, model in q_models.items():
            print(f"\n🟣 Training {name}...")
            try:
                start = time.time()
                model.fit(X_train_q, y_train)
                train_time = time.time() - start

                y_pred = model.predict(X_test_q)
                y_prob = None
                try:
                    proba = model.predict_proba(X_test_q)
                    y_prob = proba[:, 1]
                except Exception:
                    pass

                metrics = compute_metrics(y_test, y_pred, y_prob)
                metrics["train_time"] = train_time

                results[name] = {
                    "metrics": metrics,
                    "train_time": train_time,
                    "y_pred": y_pred,
                    "y_prob": y_prob,
                    "y_true": y_test,
                }
                print(f"   ✅ {name}: F1={metrics['f1_macro']:.4f}, "
                      f"AUC={metrics.get('roc_auc', 'N/A')}, "
                      f"Recall={metrics['recall']:.4f}, "
                      f"Time={train_time:.1f}s")
            except Exception as e:
                print(f"   ❌ {name} failed: {e}")
                traceback.print_exc()
                results[name] = {
                    "metrics": {"error": str(e)},
                    "train_time": 0,
                    "y_pred": np.zeros_like(y_test),
                    "y_prob": None,
                    "y_true": y_test,
                }

    # ---- Classical Models ----
    if classical:
        c_models = get_classical_models()
        for name, model in c_models.items():
            print(f"\n🟢 Training {name}...")
            try:
                start = time.time()
                model.fit(X_train_c, y_train)
                train_time = time.time() - start

                y_pred = model.predict(X_test_c)
                y_prob = None
                try:
                    proba = model.predict_proba(X_test_c)
                    y_prob = proba[:, 1]
                except Exception:
                    pass

                metrics = compute_metrics(y_test, y_pred, y_prob)
                metrics["train_time"] = train_time

                results[name] = {
                    "metrics": metrics,
                    "train_time": train_time,
                    "y_pred": y_pred,
                    "y_prob": y_prob,
                    "y_true": y_test,
                }
                print(f"   ✅ {name}: F1={metrics['f1_macro']:.4f}, "
                      f"AUC={metrics.get('roc_auc', 'N/A')}, "
                      f"Recall={metrics['recall']:.4f}, "
                      f"Time={train_time:.1f}s")
            except Exception as e:
                print(f"   ❌ {name} failed: {e}")
                traceback.print_exc()
                results[name] = {
                    "metrics": {"error": str(e)},
                    "train_time": 0,
                    "y_pred": np.zeros_like(y_test),
                    "y_prob": None,
                    "y_true": y_test,
                }

    return results


def run_repeated_experiment(df, selected_features, n_samples,
                            n_reps=N_REPETITIONS, fraud_ratio=0.5,
                            quantum=True, classical=True):
    """
    Run experiment multiple times with different random seeds for robustness.

    Returns:
        dict: {model_name: [list of metric dicts]}
        dict: {model_name: [list of train times]}
    """
    from src.data_preprocessing import prepare_experiment_data

    all_metrics = {}  # model_name -> list of metric dicts
    all_times = {}    # model_name -> list of train times

    for rep in range(n_reps):
        seed = RANDOM_SEED + rep
        print(f"\n{'='*60}")
        print(f"  REPETITION {rep+1}/{n_reps} (seed={seed}, N={n_samples})")
        print(f"{'='*60}")

        data = prepare_experiment_data(
            df, n_samples, selected_features,
            fraud_ratio=fraud_ratio, seed=seed
        )

        results = run_single_experiment(data, quantum=quantum, classical=classical)

        for model_name, res in results.items():
            if model_name not in all_metrics:
                all_metrics[model_name] = []
                all_times[model_name] = []

            if "error" not in res["metrics"]:
                all_metrics[model_name].append(res["metrics"])
                all_times[model_name].append(res["train_time"])

    return all_metrics, all_times


def aggregate_and_report(all_metrics, all_times, label=""):
    """
    Aggregate results and generate comparison report.

    Returns:
        dict: {model_name: aggregated_metrics}
        list: Statistical comparisons
    """
    # Aggregate metrics
    aggregated = {}
    for model_name, metrics_list in all_metrics.items():
        if metrics_list:
            aggregated[model_name] = aggregate_results(metrics_list)

    # Aggregate times
    time_agg = {}
    for model_name, times in all_times.items():
        if times:
            time_agg[model_name] = {
                "mean": np.mean(times),
                "std": np.std(times),
            }

    # Separate quantum and classical results
    q_results = {k: v for k, v in aggregated.items() if k in QUANTUM_MODELS_LIST}
    c_results = {k: v for k, v in aggregated.items() if k in CLASSICAL_MODELS_LIST}

    # Statistical comparisons
    comparisons = []
    for metric in ["roc_auc", "f1_macro", "recall", "precision", "mcc", "pr_auc"]:
        comps = compare_models(q_results, c_results, metric)
        comparisons.extend(comps)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY {label}")
    print(f"{'='*60}")
    from src.evaluation import format_results_table
    print(format_results_table(aggregated))

    # Print significant comparisons
    sig_comps = [c for c in comparisons if c["significant"]]
    if sig_comps:
        print(f"\n✅ SIGNIFICANT QUANTUM ADVANTAGES ({label}):")
        for c in sig_comps:
            print(f"   {c['quantum_model']} > {c['classical_model']} "
                  f"on {c['metric']}: Δ={c['difference']:+.4f} (p={c['p_value']:.4f})")

    return aggregated, comparisons, time_agg


def save_results_csv(aggregated, label="results"):
    """Save aggregated results to CSV."""
    rows = []
    for model_name, agg in aggregated.items():
        row = {"model": model_name, "type": "quantum" if model_name in QUANTUM_MODELS_LIST else "classical"}
        for metric, vals in agg.items():
            if isinstance(vals, dict) and "mean" in vals:
                row[f"{metric}_mean"] = vals["mean"]
                row[f"{metric}_std"] = vals["std"]
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(TABLES_DIR, f"{label}.csv")
    df.to_csv(path, index=False)
    print(f"💾 Saved: {path}")
    return df
