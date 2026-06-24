"""
Imbalance Ratio Experiment.

Secondary experiment testing how quantum vs classical models
handle different levels of class imbalance WITHOUT resampling.

Hypothesis: Quantum models maintain better recall for the minority class
at extreme imbalance ratios compared to classical models,
without needing SMOTE or other resampling techniques.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    IMBALANCE_RATIOS, IMBALANCE_FIXED_N,
    N_REPETITIONS, QUICK_N_REPETITIONS,
    TABLES_DIR, FIGURES_DIR,
)
from experiments.experiment_runner import (
    run_repeated_experiment, aggregate_and_report, save_results_csv,
)
from src.visualization import plot_metric_comparison


def run_imbalance_experiment(df, selected_features, quick=False):
    """
    Run the imbalance ratio experiment.

    Fixed N, varying fraud ratio to test robustness to imbalance.

    Args:
        df: Full dataframe.
        selected_features: List of feature names.
        quick: If True, fewer repetitions.

    Returns:
        dict: {fraud_ratio: {model_name: aggregated_metrics}}
    """
    n_reps = QUICK_N_REPETITIONS if quick else N_REPETITIONS
    ratios = IMBALANCE_RATIOS if not quick else [0.05, 0.20, 0.50]
    fixed_n = IMBALANCE_FIXED_N

    print("\n" + "=" * 70)
    print("  EXPERIMENT: IMBALANCE RATIO SENSITIVITY")
    print(f"  Fixed N: {fixed_n}")
    print(f"  Fraud ratios: {ratios}")
    print(f"  Repetitions: {n_reps}")
    print("=" * 70)

    all_ratio_results = {}
    all_comparisons = []

    for ratio in ratios:
        print(f"\n{'#'*60}")
        print(f"  FRAUD RATIO: {ratio*100:.0f}%")
        print(f"{'#'*60}")

        metrics, times = run_repeated_experiment(
            df, selected_features, fixed_n,
            n_reps=n_reps, fraud_ratio=ratio,
            quantum=True, classical=True,
        )

        aggregated, comparisons, time_agg = aggregate_and_report(
            metrics, times, label=f"ratio={ratio}"
        )

        all_ratio_results[ratio] = aggregated

        for c in comparisons:
            c["fraud_ratio"] = ratio
        all_comparisons.extend(comparisons)

        save_results_csv(aggregated, label=f"imbalance_ratio_{int(ratio*100)}pct")

    # ---- Generate Visualizations ----
    print("\n📊 Generating imbalance experiment visualizations...")

    # Comparison at extreme imbalance (1% fraud)
    extreme_ratio = ratios[0]
    if extreme_ratio in all_ratio_results:
        for metric in ["recall", "f1_macro", "mcc"]:
            plot_metric_comparison(
                all_ratio_results[extreme_ratio], metric=metric,
                title=f"Extreme Imbalance ({extreme_ratio*100:.0f}% fraud): {metric}",
                filename=f"imbalance_{int(extreme_ratio*100)}pct_{metric}.png",
            )

    # ---- Save Results ----
    rows = []
    for ratio, model_results in all_ratio_results.items():
        for model_name, agg in model_results.items():
            row = {"fraud_ratio": ratio, "model": model_name}
            for metric, vals in agg.items():
                if isinstance(vals, dict) and "mean" in vals:
                    row[f"{metric}_mean"] = vals["mean"]
                    row[f"{metric}_std"] = vals["std"]
            rows.append(row)

    df_results = pd.DataFrame(rows)
    path = os.path.join(TABLES_DIR, "imbalance_all_results.csv")
    df_results.to_csv(path, index=False)
    print(f"💾 Saved: {path}")

    if all_comparisons:
        df_comps = pd.DataFrame(all_comparisons)
        path = os.path.join(TABLES_DIR, "imbalance_comparisons.csv")
        df_comps.to_csv(path, index=False)
        print(f"💾 Saved: {path}")

    # Summary
    print("\n" + "=" * 70)
    print("  🏆 IMBALANCE EXPERIMENT SUMMARY")
    print("=" * 70)

    for ratio in ratios:
        sig_wins = [c for c in all_comparisons
                    if c.get("fraud_ratio") == ratio
                    and c["significant"]
                    and c["difference"] > 0]
        status = "✅" if sig_wins else "❌"
        print(f"\n  {status} Fraud ratio = {ratio*100:.0f}%: "
              f"{len(sig_wins)} significant quantum advantages")
        for w in sig_wins:
            print(f"     • {w['quantum_model']} > {w['classical_model']} "
                  f"on {w['metric']}: Δ={w['difference']:+.4f}")

    return all_ratio_results, all_comparisons
