"""
Sample Size Scaling Experiment.

The KEY experiment for demonstrating quantum advantage:
Tests how model performance varies with training set size.

Hypothesis: Quantum models (QSVM, VQC) outperform classical models
when sample size is small (N ≤ 500), while classical models
catch up or surpass as more data becomes available.

This mirrors real-world scenarios like:
- Cold-start fraud detection
- Emerging fraud pattern detection  
- Cross-border transaction analysis with limited data
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SAMPLE_SIZES, QUICK_SAMPLE_SIZES,
    N_REPETITIONS, QUICK_N_REPETITIONS,
    TABLES_DIR, FIGURES_DIR, RANDOM_SEED,
)
from experiments.experiment_runner import (
    run_repeated_experiment, aggregate_and_report, save_results_csv,
)
from src.visualization import (
    plot_sample_size_curves, plot_metric_comparison,
    plot_significance_heatmap, generate_summary_report,
)


def run_sample_size_experiment(df, selected_features, quick=False):
    """
    Run the sample size scaling experiment.

    For each sample size, runs N repetitions and aggregates results.
    Then generates learning curves and comparison charts.

    Args:
        df: Full dataframe.
        selected_features: List of feature names.
        quick: If True, use fewer sample sizes and repetitions.

    Returns:
        dict: {sample_size: {model_name: aggregated_metrics}}
    """
    sample_sizes = QUICK_SAMPLE_SIZES if quick else SAMPLE_SIZES
    n_reps = QUICK_N_REPETITIONS if quick else N_REPETITIONS

    print("\n" + "=" * 70)
    print("  EXPERIMENT: SAMPLE SIZE SCALING")
    print(f"  Sample sizes: {sample_sizes}")
    print(f"  Repetitions per size: {n_reps}")
    print("=" * 70)

    all_size_results = {}   # sample_size -> {model: aggregated}
    all_comparisons = []
    all_time_results = {}   # sample_size -> {model: time_agg}

    for n_samples in sample_sizes:
        print(f"\n{'#'*60}")
        print(f"  SAMPLE SIZE: N = {n_samples}")
        print(f"{'#'*60}")

        metrics, times = run_repeated_experiment(
            df, selected_features, n_samples,
            n_reps=n_reps, fraud_ratio=0.5,
            quantum=True, classical=True,
        )

        aggregated, comparisons, time_agg = aggregate_and_report(
            metrics, times, label=f"N={n_samples}"
        )

        all_size_results[n_samples] = aggregated
        all_time_results[n_samples] = time_agg

        # Tag comparisons with sample size
        for c in comparisons:
            c["sample_size"] = n_samples
        all_comparisons.extend(comparisons)

        # Save intermediate results
        save_results_csv(aggregated, label=f"sample_size_N{n_samples}")

    # ---- Generate Visualizations ----
    print("\n📊 Generating visualizations...")

    # 1. Learning curves (THE key chart)
    for metric in ["roc_auc", "f1_macro", "recall", "mcc", "pr_auc"]:
        plot_sample_size_curves(
            all_size_results, metric=metric,
            title=f"Quantum vs Classical: {metric.replace('_', ' ').title()} by Sample Size",
            filename=f"sample_size_{metric}.png",
        )

    # 2. Comparison at the smallest sample size (quantum advantage zone)
    smallest_n = sample_sizes[0]
    if smallest_n in all_size_results:
        for metric in ["f1_macro", "recall", "roc_auc"]:
            plot_metric_comparison(
                all_size_results[smallest_n], metric=metric,
                title=f"Model Comparison at N={smallest_n} ({metric})",
                filename=f"comparison_N{smallest_n}_{metric}.png",
            )

    # 3. Significance heatmaps for different sample sizes
    for n_samples in sample_sizes:
        size_comps = [c for c in all_comparisons if c.get("sample_size") == n_samples]
        if size_comps:
            plot_significance_heatmap(
                size_comps, metric="f1_macro",
                title=f"Statistical Significance at N={n_samples}",
                filename=f"significance_N{n_samples}.png",
            )

    # ---- Save All Results ----
    _save_comprehensive_results(all_size_results, all_comparisons, all_time_results)

    # ---- Generate Summary Report ----
    # Find the best quantum advantage point
    _print_quantum_advantage_summary(all_size_results, all_comparisons, sample_sizes)

    return all_size_results, all_comparisons


def _save_comprehensive_results(all_size_results, all_comparisons, all_time_results):
    """Save all results to CSV files."""
    # Flat table of all results
    rows = []
    for n_samples, model_results in all_size_results.items():
        for model_name, agg in model_results.items():
            row = {"sample_size": n_samples, "model": model_name}
            for metric, vals in agg.items():
                if isinstance(vals, dict) and "mean" in vals:
                    row[f"{metric}_mean"] = vals["mean"]
                    row[f"{metric}_std"] = vals["std"]
            rows.append(row)

    df_results = pd.DataFrame(rows)
    path = os.path.join(TABLES_DIR, "sample_size_all_results.csv")
    df_results.to_csv(path, index=False)
    print(f"💾 Saved: {path}")

    # Comparisons table
    if all_comparisons:
        df_comps = pd.DataFrame(all_comparisons)
        path = os.path.join(TABLES_DIR, "sample_size_comparisons.csv")
        df_comps.to_csv(path, index=False)
        print(f"💾 Saved: {path}")

    # Time results
    time_rows = []
    for n_samples, time_data in all_time_results.items():
        for model_name, t in time_data.items():
            time_rows.append({
                "sample_size": n_samples,
                "model": model_name,
                "time_mean": t["mean"],
                "time_std": t["std"],
            })
    if time_rows:
        df_times = pd.DataFrame(time_rows)
        path = os.path.join(TABLES_DIR, "sample_size_times.csv")
        df_times.to_csv(path, index=False)
        print(f"💾 Saved: {path}")


def _print_quantum_advantage_summary(all_size_results, all_comparisons, sample_sizes):
    """Print a summary highlighting quantum advantage findings."""
    print("\n" + "=" * 70)
    print("  🏆 QUANTUM ADVANTAGE SUMMARY")
    print("=" * 70)

    for n in sample_sizes:
        sig_wins = [c for c in all_comparisons
                    if c.get("sample_size") == n
                    and c["significant"]
                    and c["difference"] > 0]
        total_comps = [c for c in all_comparisons
                       if c.get("sample_size") == n]

        status = "✅" if sig_wins else "❌"
        print(f"\n  {status} N = {n}: {len(sig_wins)}/{len(total_comps)} "
              f"significant quantum advantages")

        if sig_wins:
            for w in sig_wins:
                print(f"     • {w['quantum_model']} > {w['classical_model']} "
                      f"on {w['metric']}: "
                      f"Δ={w['difference']:+.4f} (p={w['p_value']:.4f})")

    # Overall conclusion
    small_n_wins = [c for c in all_comparisons
                    if c.get("sample_size", 0) <= 500
                    and c["significant"]
                    and c["difference"] > 0]
    large_n_wins = [c for c in all_comparisons
                    if c.get("sample_size", 0) > 500
                    and c["significant"]
                    and c["difference"] > 0]

    print(f"\n  📊 CONCLUSION:")
    print(f"     Quantum wins at small N (≤500): {len(small_n_wins)} significant advantages")
    print(f"     Quantum wins at large N (>500): {len(large_n_wins)} significant advantages")

    if len(small_n_wins) > len(large_n_wins):
        print(f"\n  🎯 HYPOTHESIS CONFIRMED: Quantum ML shows stronger advantage "
              f"at smaller sample sizes!")
    elif small_n_wins:
        print(f"\n  🎯 PARTIAL SUPPORT: Quantum ML shows some advantage at small N")
    else:
        print(f"\n  ⚠️  HYPOTHESIS NOT CONFIRMED in this experiment. "
              f"Consider adjusting quantum circuit architecture or feature encoding.")
