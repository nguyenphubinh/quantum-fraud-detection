"""
Visualization module for Quantum ML vs Classical ML benchmark.

Generates publication-quality charts:
- Model comparison bar charts
- Learning curves (performance vs sample size)
- ROC and PR curves
- Confusion matrices
- Statistical significance heatmaps
- Training time comparison
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FIGURES_DIR, MODEL_COLORS, QUANTUM_MODELS_LIST, CLASSICAL_MODELS_LIST,
    QUANTUM_COLOR, CLASSICAL_COLOR,
)

# Set global style
plt.rcParams.update({
    "figure.figsize": (12, 8),
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})
sns.set_style("whitegrid")


def _get_color(model_name):
    """Get color for a model, with fallback."""
    return MODEL_COLORS.get(model_name, "#64748B")


def _is_quantum(model_name):
    """Check if model is quantum-type."""
    return model_name in QUANTUM_MODELS_LIST


def plot_metric_comparison(results_dict, metric="f1_macro",
                           title=None, filename=None):
    """
    Bar chart comparing all models on a single metric.

    Quantum models shown in purple tones, classical in green tones.
    Error bars show ±1 std from multiple runs.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    models = sorted(results_dict.keys(),
                    key=lambda m: results_dict[m].get(metric, {}).get("mean", 0),
                    reverse=True)

    means = []
    stds = []
    colors = []
    edge_colors = []

    for m in models:
        agg = results_dict[m].get(metric, {})
        means.append(agg.get("mean", 0))
        stds.append(agg.get("std", 0))
        c = _get_color(m)
        colors.append(c)
        edge_colors.append("gold" if _is_quantum(m) else "#333")

    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=colors, edgecolor=edge_colors, linewidth=2,
                  alpha=0.85)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005,
                f"{mean:.3f}", ha="center", va="bottom", fontweight="bold",
                fontsize=10)

    # Mark quantum models with a star
    for i, m in enumerate(models):
        if _is_quantum(m):
            ax.text(i, -0.02, "★ QUANTUM", ha="center", fontsize=8,
                    color=QUANTUM_COLOR, fontweight="bold",
                    transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"Model Comparison: {metric.replace('_', ' ').title()}")
    ax.set_ylim(0, min(1.15, max(means) + max(stds) + 0.1))

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=QUANTUM_COLOR, edgecolor="gold", linewidth=2,
              label="Quantum Models"),
        Patch(facecolor=CLASSICAL_COLOR, edgecolor="#333", linewidth=2,
              label="Classical Models"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    if filename:
        path = os.path.join(FIGURES_DIR, filename)
        plt.savefig(path)
        print(f"📊 Saved: {path}")
    plt.close()
    return fig


def plot_sample_size_curves(sample_size_results, metric="f1_macro",
                            title=None, filename=None):
    """
    Line chart showing how each model's performance varies with sample size.

    This is THE key chart for demonstrating quantum advantage at small N.

    Args:
        sample_size_results: Dict {sample_size: {model_name: aggregated_metrics}}
        metric: Metric to plot.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Collect all model names
    all_models = set()
    for sz_results in sample_size_results.values():
        all_models.update(sz_results.keys())

    sample_sizes = sorted(sample_size_results.keys())

    for model_name in sorted(all_models):
        means = []
        stds = []
        valid_sizes = []

        for sz in sample_sizes:
            if model_name in sample_size_results[sz]:
                agg = sample_size_results[sz][model_name]
                if metric in agg:
                    means.append(agg[metric]["mean"])
                    stds.append(agg[metric]["std"])
                    valid_sizes.append(sz)

        if not valid_sizes:
            continue

        means = np.array(means)
        stds = np.array(stds)
        color = _get_color(model_name)
        is_q = _is_quantum(model_name)

        ax.plot(valid_sizes, means, "o-" if is_q else "s--",
                color=color, label=model_name,
                linewidth=3 if is_q else 1.5,
                markersize=10 if is_q else 6,
                alpha=1.0 if is_q else 0.7,
                zorder=10 if is_q else 5)

        ax.fill_between(valid_sizes, means - stds, means + stds,
                        alpha=0.15, color=color)

    ax.set_xlabel("Sample Size (N)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"Performance vs Sample Size: {metric.replace('_', ' ').title()}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Add annotation for quantum advantage zone
    ax.axvspan(sample_sizes[0], min(500, max(sample_sizes)),
               alpha=0.05, color=QUANTUM_COLOR,
               label="_Quantum Advantage Zone")
    ax.text(min(300, max(sample_sizes) * 0.3), ax.get_ylim()[0] + 0.02,
            "← Quantum Advantage Zone",
            fontsize=10, color=QUANTUM_COLOR, fontstyle="italic")

    plt.tight_layout()
    if filename:
        path = os.path.join(FIGURES_DIR, filename)
        plt.savefig(path)
        print(f"📊 Saved: {path}")
    plt.close()
    return fig


def plot_multi_metric_comparison(results_dict, metrics=None,
                                  title=None, filename=None):
    """
    Grouped bar chart comparing models across multiple metrics simultaneously.
    """
    if metrics is None:
        metrics = ["roc_auc", "f1_macro", "recall", "precision", "mcc"]

    models = sorted(results_dict.keys())
    n_models = len(models)
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=(16, 8))

    bar_width = 0.8 / n_metrics
    x = np.arange(n_models)

    for i, metric in enumerate(metrics):
        values = []
        errors = []
        for m in models:
            agg = results_dict[m].get(metric, {})
            values.append(agg.get("mean", 0))
            errors.append(agg.get("std", 0))

        offset = (i - n_metrics / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width * 0.9,
                      yerr=errors, capsize=3,
                      label=metric.replace("_", " ").title(),
                      alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title or "Multi-Metric Model Comparison")
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylim(0, 1.15)

    # Mark quantum models
    for i, m in enumerate(models):
        if _is_quantum(m):
            ax.axvspan(i - 0.45, i + 0.45, alpha=0.05, color=QUANTUM_COLOR)

    plt.tight_layout()
    if filename:
        path = os.path.join(FIGURES_DIR, filename)
        plt.savefig(path)
        print(f"📊 Saved: {path}")
    plt.close()
    return fig


def plot_training_time_comparison(time_dict, title=None, filename=None):
    """
    Bar chart comparing training times between quantum and classical models.

    Args:
        time_dict: {model_name: {"mean": ..., "std": ...}}
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    models = sorted(time_dict.keys())
    times = [time_dict[m]["mean"] for m in models]
    stds = [time_dict[m].get("std", 0) for m in models]
    colors = [_get_color(m) for m in models]

    bars = ax.bar(range(len(models)), times, yerr=stds, capsize=5,
                  color=colors, alpha=0.85, edgecolor="#333")

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{t:.1f}s", ha="center", va="bottom", fontweight="bold")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Training Time (seconds)")
    ax.set_title(title or "Training Time Comparison")
    ax.set_yscale("log")

    plt.tight_layout()
    if filename:
        path = os.path.join(FIGURES_DIR, filename)
        plt.savefig(path)
        print(f"📊 Saved: {path}")
    plt.close()
    return fig


def plot_significance_heatmap(comparisons, metric="f1_macro",
                               title=None, filename=None):
    """
    Heatmap showing p-values from statistical tests
    (quantum model in rows, classical model in columns).
    Green = quantum significantly better, red = not significant.
    """
    # Filter comparisons for the specified metric
    filtered = [c for c in comparisons if c["metric"] == metric]
    if not filtered:
        print(f"⚠️  No comparisons found for metric: {metric}")
        return None

    q_models = sorted(set(c["quantum_model"] for c in filtered))
    c_models = sorted(set(c["classical_model"] for c in filtered))

    # Build p-value matrix
    p_matrix = np.ones((len(q_models), len(c_models)))
    diff_matrix = np.zeros((len(q_models), len(c_models)))

    for comp in filtered:
        i = q_models.index(comp["quantum_model"])
        j = c_models.index(comp["classical_model"])
        p_matrix[i, j] = comp["p_value"]
        diff_matrix[i, j] = comp["difference"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # P-value heatmap
    sns.heatmap(p_matrix, annot=True, fmt=".3f",
                xticklabels=c_models, yticklabels=q_models,
                cmap="RdYlGn_r", vmin=0, vmax=0.1,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title(f"P-values (Quantum > Classical)\n{metric}")
    axes[0].set_xlabel("Classical Model")
    axes[0].set_ylabel("Quantum Model")

    # Performance difference heatmap
    sns.heatmap(diff_matrix, annot=True, fmt=".4f",
                xticklabels=c_models, yticklabels=q_models,
                cmap="RdYlGn", center=0,
                ax=axes[1], linewidths=0.5)
    axes[1].set_title(f"Quantum - Classical Difference\n{metric}")
    axes[1].set_xlabel("Classical Model")
    axes[1].set_ylabel("Quantum Model")

    plt.suptitle(title or f"Statistical Significance Analysis: {metric}", y=1.02)
    plt.tight_layout()

    if filename:
        path = os.path.join(FIGURES_DIR, filename)
        plt.savefig(path)
        print(f"📊 Saved: {path}")
    plt.close()
    return fig


def plot_confusion_matrices(confusion_data, title=None, filename=None):
    """
    Plot confusion matrices for multiple models side by side.

    Args:
        confusion_data: Dict {model_name: {"y_true": ..., "y_pred": ...}}
    """
    models = sorted(confusion_data.keys())
    n_models = len(models)
    n_cols = min(4, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_models > 1 else axes

    for idx, model_name in enumerate(models):
        ax = axes[idx] if n_models > 1 else axes[0]
        data = confusion_data[model_name]

        from sklearn.metrics import confusion_matrix as cm_func
        cm = cm_func(data["y_true"], data["y_pred"], labels=[0, 1])

        color = _get_color(model_name)
        cmap = sns.light_palette(color, as_cmap=True)

        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                    xticklabels=["Legit", "Fraud"],
                    yticklabels=["Legit", "Fraud"],
                    linewidths=0.5)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        prefix = "★ " if _is_quantum(model_name) else ""
        ax.set_title(f"{prefix}{model_name}")

    # Hide unused axes
    for idx in range(n_models, len(axes) if isinstance(axes, np.ndarray) else 1):
        if isinstance(axes, np.ndarray):
            axes[idx].set_visible(False)

    plt.suptitle(title or "Confusion Matrices", fontsize=14, y=1.02)
    plt.tight_layout()

    if filename:
        path = os.path.join(FIGURES_DIR, filename)
        plt.savefig(path)
        print(f"📊 Saved: {path}")
    plt.close()
    return fig


def generate_summary_report(all_results, sample_size_results=None,
                             comparisons=None, filename="summary_report.txt"):
    """
    Generate a text summary report of all experimental results.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("  QUANTUM ML vs CLASSICAL ML — EXPERIMENT REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Overall results
    if all_results:
        lines.append("📊 OVERALL MODEL PERFORMANCE:")
        lines.append("-" * 70)
        for model_name in sorted(all_results.keys()):
            agg = all_results[model_name]
            is_q = "🟣 QUANTUM" if _is_quantum(model_name) else "🟢 Classical"
            lines.append(f"\n  {is_q}: {model_name}")
            for metric in ["roc_auc", "f1_macro", "recall", "precision", "mcc"]:
                if metric in agg:
                    m = agg[metric]
                    lines.append(f"    {metric:<14}: {m['mean']:.4f} ± {m['std']:.4f}")

    # Statistical comparisons
    if comparisons:
        lines.append("\n" + "=" * 70)
        lines.append("📈 STATISTICAL SIGNIFICANCE (Quantum vs Classical):")
        lines.append("-" * 70)
        for comp in comparisons:
            symbol = "✅" if comp["significant"] else "❌"
            lines.append(
                f"  {symbol} {comp['quantum_model']} vs {comp['classical_model']} "
                f"({comp['metric']}): "
                f"Δ={comp['difference']:+.4f}, p={comp['p_value']:.4f} "
                f"— {comp['interpretation']}"
            )

    # Key findings
    lines.append("\n" + "=" * 70)
    lines.append("🏆 KEY FINDINGS:")
    lines.append("-" * 70)

    if comparisons:
        significant_wins = [c for c in comparisons
                           if c["significant"] and c["difference"] > 0]
        if significant_wins:
            lines.append(f"  ✅ Quantum models showed SIGNIFICANT advantage "
                        f"in {len(significant_wins)} comparisons")
            for w in significant_wins:
                lines.append(f"     • {w['quantum_model']} > {w['classical_model']} "
                            f"on {w['metric']} (Δ={w['difference']:+.4f}, p={w['p_value']:.4f})")
        else:
            lines.append("  ⚠️  No statistically significant quantum advantage found")

    report = "\n".join(lines)

    # Save report
    from config import RESULTS_DIR
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"📄 Report saved: {path}")

    return report
