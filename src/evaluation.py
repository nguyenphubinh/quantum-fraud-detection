"""
Evaluation framework for Quantum ML vs Classical ML comparison.

Provides:
- Comprehensive metric computation (AUC, F1, Recall, Precision, MCC, PR-AUC)
- Statistical significance testing (paired t-test, Wilcoxon signed-rank)
- Bootstrap confidence intervals
- Result aggregation across multiple runs
"""
import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, classification_report,
)


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute a comprehensive set of classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for positive class (optional).

    Returns:
        dict: Metric name -> value.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = 0.0
        try:
            metrics["pr_auc"] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics["pr_auc"] = 0.0

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["true_positives"] = int(tp)
    metrics["false_positives"] = int(fp)
    metrics["true_negatives"] = int(tn)
    metrics["false_negatives"] = int(fn)
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0

    return metrics


def aggregate_results(results_list):
    """
    Aggregate metrics from multiple runs.

    Args:
        results_list: List of metric dicts from compute_metrics().

    Returns:
        dict: {metric_name: {"mean": ..., "std": ..., "values": [...]}}
    """
    if not results_list:
        return {}

    metric_names = [k for k in results_list[0].keys()
                    if isinstance(results_list[0][k], (int, float, np.floating))]

    aggregated = {}
    for metric in metric_names:
        values = [r[metric] for r in results_list if metric in r]
        aggregated[metric] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "values": values,
        }

    return aggregated


def statistical_test(values_a, values_b, test="wilcoxon"):
    """
    Test whether model A is statistically significantly better than model B.

    Args:
        values_a: Metric values from model A (e.g., across CV folds).
        values_b: Metric values from model B.
        test: 'wilcoxon' (non-parametric) or 'ttest' (parametric).

    Returns:
        dict with statistic, p_value, and significance interpretation.
    """
    values_a = np.array(values_a)
    values_b = np.array(values_b)

    # Check if differences exist
    if np.all(values_a == values_b):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "interpretation": "No difference (identical values)",
        }

    if test == "wilcoxon":
        try:
            stat, p_value = stats.wilcoxon(values_a, values_b, alternative="greater")
        except ValueError:
            # Wilcoxon requires at least 10 samples ideally; fall back to ttest
            stat, p_value = stats.ttest_rel(values_a, values_b, alternative="greater")
    elif test == "ttest":
        stat, p_value = stats.ttest_rel(values_a, values_b, alternative="greater")
    else:
        raise ValueError(f"Unknown test: {test}")

    significant = p_value < 0.05

    if significant:
        if p_value < 0.001:
            interp = "Highly significant (p < 0.001)"
        elif p_value < 0.01:
            interp = "Very significant (p < 0.01)"
        else:
            interp = "Significant (p < 0.05)"
    else:
        interp = "Not significant (p >= 0.05)"

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": significant,
        "interpretation": interp,
    }


def bootstrap_confidence_interval(values, confidence=0.95, n_bootstrap=1000):
    """
    Compute bootstrap confidence interval for the mean.

    Args:
        values: Array of metric values.
        confidence: Confidence level.
        n_bootstrap: Number of bootstrap samples.

    Returns:
        tuple: (lower_bound, mean, upper_bound).
    """
    values = np.array(values)
    rng = np.random.RandomState(42)

    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    alpha = (1 - confidence) / 2
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1 - alpha) * 100)

    return lower, np.mean(values), upper


def compare_models(quantum_results, classical_results, metric="f1_macro"):
    """
    Compare quantum and classical model results with statistical testing.

    Args:
        quantum_results: Dict {model_name: aggregated_results} for quantum models.
        classical_results: Dict {model_name: aggregated_results} for classical models.
        metric: Metric to compare on.

    Returns:
        list of dicts with comparison results.
    """
    comparisons = []

    for q_name, q_res in quantum_results.items():
        if metric not in q_res:
            continue
        q_values = q_res[metric]["values"]

        for c_name, c_res in classical_results.items():
            if metric not in c_res:
                continue
            c_values = c_res[metric]["values"]

            # Ensure same number of values
            min_len = min(len(q_values), len(c_values))
            q_v = q_values[:min_len]
            c_v = c_values[:min_len]

            if min_len < 2:
                continue

            test_result = statistical_test(q_v, c_v, test="ttest")

            comparisons.append({
                "quantum_model": q_name,
                "classical_model": c_name,
                "metric": metric,
                "quantum_mean": np.mean(q_v),
                "quantum_std": np.std(q_v),
                "classical_mean": np.mean(c_v),
                "classical_std": np.std(c_v),
                "difference": np.mean(q_v) - np.mean(c_v),
                "p_value": test_result["p_value"],
                "significant": test_result["significant"],
                "interpretation": test_result["interpretation"],
            })

    return comparisons


def format_results_table(results_dict, metrics=None):
    """
    Format results as a printable table string.

    Args:
        results_dict: {model_name: aggregated_results}
        metrics: List of metrics to include (default: main metrics).

    Returns:
        str: Formatted table.
    """
    if metrics is None:
        metrics = ["roc_auc", "f1_macro", "recall", "precision", "mcc", "pr_auc"]

    # Header
    header = f"{'Model':<18}"
    for m in metrics:
        header += f" | {m:<14}"
    line = "-" * len(header)

    rows = [header, line]

    for model_name, agg in sorted(results_dict.items()):
        row = f"{model_name:<18}"
        for m in metrics:
            if m in agg:
                mean = agg[m]["mean"]
                std = agg[m]["std"]
                row += f" | {mean:.4f}±{std:.4f}"
            else:
                row += f" | {'N/A':>14}"
        rows.append(row)

    return "\n".join(rows)
