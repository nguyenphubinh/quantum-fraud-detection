"""
Quantum ML vs Classical ML — Full Pipeline Runner

Entry point for the entire benchmark project.
Downloads data, preprocesses, trains all models, evaluates,
and generates comparison visualizations.

Usage:
    python run_all.py --quick-test     # Fast smoke test (~10 min)
    python run_all.py --full           # Full experiment (~4-8 hours)
    python run_all.py --sample-size    # Only sample size experiment
    python run_all.py --imbalance      # Only imbalance experiment
"""
import argparse
import os
import sys
import time

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATASET_PATH, RESULTS_DIR, FIGURES_DIR, TABLES_DIR
import config

# Auto-detect Kaggle/GPU environment to switch device to lightning.gpu
try:
    import pennylane as qml
    # Check if lightning.gpu is available
    if "lightning.gpu" in qml.devices.Device.capabilities():
        config.QML_DEVICE = "lightning.gpu"
        print("[GPU] Detected GPU-enabled device capability. Using 'lightning.gpu' for simulation!")
    else:
        # Fallback to lightning.qubit if available, else default.qubit
        config.QML_DEVICE = "lightning.qubit"
        print("[CPU] GPU backend not detected. Using 'lightning.qubit' for faster CPU simulation.")
except Exception as e:
    print(f"[DEVICE] Defaulting to default.qubit: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Quantum ML vs Classical ML Benchmark Pipeline"
    )
    parser.add_argument(
        "--quick-test", action="store_true",
        help="Run a quick smoke test with small sample sizes and few repetitions"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run the full experiment suite (takes several hours)"
    )
    parser.add_argument(
        "--sample-size", action="store_true",
        help="Run only the sample size scaling experiment"
    )
    parser.add_argument(
        "--imbalance", action="store_true",
        help="Run only the imbalance ratio experiment"
    )
    args = parser.parse_args()

    # Default to quick-test if no mode specified
    if not any([args.quick_test, args.full, args.sample_size, args.imbalance]):
        args.quick_test = True

    quick = args.quick_test
    run_sample_size = args.full or args.sample_size or args.quick_test
    run_imbalance = args.full or args.imbalance

    print("=" * 70)
    print("  [QML] QUANTUM ML vs CLASSICAL ML BENCHMARK")
    print("  Proving Quantum Advantage in Financial Fraud Detection")
    print("=" * 70)
    mode = "QUICK TEST" if quick else "FULL EXPERIMENT"
    print(f"  Mode: {mode}")
    print(f"  Results: {RESULTS_DIR}")
    print("=" * 70)

    total_start = time.time()

    # ---- Step 1: Check/Download Dataset ----
    print("\n[Step 1] Checking dataset...")
    if not os.path.exists(DATASET_PATH):
        print("Dataset not found. Attempting download...")
        from download_data import main as download_main
        download_main()
        if not os.path.exists(DATASET_PATH):
            print("\n[ERROR] Dataset not found!")
            print(f"   Please download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
            print(f"   And place 'creditcard.csv' in: {os.path.dirname(DATASET_PATH)}")
            sys.exit(1)
    print(f"[OK] Dataset found at: {DATASET_PATH}")

    # ---- Step 2: Load and Preprocess ----
    print("\n[Step 2] Loading and preprocessing data...")
    from src.data_preprocessing import load_dataset, get_or_compute_features

    df = load_dataset()
    selected_features = get_or_compute_features(df)

    print(f"[OK] Using features: {selected_features}")

    # ---- Step 3: Run Experiments ----
    all_results = {}

    if run_sample_size:
        print("\n[Step 3a] Running Sample Size Scaling Experiment...")
        from experiments.exp_sample_size import run_sample_size_experiment
        size_results, size_comparisons = run_sample_size_experiment(
            df, selected_features, quick=quick
        )
        all_results["sample_size"] = {
            "results": size_results,
            "comparisons": size_comparisons,
        }

    if run_imbalance:
        print("\n[Step 3b] Running Imbalance Ratio Experiment...")
        from experiments.exp_imbalance_ratio import run_imbalance_experiment
        imb_results, imb_comparisons = run_imbalance_experiment(
            df, selected_features, quick=quick
        )
        all_results["imbalance"] = {
            "results": imb_results,
            "comparisons": imb_comparisons,
        }

    # ---- Step 4: Generate Summary Report ----
    print("\n[Step 4] Generating summary report...")
    from src.visualization import generate_summary_report

    # Collect all aggregated results for the report
    if "sample_size" in all_results:
        # Use smallest sample size results for the main report
        smallest = min(all_results["sample_size"]["results"].keys())
        report_results = all_results["sample_size"]["results"][smallest]
        report_comparisons = [
            c for c in all_results["sample_size"]["comparisons"]
            if c.get("sample_size") == smallest
        ]
        report = generate_summary_report(
            report_results,
            comparisons=report_comparisons,
            filename="summary_report.txt",
        )
        print(report)

    # ---- Done ----
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  [DONE] ALL EXPERIMENTS COMPLETE!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Tables:  {TABLES_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
