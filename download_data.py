"""
Download the Credit Card Fraud Detection dataset.

This script downloads the dataset from Kaggle. You have two options:
1. Use kagglehub (requires Kaggle API credentials)
2. Manual download: Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   and place creditcard.csv in the data/ directory.
"""
import os
import sys

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, DATASET_PATH


def download_with_kagglehub():
    """Download using kagglehub library."""
    try:
        import kagglehub
        print("[DOWNLOAD] Downloading Credit Card Fraud Detection dataset via kagglehub...")
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        print(f"[OK] Downloaded to: {path}")

        # Copy to our data directory
        import shutil
        src_csv = os.path.join(path, "creditcard.csv")
        if os.path.exists(src_csv):
            shutil.copy2(src_csv, DATASET_PATH)
            print(f"[OK] Copied to: {DATASET_PATH}")
        else:
            # Search for the CSV in subdirectories
            for root, dirs, files in os.walk(path):
                for f in files:
                    if f == "creditcard.csv":
                        shutil.copy2(os.path.join(root, f), DATASET_PATH)
                        print(f"[OK] Copied to: {DATASET_PATH}")
                        return True
            print(f"[WARN] creditcard.csv not found in {path}")
            return False
        return True
    except ImportError:
        print("[WARN] kagglehub not installed. Install with: pip install kagglehub")
        return False
    except Exception as e:
        print(f"[WARN] kagglehub download failed: {e}")
        return False


def check_dataset():
    """Check if dataset already exists."""
    if os.path.exists(DATASET_PATH):
        file_size = os.path.getsize(DATASET_PATH) / (1024 * 1024)  # MB
        print(f"[OK] Dataset found: {DATASET_PATH} ({file_size:.1f} MB)")
        return True
    return False


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if check_dataset():
        print("Dataset already downloaded. Skipping.")
        return

    print("=" * 60)
    print("  Credit Card Fraud Detection Dataset Downloader")
    print("=" * 60)

    # Try kagglehub first
    if download_with_kagglehub():
        if check_dataset():
            return

    # Manual instructions
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS:")
    print("=" * 60)
    print(f"""
1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Click 'Download' button
3. Extract the ZIP file
4. Place 'creditcard.csv' in: {DATA_DIR}

Alternative: Use Kaggle CLI:
  kaggle datasets download -d mlg-ulb/creditcardfraud -p {DATA_DIR} --unzip

After placing the file, run this script again to verify.
""")


if __name__ == "__main__":
    main()
