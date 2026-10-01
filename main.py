
from pathlib import Path
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parent

PREPROCESSING_SCRIPT = BASE_DIR / "src" / "preprocessing.py"
MODEL_SCRIPT = BASE_DIR / "src" / "model_training.py"


def run_script(script_path):
    """Run a Python script and stop if it fails."""
    print("\n" + "=" * 60)
    print(f"Running: {script_path.name}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR
    )

    if result.returncode != 0:
        raise SystemExit(
            f"\nError: {script_path.name} failed. "
            "Check the error message above."
        )


def main():
    print("=" * 60)
    print("CUSTOMER LIFETIME VALUE PREDICTION")
    print("E-Commerce Machine Learning Project")
    print("=" * 60)

    run_script(PREPROCESSING_SCRIPT)
    run_script(MODEL_SCRIPT)

    print("\n" + "=" * 60)
    print("PROJECT PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print("\nGenerated outputs:")
    print("- results/processed_customer_data.csv")
    print("- results/model_comparison.csv")
    print("- results/predictions.csv")
    print("- models/linear_regression.pkl")
    print("- models/decision_tree.pkl")
    print("- visualizations/linear_regression_actual_vs_predicted.png")
    print("- visualizations/decision_tree.png")


if __name__ == "__main__":
    main()