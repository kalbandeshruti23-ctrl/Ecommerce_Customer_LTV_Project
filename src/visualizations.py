from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "results" / "processed_customer_data.csv"
PREDICTIONS_PATH = BASE_DIR / "results" / "predictions.csv"
MODEL_PATH = BASE_DIR / "models" / "decision_tree.pkl"
VIS_DIR = BASE_DIR / "visualizations"

VIS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)
predictions = pd.read_csv(PREDICTIONS_PATH)

print("Creating project visualizations...")


# --------------------------------------------------
# 1. LTV DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.hist(
    df["LTV"],
    bins=30,
    edgecolor="black"
)

plt.title("Customer Lifetime Value Distribution")
plt.xlabel("Customer Lifetime Value")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(
    VIS_DIR / "ltv_distribution.png",
    dpi=200
)

plt.close()


# --------------------------------------------------
# 2. TOTAL SPENDING VS LTV
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    df["Total_Spending"],
    df["LTV"],
    alpha=0.6
)

plt.title("Total Spending vs Customer Lifetime Value")
plt.xlabel("Total Spending")
plt.ylabel("LTV")
plt.tight_layout()

plt.savefig(
    VIS_DIR / "spending_vs_ltv.png",
    dpi=200
)

plt.close()


# --------------------------------------------------
# 3. PURCHASE FREQUENCY VS LTV
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    df["Purchase_Frequency"],
    df["LTV"],
    alpha=0.6
)

plt.title("Purchase Frequency vs Customer Lifetime Value")
plt.xlabel("Purchase Frequency")
plt.ylabel("LTV")
plt.tight_layout()

plt.savefig(
    VIS_DIR / "purchase_frequency_vs_ltv.png",
    dpi=200
)

plt.close()


# --------------------------------------------------
# 4. CUSTOMER SEGMENT DISTRIBUTION
# --------------------------------------------------

segment_order = [
    "Low Value",
    "Medium Value",
    "High Value",
    "Premium"
]

segment_counts = (
    df["Customer_Segment"]
    .value_counts()
    .reindex(segment_order)
)

plt.figure(figsize=(9, 6))

plt.bar(
    segment_counts.index,
    segment_counts.values,
    edgecolor="black"
)

plt.title("Customer Segment Distribution")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    VIS_DIR / "customer_segment_distribution.png",
    dpi=200
)

plt.close()


# --------------------------------------------------
# 5. AVERAGE LTV BY SEGMENT
# --------------------------------------------------

average_ltv = (
    df.groupby("Customer_Segment")["LTV"]
    .mean()
    .reindex(segment_order)
)

plt.figure(figsize=(9, 6))

plt.bar(
    average_ltv.index,
    average_ltv.values,
    edgecolor="black"
)

plt.title("Average LTV by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Average LTV")

plt.tight_layout()

plt.savefig(
    VIS_DIR / "average_ltv_by_segment.png",
    dpi=200
)

plt.close()


# --------------------------------------------------
# 6. RESIDUAL ANALYSIS
# --------------------------------------------------

residuals = (
    predictions["Actual_LTV"]
    - predictions["Linear_Regression_Predicted"]
)

plt.figure(figsize=(9, 6))

plt.scatter(
    predictions["Linear_Regression_Predicted"],
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title("Residual Analysis - Linear Regression")
plt.xlabel("Predicted LTV")
plt.ylabel("Residual")

plt.tight_layout()

plt.savefig(
    VIS_DIR / "linear_regression_residuals.png",
    dpi=200
)

plt.close()


# --------------------------------------------------
# 7. FEATURE IMPORTANCE
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

X = df.drop(
    columns=[
        "Customer_ID",
        "LTV",
        "Customer_Segment"
    ],
    errors="ignore"
)

y = df["LTV"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Permutation importance works with the complete pipeline
importance = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=5,
    random_state=42,
    scoring="r2"
)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": importance.importances_mean
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
).head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"][::-1],
    importance_df["Importance"][::-1],
    edgecolor="black"
)

plt.title("Top 10 Feature Importance - Decision Tree")
plt.xlabel("Permutation Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    VIS_DIR / "feature_importance.png",
    dpi=200
)

plt.close()

# Save importance data
importance_df.to_csv(
    BASE_DIR / "results" / "feature_importance.csv",
    index=False
)


print("\nAll visualizations created successfully!")

print("\nFiles created:")

for file in sorted(VIS_DIR.glob("*.png")):
    print("-", file.name)

print(
    "\nFeature importance saved to "
    "results/feature_importance.csv"
)