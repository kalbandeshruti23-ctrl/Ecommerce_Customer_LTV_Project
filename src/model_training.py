from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "results" / "processed_customer_data.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
VIS_DIR = BASE_DIR / "visualizations"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
VIS_DIR.mkdir(exist_ok=True)


# Load processed dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# Target and features
TARGET = "LTV"

# Exclude identifiers, target, and segment labels
excluded_columns = ["Customer_ID", TARGET, "Customer_Segment"]
X = df.drop(columns=excluded_columns, errors="ignore")
y = df[TARGET]

# Separate numeric and categorical features
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# Preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=8,
        min_samples_split=10,
        random_state=42
    )
}

results = []
trained_models = {}

for model_name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5

    results.append({
        "Model": model_name,
        "R2 Score": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    })

    trained_models[model_name] = pipeline

    filename = model_name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(pipeline, MODEL_DIR / filename)

    print(f"\n{model_name}")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE:      {mae:.2f}")
    print(f"RMSE:     {rmse:.2f}")


# Save model comparison
comparison = pd.DataFrame(results)
comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

print("\nModel comparison:")
print(comparison.to_string(index=False))


# Save test predictions
prediction_df = pd.DataFrame({
    "Actual_LTV": y_test.values,
    "Linear_Regression_Predicted": trained_models[
        "Linear Regression"
    ].predict(X_test),
    "Decision_Tree_Predicted": trained_models[
        "Decision Tree"
    ].predict(X_test)
})

prediction_df.to_csv(
    RESULTS_DIR / "predictions.csv",
    index=False
)


# Predicted vs actual plot for Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(
    prediction_df["Actual_LTV"],
    prediction_df["Linear_Regression_Predicted"],
    alpha=0.6
)

min_value = min(
    prediction_df["Actual_LTV"].min(),
    prediction_df["Linear_Regression_Predicted"].min()
)
max_value = max(
    prediction_df["Actual_LTV"].max(),
    prediction_df["Linear_Regression_Predicted"].max()
)

plt.plot([min_value, max_value], [min_value, max_value], "--")
plt.xlabel("Actual LTV")
plt.ylabel("Predicted LTV")
plt.title("Linear Regression: Actual vs Predicted LTV")
plt.tight_layout()
plt.savefig(VIS_DIR / "linear_regression_actual_vs_predicted.png")
plt.close()


# Decision Tree visualization
tree_model = trained_models["Decision Tree"].named_steps["model"]
feature_names = trained_models[
    "Decision Tree"
].named_steps["preprocessor"].get_feature_names_out()

plt.figure(figsize=(20, 10))
plot_tree(
    tree_model,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=7
)
plt.title("Decision Tree Regression (First 3 Levels)")
plt.tight_layout()
plt.savefig(VIS_DIR / "decision_tree.png", dpi=200)
plt.close()


print("\nAll files saved successfully!")
print("Models:", MODEL_DIR)
print("Comparison:", RESULTS_DIR / "model_comparison.csv")
print("Predictions:", RESULTS_DIR / "predictions.csv")
print("Visualizations:", VIS_DIR)