
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CUSTOMER LIFETIME VALUE - DATA PREPROCESSING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "ecommerce_customer_ltv_3000.xlsx"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load the original customer dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_excel(
        DATA_PATH,
        sheet_name="Customer_Data"
    )

    print("\nDataset loaded successfully.")
    print("Original shape:", df.shape)

    return df


def preprocess_data(df):
    """Clean data and create customer behavior features."""

    df = df.copy()

    # --------------------------------------------------------
    # 1. Remove duplicate records
    # --------------------------------------------------------

    df.drop_duplicates(
        subset=["Customer_ID"],
        keep="first",
        inplace=True
    )

    # --------------------------------------------------------
    # 2. Handle missing values
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:
        df[column] = df[column].fillna(
            df[column].median()
        )

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in categorical_columns:
        if column != "Customer_ID":
            df[column] = df[column].fillna(
                df[column].mode()[0]
            )

    # --------------------------------------------------------
    # 3. Ensure valid values
    # --------------------------------------------------------

    df["Total_Purchases"] = (
        df["Total_Purchases"].clip(lower=0)
    )

    df["Total_Spending"] = (
        df["Total_Spending"].clip(lower=0)
    )

    df["Average_Order_Value"] = (
        df["Average_Order_Value"].clip(lower=0)
    )

    df["Recency_Days"] = (
        df["Recency_Days"].clip(lower=0)
    )

    df["Customer_Tenure_Months"] = (
        df["Customer_Tenure_Months"].clip(lower=0)
    )

    df["LTV"] = df["LTV"].clip(lower=0)

    # --------------------------------------------------------
    # 4. Feature engineering
    # --------------------------------------------------------

    # Average historical spending per purchase
    df["Calculated_Average_Order_Value"] = (
        df["Total_Spending"]
        / df["Total_Purchases"].replace(0, 1)
    )

    # Purchase frequency per month of customer tenure
    df["Purchases_Per_Tenure_Month"] = (
        df["Total_Purchases"]
        / df["Customer_Tenure_Months"].replace(0, 1)
    )

    # RFM: Recency score
    # Higher score means a more recent purchase
    df["Recency_Score"] = (
        366 - df["Recency_Days"]
    )

    # RFM: Frequency
    df["Frequency_Score"] = df["Total_Purchases"]

    # RFM: Monetary value
    df["Monetary_Score"] = df["Total_Spending"]

    # Engagement score
    df["Engagement_Score"] = (
        df["Website_Visits"] * 0.6
        + df["Customer_Satisfaction"] * 10
        - df["Support_Interactions"] * 0.5
    )

    # Customer segments based on spending
    df["Customer_Segment"] = pd.qcut(
        df["Total_Spending"].rank(method="first"),
        q=4,
        labels=[
            "Low Value",
            "Medium Value",
            "High Value",
            "Premium"
        ]
    )

    # --------------------------------------------------------
    # 5. Save processed dataset
    # --------------------------------------------------------

    output_path = (
        RESULTS_DIR / "processed_customer_data.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("\nPreprocessing completed.")
    print("Processed shape:", df.shape)
    print("Missing values remaining:", df.isnull().sum().sum())
    print("Saved to:", output_path)

    return df


def main():
    df = load_data()
    processed_df = preprocess_data(df)

    print("\nFirst 5 processed records:")
    print(processed_df.head())

    print("\nCustomer segment distribution:")
    print(
        processed_df["Customer_Segment"].value_counts()
    )


if __name__ == "__main__":
    main()