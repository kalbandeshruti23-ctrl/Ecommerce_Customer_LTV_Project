
# Customer Lifetime Value Prediction for E-Commerce

## 1. Project Overview

Customer Lifetime Value (LTV) Prediction is a machine learning project
that estimates the potential lifetime value of e-commerce customers
using customer demographics, purchasing behaviour, engagement, and
transaction-related features.

The project applies regression algorithms to estimate customer value
and provides an interactive Streamlit dashboard for data exploration,
model comparison, and customer-level predictions.

**Note:** The dataset used in this academic project is synthetic.
The results demonstrate a machine-learning workflow and should not
be interpreted as validated predictions of real customer behaviour.

## 2. Objectives

- Predict customer lifetime value using regression models.
- Analyse customer purchasing patterns and spending behaviour.
- Perform data cleaning and preprocessing.
- Apply feature engineering, including RFM-related features.
- Compare Linear Regression and Decision Tree Regression.
- Evaluate models using R², MAE, MSE, and RMSE.
- Visualize customer segments and model performance.
- Develop an interactive Streamlit dashboard.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and evaluation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Joblib | Saving trained models |
| Streamlit | Interactive dashboard |
| OpenPyXL | Reading Excel datasets |

## 4. Machine Learning Algorithms

### Linear Regression

Linear Regression estimates the relationship between customer
features and lifetime value. It provides a baseline regression model.

### Decision Tree Regression

Decision Tree Regression learns decision rules from customer data
to estimate lifetime value. It can model nonlinear relationships.

## 5. Dataset

The project uses a synthetic dataset containing 3,000 customer records.

Important features include:

- Customer_ID
- Age
- Gender
- City
- Total_Purchases
- Total_Spending
- Average_Order_Value
- Purchase_Frequency
- Recency_Days
- Customer_Tenure_Months
- Website_Visits
- Discount_Usage
- Product_Category
- Customer_Satisfaction
- Support_Interactions
- LTV

The dataset also includes engineered features and customer segments
created during preprocessing.

## 6. Methodology

1. Load the customer dataset.
2. Inspect and clean the data.
3. Handle missing values and duplicate records.
4. Perform feature engineering and customer segmentation.
5. Separate input features and the LTV target.
6. Split the data into training and testing sets.
7. Train Linear Regression and Decision Tree Regression models.
8. Evaluate model performance.
9. Save trained models and prediction results.
10. Visualize results through charts.
11. Use Streamlit to explore data and estimate customer LTV.

## 7. Model Evaluation

The models are evaluated using the following metrics:

- **R² Score:** Measures the proportion of target variation explained
  by the model.
- **MAE:** Measures the average absolute prediction error.
- **MSE:** Measures the average squared prediction error.
- **RMSE:** Measures prediction error in the target's units.

The comparison results are saved in:

`results/model_comparison.csv`

## 8. Project Structure

```text
Ecommerce_Customer_LTV_Project/
│
├── data/
│   └── ecommerce_customer_ltv_3000.xlsx
│
├── models/
│   ├── linear_regression.pkl
│   └── decision_tree.pkl
│
├── results/
│   ├── processed_customer_data.csv
│   ├── model_comparison.csv
│   └── predictions.csv
│
├── src/
│   ├── preprocessing.py
│   └── model_training.py
│
├── visualizations/
│   ├── linear_regression_actual_vs_predicted.png
│   └── decision_tree.png
│
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

## 9. Installation

Create and activate a virtual environment.

### Windows

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

## 10. Execution

### Run the complete machine-learning pipeline

```powershell
python main.py
```

### Run the Streamlit dashboard

```powershell
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

## 11. Dashboard Features

- Customer analytics overview
- Customer Lifetime Value distribution
- Customer segment analysis
- Regression model comparison
- Interactive customer LTV prediction

## 12. Results

The project generates model evaluation metrics, predicted customer
values, trained model files, and visualizations.

The numerical results should be reported from the actual output
of `results/model_comparison.csv`.

Because the dataset is synthetic, model performance may reflect
relationships built into the generated data. Real transaction data
would be required to validate practical business performance.

## 13. Future Scope

- Use real e-commerce transaction data.
- Explore Random Forest and gradient-boosting models.
- Perform hyperparameter tuning and cross-validation.
- Develop time-based customer value forecasts.
- Integrate customer segmentation with marketing strategies.
- Deploy the dashboard for business users.

## 14. Conclusion

This project demonstrates an end-to-end machine-learning workflow
for customer lifetime value estimation. It includes data
preprocessing, feature engineering, regression model training,
performance evaluation, visualization, and an interactive dashboard.
The system provides an academic foundation for understanding
customer analytics and can be extended using real-world data.

## Author

Developed as an academic machine-learning project.