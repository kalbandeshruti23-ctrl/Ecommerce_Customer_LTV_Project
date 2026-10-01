from pathlib import Path
import base64

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Lifetime Value Prediction",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "results" / "processed_customer_data.csv"

MODEL_LR_FILE = BASE_DIR / "models" / "linear_regression.pkl"

MODEL_DT_FILE = BASE_DIR / "models" / "decision_tree.pkl"

MODEL_COMPARISON_FILE = (
    BASE_DIR / "results" / "model_comparison.csv"
)

BACKGROUND_FILE = BASE_DIR / "assets" / "background.jpg"


# ============================================================
# BACKGROUND IMAGE
# ============================================================

def set_background(image_path):

    if image_path.exists():

        try:

            with open(image_path, "rb") as file:

                encoded = base64.b64encode(
                    file.read()
                ).decode()

            st.markdown(
                f"""
                <style>

                .stApp {{
                    background-image:
                        linear-gradient(
                            rgba(255, 248, 240, 0.62),
                            rgba(255, 248, 240, 0.62)
                        ),
                        url("data:image/jpeg;base64,{encoded}");

                    background-size: cover;
                    background-position: center;
                    background-attachment: fixed;
                    background-repeat: no-repeat;
                }}

                </style>
                """,
                unsafe_allow_html=True
            )

        except Exception:

            st.markdown(
                """
                <style>
                .stApp {
                    background-color: #FFF8F0;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

    else:

        st.markdown(
            """
            <style>

            .stApp {
                background-color: #FFF8F0;
            }

            </style>
            """,
            unsafe_allow_html=True
        )


set_background(BACKGROUND_FILE)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       MAIN PAGE
       ====================================================== */

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    [data-testid="stSidebar"] {

        background-color: #111111;

        border-right: 4px solid #FF7A00;
    }

    [data-testid="stSidebar"] * {

        color: white !important;

    }


    /* ======================================================
       HEADINGS
       ====================================================== */

    h1 {

        color: #111111 !important;

        font-weight: 900 !important;

        font-size: 36px !important;

    }

    h2 {

        color: #111111 !important;

        font-weight: 900 !important;

    }

    h3 {

        color: #111111 !important;

        font-weight: 800 !important;

    }


    /* ======================================================
       SUBHEADER
       ====================================================== */

    .stSubheader {

        color: #FF7A00 !important;

        font-weight: 900 !important;

    }


    /* ======================================================
       NORMAL TEXT
       ====================================================== */

    p {

        color: #222222;

        font-weight: 600;

    }


    /* ======================================================
       METRIC CARDS
       ====================================================== */

    [data-testid="stMetric"] {

        background-color:
            rgba(255, 255, 255, 0.94);

        border:
            2px solid #FF7A00;

        border-radius:
            15px;

        padding:
            18px;

        box-shadow:
            0 6px 18px
            rgba(0, 0, 0, 0.14);

    }


    [data-testid="stMetricLabel"] {

        color:
            #555555 !important;

        font-weight:
            800 !important;

    }


    [data-testid="stMetricValue"] {

        color:
            #111111 !important;

        font-weight:
            900 !important;

    }


    /* ======================================================
       BUTTON
       ====================================================== */

    .stButton > button {

        background-color:
            #FF7A00;

        color:
            white;

        font-weight:
            900;

        border:
            2px solid #111111;

        border-radius:
            10px;

    }


    .stButton > button:hover {

        background-color:
            #111111;

        color:
            #FF7A00;

        border:
            2px solid #FF7A00;

    }


    /* ======================================================
       FORM
       ====================================================== */

    [data-testid="stForm"] {

        background-color:
            rgba(255, 255, 255, 0.94);

        border:
            2px solid #FFB366;

        border-radius:
            15px;

        padding:
            22px;

        box-shadow:
            0 6px 18px
            rgba(0, 0, 0, 0.10);

    }


    /* ======================================================
       INPUTS
       ====================================================== */

    .stTextInput input,
    .stNumberInput input {

        background-color:
            white;

        color:
            #111111;

        border-radius:
            8px;

    }


    div[data-baseweb="select"] {

        background-color:
            white;

        border-radius:
            8px;

    }


    /* ======================================================
       DATAFRAME
       ====================================================== */

    [data-testid="stDataFrame"] {

        border:
            2px solid #FF7A00;

        border-radius:
            10px;

        overflow:
            hidden;

    }


    /* ======================================================
       PREDICTION RESULT
       ====================================================== */

    .prediction-box {

        background-color:
            rgba(255, 248, 240, 0.97);

        border:
            3px solid #FF7A00;

        border-radius:
            16px;

        padding:
            25px;

        text-align:
            center;

        margin-top:
            20px;

        box-shadow:
            0 7px 22px
            rgba(0, 0, 0, 0.15);

    }

    .prediction-text {

        color:
            #555555;

        font-size:
            17px;

        font-weight:
            800;

    }

    .prediction-number {

        color:
            #111111;

        font-size:
            40px;

        font-weight:
            900;

        margin-top:
            8px;

    }


    /* ======================================================
       DIVIDER
       ====================================================== */

    hr {

        border-color:
            #FF7A00 !important;

    }


    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not DATA_FILE.exists():

        return None

    return pd.read_csv(DATA_FILE)


df = load_data()


# ============================================================
# CHECK DATA
# ============================================================

if df is None:

    st.error(
        "❌ processed_customer_data.csv was not found."
    )

    st.info(
        "Check that the file is inside the results folder."
    )

    st.stop()


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = {}

    if MODEL_LR_FILE.exists():

        models["Linear Regression"] = joblib.load(
            MODEL_LR_FILE
        )

    if MODEL_DT_FILE.exists():

        models["Decision Tree"] = joblib.load(
            MODEL_DT_FILE
        )

    return models


models = load_models()


# ============================================================
# LOAD MODEL COMPARISON
# ============================================================

@st.cache_data
def load_model_comparison():

    if MODEL_COMPARISON_FILE.exists():

        return pd.read_csv(
            MODEL_COMPARISON_FILE
        )

    return None


model_comparison = load_model_comparison()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <h2 style="
        text-align:center;
        color:white !important;
        font-weight:900;
    ">
        🛍️ Customer Analytics
    </h2>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")


page = st.sidebar.radio(
    "📌 Navigation",
    [
        "📊 Dashboard Overview",
        "👥 Customer Analysis",
        "🤖 Model Comparison",
        "🔮 Predict Customer LTV"
    ]
)


st.sidebar.markdown("---")


st.sidebar.markdown(
    """
    **PROJECT**

    Customer Lifetime Value Prediction

    **TECHNIQUES**

    Machine Learning  
    Customer Analytics  
    Predictive Analytics

    **MODELS**

    Linear Regression  
    Decision Tree
    """
)


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "🛍️ Customer Lifetime Value Prediction"
)

st.subheader(
    "E-Commerce Customer Analytics Dashboard"
)

st.write(
    "Analyze customer purchasing behavior, identify valuable "
    "customers, compare machine learning models, and predict "
    "Customer Lifetime Value."
)

st.divider()


# ============================================================
# PAGE 1 — DASHBOARD OVERVIEW
# ============================================================

if page == "📊 Dashboard Overview":

    st.header("📊 Dashboard Overview")

    st.write(
        "Get a complete overview of customer value, spending "
        "behavior, purchasing activity, and customer segments."
    )


    # --------------------------------------------------------
    # KPI VALUES
    # --------------------------------------------------------

    total_customers = len(df)


    if "LTV" in df.columns:

        average_ltv = df["LTV"].mean()

    else:

        average_ltv = 0


    if "Total_Spending" in df.columns:

        total_spending = df["Total_Spending"].sum()

    else:

        total_spending = 0


    if "Total_Purchases" in df.columns:

        average_purchases = (
            df["Total_Purchases"].mean()
        )

    else:

        average_purchases = 0


    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "👥 Total Customers",
            f"{total_customers:,}"
        )


    with col2:

        st.metric(
            "💰 Average LTV",
            f"{average_ltv:,.2f}"
        )


    with col3:

        st.metric(
            "🛒 Total Spending",
            f"{total_spending:,.2f}"
        )


    with col4:

        st.metric(
            "📦 Avg Purchases",
            f"{average_purchases:.2f}"
        )


    st.divider()


    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    chart1, chart2 = st.columns(2)


    with chart1:

        st.subheader(
            "💰 LTV Distribution"
        )


        if "LTV" in df.columns:

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )


            ax.hist(
                df["LTV"].dropna(),
                bins=25,
                edgecolor="black"
            )


            ax.set_xlabel(
                "Customer Lifetime Value"
            )

            ax.set_ylabel(
                "Number of Customers"
            )


            ax.set_title(
                "Customer Lifetime Value Distribution",
                fontweight="bold"
            )


            ax.grid(
                axis="y",
                alpha=0.25
            )


            st.pyplot(fig)

            plt.close(fig)


    with chart2:

        st.subheader(
            "👥 Customer Segments"
        )


        if "Customer_Segment" in df.columns:

            segment_counts = (
                df["Customer_Segment"]
                .value_counts()
            )


            fig, ax = plt.subplots(
                figsize=(8, 5)
            )


            ax.bar(
                segment_counts.index.astype(str),
                segment_counts.values,
                edgecolor="black"
            )


            ax.set_xlabel(
                "Customer Segment"
            )

            ax.set_ylabel(
                "Number of Customers"
            )


            ax.set_title(
                "Customer Segment Distribution",
                fontweight="bold"
            )


            ax.tick_params(
                axis="x",
                rotation=20
            )


            ax.grid(
                axis="y",
                alpha=0.25
            )


            st.pyplot(fig)

            plt.close(fig)


    # --------------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------------

    st.subheader(
        "📋 Dataset Preview"
    )


    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ============================================================
# PAGE 2 — CUSTOMER ANALYSIS
# ============================================================

elif page == "👥 Customer Analysis":

    st.header(
        "👥 Customer Analysis"
    )

    st.write(
        "Explore customer behavior and understand "
        "the factors associated with Customer Lifetime Value."
    )


    numeric_columns = (
        df.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )


    if numeric_columns:

        selected_column = st.selectbox(
            "📌 Select a variable to analyze",
            numeric_columns
        )


        chart1, chart2 = st.columns(2)


        with chart1:

            st.subheader(
                f"📊 {selected_column} Distribution"
            )


            fig, ax = plt.subplots(
                figsize=(8, 5)
            )


            ax.hist(
                df[selected_column].dropna(),
                bins=25,
                edgecolor="black"
            )


            ax.set_xlabel(
                selected_column
            )

            ax.set_ylabel(
                "Frequency"
            )


            ax.set_title(
                f"{selected_column} Distribution",
                fontweight="bold"
            )


            ax.grid(
                axis="y",
                alpha=0.25
            )


            st.pyplot(fig)

            plt.close(fig)


        with chart2:

            st.subheader(
                "💰 Average LTV by Segment"
            )


            if (
                "Customer_Segment" in df.columns
                and "LTV" in df.columns
            ):

                segment_ltv = (
                    df.groupby(
                        "Customer_Segment"
                    )["LTV"]
                    .mean()
                    .sort_values(
                        ascending=False
                    )
                )


                fig, ax = plt.subplots(
                    figsize=(8, 5)
                )


                ax.bar(
                    segment_ltv.index.astype(str),
                    segment_ltv.values,
                    edgecolor="black"
                )


                ax.set_xlabel(
                    "Customer Segment"
                )

                ax.set_ylabel(
                    "Average LTV"
                )


                ax.set_title(
                    "Average LTV by Customer Segment",
                    fontweight="bold"
                )


                ax.tick_params(
                    axis="x",
                    rotation=20
                )


                ax.grid(
                    axis="y",
                    alpha=0.25
                )


                st.pyplot(fig)

                plt.close(fig)


    st.divider()


    # --------------------------------------------------------
    # CUSTOMER SEARCH
    # --------------------------------------------------------

    st.subheader(
        "🔎 Customer Records"
    )


    search_term = st.text_input(
        "Search Customer ID",
        placeholder="Enter Customer ID..."
    )


    display_df = df.copy()


    if (
        search_term
        and "Customer_ID" in df.columns
    ):

        display_df = df[
            df["Customer_ID"]
            .astype(str)
            .str.contains(
                search_term,
                case=False,
                na=False
            )
        ]


    st.dataframe(
        display_df,
        use_container_width=True
    )


# ============================================================
# PAGE 3 — MODEL COMPARISON
# ============================================================

elif page == "🤖 Model Comparison":

    st.header(
        "🤖 Model Comparison"
    )

    st.write(
        "Compare the performance of the trained machine "
        "learning models using standard regression metrics."
    )


    if model_comparison is not None:

        st.subheader(
            "📊 Model Performance"
        )


        st.dataframe(
            model_comparison,
            use_container_width=True
        )


        # ----------------------------------------------------
        # FIND MODEL COLUMN
        # ----------------------------------------------------

        model_column = None


        possible_model_columns = [
            "Model",
            "model",
            "Model_Name",
            "Model Name"
        ]


        for column in possible_model_columns:

            if column in model_comparison.columns:

                model_column = column

                break


        # ----------------------------------------------------
        # FIND METRIC COLUMNS
        # ----------------------------------------------------

        metric_columns = [
            column
            for column in [
                "R2 Score",
                "R2",
                "MAE",
                "MSE",
                "RMSE"
            ]
            if column in model_comparison.columns
        ]


        if metric_columns:

            selected_metric = st.selectbox(
                "📌 Select Evaluation Metric",
                metric_columns
            )


            if model_column:

                chart_data = (
                    model_comparison[
                        [
                            model_column,
                            selected_metric
                        ]
                    ]
                    .dropna()
                )


                fig, ax = plt.subplots(
                    figsize=(10, 5)
                )


                ax.bar(
                    chart_data[
                        model_column
                    ].astype(str),
                    chart_data[
                        selected_metric
                    ],
                    edgecolor="black"
                )


                ax.set_xlabel(
                    "Model"
                )

                ax.set_ylabel(
                    selected_metric
                )


                ax.set_title(
                    f"Model Comparison - {selected_metric}",
                    fontweight="bold"
                )


                ax.tick_params(
                    axis="x",
                    rotation=15
                )


                ax.grid(
                    axis="y",
                    alpha=0.25
                )


                st.pyplot(fig)

                plt.close(fig)


    else:

        st.warning(
            "⚠️ model_comparison.csv was not found."
        )


# ============================================================
# PAGE 4 — PREDICT CUSTOMER LTV
# ============================================================

elif page == "🔮 Predict Customer LTV":

    st.header(
        "🔮 Predict Customer Lifetime Value"
    )

    st.write(
        "Enter customer information and select a machine "
        "learning model to estimate Customer Lifetime Value."
    )


    if not models:

        st.error(
            "❌ No trained models were found."
        )

        st.stop()


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    excluded_columns = [
        "Customer_ID",
        "LTV",
        "Customer_Segment"
    ]


    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]


    numeric_features = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]


    categorical_features = [
        column
        for column in feature_columns
        if column not in numeric_features
    ]


    # --------------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------------

    selected_model_name = st.selectbox(
        "🤖 Select Prediction Model",
        list(models.keys())
    )


    selected_model = models[
        selected_model_name
    ]


    # --------------------------------------------------------
    # FORM
    # --------------------------------------------------------

    with st.form(
        "prediction_form"
    ):

        st.subheader(
            "📝 Customer Information"
        )


        input_values = {}


        # ----------------------------------------------------
        # NUMERIC INPUTS
        # ----------------------------------------------------

        if numeric_features:

            st.markdown(
                "**📊 Numeric Features**"
            )


            numeric_columns = st.columns(2)


            for index, feature in enumerate(
                numeric_features
            ):

                with numeric_columns[
                    index % 2
                ]:

                    series = pd.to_numeric(
                        df[feature],
                        errors="coerce"
                    )


                    default_value = float(
                        series.median()
                    )


                    min_value = float(
                        series.min()
                    )


                    max_value = float(
                        series.max()
                    )


                    if min_value == max_value:

                        max_value = (
                            min_value + 1
                        )


                    input_values[feature] = (
                        st.number_input(
                            feature,
                            min_value=min_value,
                            max_value=max_value,
                            value=default_value
                        )
                    )


        # ----------------------------------------------------
        # CATEGORICAL INPUTS
        # ----------------------------------------------------

        if categorical_features:

            st.markdown(
                "**🏷️ Categorical Features**"
            )


            categorical_columns = st.columns(2)


            for index, feature in enumerate(
                categorical_features
            ):

                with categorical_columns[
                    index % 2
                ]:

                    categories = (
                        df[feature]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )


                    if categories:

                        input_values[feature] = (
                            st.selectbox(
                                feature,
                                categories
                            )
                        )


        st.markdown("---")


        predict_button = st.form_submit_button(
            "🔮 Predict Customer LTV"
        )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if predict_button:

        try:

            input_df = pd.DataFrame(
                [input_values]
            )


            input_df = input_df[
                feature_columns
            ]


            prediction = selected_model.predict(
                input_df
            )


            predicted_ltv = float(
                prediction[0]
            )


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            # Build HTML with the opening tag at the very start of the string.
            # This prevents Streamlit Markdown from treating it as a code block.
            prediction_html = (
                f'<div class="prediction-box">'
                f'<div class="prediction-text">Predicted Customer Lifetime Value</div>'
                f'<div class="prediction-number">₹{predicted_ltv:,.2f}</div>'
                f'<div style="margin-top:10px; color:#FF7A00; '
                f'font-weight:900; font-size:16px;">'
                f'🤖 Model: {selected_model_name}'
                f'</div>'
                f'</div>'
            )

            st.markdown(prediction_html, unsafe_allow_html=True)


            # ------------------------------------------------
            # CUSTOMER SEGMENT
            # ------------------------------------------------

            if (
                "Customer_Segment" in df.columns
                and "LTV" in df.columns
            ):

                q1 = df["LTV"].quantile(
                    0.25
                )

                q2 = df["LTV"].quantile(
                    0.50
                )

                q3 = df["LTV"].quantile(
                    0.75
                )


                if predicted_ltv <= q1:

                    segment = "Low Value"

                elif predicted_ltv <= q2:

                    segment = "Medium Value"

                elif predicted_ltv <= q3:

                    segment = "High Value"

                else:

                    segment = "Premium"


                st.success(
                    f"🎯 Predicted Customer Segment: **{segment}**"
                )


        except Exception as error:

            st.error(
                "❌ Prediction could not be completed."
            )

            st.exception(error)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🛍️ Customer Lifetime Value Prediction | "
    "Machine Learning • Customer Analytics • Predictive Modeling | "
    "M.Tech Academic Project"
)