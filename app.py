import io
import os

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

# ============================================================
# FRAUDSHIELD AI — CREDIT CARD FRAUD DETECTION DASHBOARD
# Created by: Abeeha Zeeshan
# ============================================================

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- CUSTOM STYLING --------------------

st.markdown(
    """
    <style>
        .main {
            background-color: #f6f8fc;
        }

        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2rem;
        }

        .hero {
            padding: 1.5rem;
            border-radius: 16px;
            background: linear-gradient(120deg, #101d3a, #174b78);
            color: white;
            margin-bottom: 1.2rem;
        }

        .hero h1 {
            color: white;
            margin-bottom: 0.3rem;
        }

        .hero p {
            color: #dce8f8;
            margin-bottom: 0;
        }

        div[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border: 1px solid #d7deea;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

div[data-testid="stMetric"] label,
div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {
    color: #475569 !important;
    opacity: 1 !important;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] div {
    color: #14213d !important;
    opacity: 1 !important;
}

div[data-testid="stMetricDelta"] {
    color: #334155 !important;
}

        .notice {
            padding: 12px 15px;
            border-radius: 10px;
            background: #edf4ff;
            border-left: 4px solid #3574c8;
            margin: 10px 0 18px 0;
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- DEMO DATA --------------------

def make_demo_data(n=6000, seed=42):
    """Create synthetic data for demonstration only."""
    rng = np.random.default_rng(seed)

    fraud_count = max(30, int(n * 0.025))
    labels = np.zeros(n, dtype=int)
    labels[:fraud_count] = 1
    rng.shuffle(labels)

    data = {
        "Time": rng.uniform(0, 172800, n),
        "Amount": np.clip(rng.exponential(65, n), 0, 2500),
    }

    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1, n)

    df = pd.DataFrame(data)
    fraud_mask = labels == 1

    # Add artificial patterns so the demo model has something to learn.
    df.loc[fraud_mask, "Amount"] *= 2.5
    df.loc[fraud_mask, "V1"] += rng.normal(2.0, 0.7, fraud_count)
    df.loc[fraud_mask, "V4"] += rng.normal(2.0, 0.7, fraud_count)
    df.loc[fraud_mask, "V14"] -= rng.normal(2.0, 0.7, fraud_count)

    df["Class"] = labels
    return df


# -------------------- DATA PREPARATION --------------------

def prepare_dataset(df):
    """Validate and clean a credit card transaction dataset."""

    if df is None or df.empty:
        raise ValueError("The dataset is empty.")

    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    if "Class" not in df.columns:
        raise ValueError(
            "The CSV must contain a 'Class' column, where 0 means "
            "legitimate and 1 means fraud."
        )

    # Keep only rows with valid binary target labels.
    df["Class"] = pd.to_numeric(df["Class"], errors="coerce")
    df = df.dropna(subset=["Class"])
    df = df[df["Class"].isin([0, 1])].copy()
    df["Class"] = df["Class"].astype(int)

    if df.empty:
        raise ValueError("No valid Class labels (0 or 1) were found.")

    if df["Class"].nunique() < 2:
        raise ValueError(
            "The dataset needs both legitimate transactions (0) "
            "and fraud transactions (1)."
        )

    feature_columns = [col for col in df.columns if col != "Class"]

    if not feature_columns:
        raise ValueError("No feature columns were found.")

    # Convert features to numeric values.
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[feature_columns] = df[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    # Fill missing values using each column's median.
    for col in feature_columns:
        median_value = df[col].median()
        if pd.isna(median_value):
            median_value = 0
        df[col] = df[col].fillna(median_value)

    return df, feature_columns


# -------------------- CACHED MODEL TRAINING --------------------

@st.cache_resource(show_spinner=False)
def train_model_cached(X_train, y_train):
    """
    Train and cache the model so normal Streamlit reruns
    do not retrain it unnecessarily.
    """
    model = RandomForestClassifier(
        n_estimators=30,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    return model


# -------------------- LOAD DATA --------------------

st.sidebar.title("🛡️ FraudShield AI")
st.sidebar.caption("AI-powered transaction analysis")

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload a transaction CSV",
    type=["csv"],
    help="Your CSV must contain a Class column with 0 and 1 labels.",
)

demo_mode = False
data_source = ""

try:
    if uploaded_file is not None:
        df_raw = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
        data_source = "Uploaded CSV dataset"

    elif os.path.exists("creditcard.csv"):
        if os.path.getsize("creditcard.csv") == 0:
            st.error(
                "Your local creditcard.csv file is empty. "
                "Replace it with the extracted Kaggle dataset."
            )
            st.stop()

        df_raw = pd.read_csv("creditcard.csv")
        data_source = "Local Kaggle dataset"

    else:
        df_raw = make_demo_data()
        demo_mode = True
        data_source = "Synthetic demonstration dataset"

    df, feature_columns = prepare_dataset(df_raw)

except Exception as exc:
    st.error(f"Could not load or process the dataset: {exc}")
    st.info(
        "Check that creditcard.csv is in the same folder as app.py, "
        "or upload a valid CSV using the sidebar."
    )
    st.stop()


# -------------------- HEADER --------------------

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ FraudShield AI</h1>
        <p>
            Machine-learning dashboard for exploring credit card
            fraud detection, transaction patterns, and model performance.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if demo_mode:
    st.warning(
        "DEMO MODE: Synthetic data is being used. These results do not "
        "represent real credit card transactions or real-world fraud "
        "detection performance."
    )
else:
    st.success(f"Dataset loaded successfully: {data_source}")

st.caption(
    "Educational project by Abeeha Zeeshan. This tool is a prototype, "
    "not a production banking or fraud prevention system."
)

# -------------------- DATASET SUMMARY --------------------

total_transactions = len(df)
fraud_count = int((df["Class"] == 1).sum())
legitimate_count = int((df["Class"] == 0).sum())
fraud_percentage = (
    fraud_count / total_transactions * 100
    if total_transactions
    else 0
)

st.subheader("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total transactions", f"{total_transactions:,}")
col2.metric("Legitimate transactions", f"{legitimate_count:,}")
col3.metric("Fraud transactions", f"{fraud_count:,}")
col4.metric("Fraud percentage", f"{fraud_percentage:.3f}%")

st.markdown("---")

# -------------------- TRAIN / TEST SPLIT --------------------

X = df[feature_columns].copy()
y = df["Class"].copy()

if y.value_counts().min() < 2:
    st.error(
        "Each class needs at least two examples for a stratified "
        "train/test split. Please use a larger dataset."
    )
    st.stop()

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )
except ValueError as exc:
    st.error(f"Could not split the dataset: {exc}")
    st.stop()

# Limit training rows for faster first runs on ordinary computers.
# The test set remains separate and is used for evaluation.
MAX_TRAIN_ROWS = 80000

if len(X_train) > MAX_TRAIN_ROWS:
    X_train, _, y_train, _ = train_test_split(
        X_train,
        y_train,
        train_size=MAX_TRAIN_ROWS,
        stratify=y_train,
        random_state=42,
    )

st.sidebar.markdown("---")
st.sidebar.caption(f"Training rows: {len(X_train):,}")
st.sidebar.caption(f"Evaluation rows: {len(X_test):,}")
st.sidebar.caption(f"Features: {len(feature_columns)}")

# -------------------- TRAIN MODEL --------------------

with st.spinner(
    "Training or loading the cached fraud detection model..."
):
    try:
        model = train_model_cached(X_train, y_train)
    except Exception as exc:
        st.error(f"Model training failed: {exc}")
        st.stop()

# -------------------- EVALUATION --------------------

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

try:
    auc = roc_auc_score(y_test, y_probability)
except ValueError:
    auc = float("nan")

st.subheader("Model Performance")

metric1, metric2, metric3, metric4, metric5 = st.columns(5)

metric1.metric("Accuracy", f"{accuracy * 100:.2f}%")
metric2.metric("Precision", f"{precision * 100:.2f}%")
metric3.metric("Recall", f"{recall * 100:.2f}%")
metric4.metric("F1 score", f"{f1 * 100:.2f}%")
metric5.metric(
    "ROC-AUC",
    f"{auc:.3f}" if not np.isnan(auc) else "N/A",
)

st.caption(
    "Metrics are calculated on the held-out test split. For large datasets, "
    "training uses a stratified subset of up to 80,000 rows to improve speed. "
    "Accuracy alone can be misleading when fraud is rare."
)

# -------------------- NAVIGATION --------------------

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Explore Transactions",
        "Check a Transaction",
        "About the Project",
    ],
)

# -------------------- DASHBOARD PAGE --------------------

if page == "Dashboard":
    st.subheader("Fraud Detection Dashboard")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        class_counts = pd.DataFrame(
            {
                "Transaction type": [
                    "Legitimate",
                    "Fraud",
                ],
                "Count": [
                    legitimate_count,
                    fraud_count,
                ],
            }
        )

        fig_class = px.bar(
            class_counts,
            x="Transaction type",
            y="Count",
            color="Transaction type",
            title="Legitimate vs Fraud Transactions",
            color_discrete_map={
                "Legitimate": "#2878B5",
                "Fraud": "#D9485F",
            },
        )
        fig_class.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_class, use_container_width=True)

    with chart_col2:
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Predicted Legitimate", "Predicted Fraud"],
                y=["Actual Legitimate", "Actual Fraud"],
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                showscale=True,
            )
        )
        fig_cm.update_layout(
            title="Confusion Matrix",
            xaxis_title="Model prediction",
            yaxis_title="Actual class",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Feature Importance")

    importance_df = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False).head(12)

    fig_importance = px.bar(
        importance_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 12 Features Used by the Model",
    )
    fig_importance.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis_title="",
    )
    st.plotly_chart(fig_importance, use_container_width=True)

    st.subheader("Prediction Results on Test Data")

    results_df = pd.DataFrame(
        {
            "Actual class": y_test.to_numpy(),
            "Predicted class": y_pred,
            "Fraud probability": y_probability,
        }
    )

    results_df["Actual class"] = results_df["Actual class"].map(
        {0: "Legitimate", 1: "Fraud"}
    )
    results_df["Predicted class"] = results_df["Predicted class"].map(
        {0: "Legitimate", 1: "Fraud"}
    )
    results_df["Fraud probability"] = (
        results_df["Fraud probability"].map(lambda value: f"{value:.2%}")
    )

    st.dataframe(
        results_df.head(20),
        use_container_width=True,
        hide_index=True,
    )

# -------------------- EXPLORE TRANSACTIONS PAGE --------------------

elif page == "Explore Transactions":
    st.subheader("Explore Transactions")

    display_df = df.copy()
    display_df["Transaction type"] = display_df["Class"].map(
        {0: "Legitimate", 1: "Fraud"}
    )

    filter_choice = st.selectbox(
        "Filter by transaction type",
        ["All transactions", "Legitimate only", "Fraud only"],
    )

    if filter_choice == "Legitimate only":
        display_df = display_df[display_df["Class"] == 0]
    elif filter_choice == "Fraud only":
        display_df = display_df[display_df["Class"] == 1]

    st.write(f"Showing {len(display_df):,} transactions.")

    st.dataframe(
        display_df.head(1000),
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download displayed transactions as CSV",
        data=csv_bytes,
        file_name="fraudshield_transactions.csv",
        mime="text/csv",
    )

    if "Amount" in df.columns:
        st.subheader("Transaction Amount Distribution")

        fig_amount = px.histogram(
            display_df,
            x="Amount",
            color="Transaction type",
            nbins=40,
            title="Transaction Amounts",
            color_discrete_map={
                "Legitimate": "#2878B5",
                "Fraud": "#D9485F",
            },
        )
        st.plotly_chart(fig_amount, use_container_width=True)

# -------------------- SINGLE TRANSACTION PAGE --------------------

elif page == "Check a Transaction":
    st.subheader("Check a Transaction")
    st.write(
        "Enter transaction details to obtain an educational model prediction."
    )

    if "Amount" not in feature_columns:
        st.warning(
            "This dataset has no Amount column. You can still explore "
            "the dashboard, but this transaction form needs an Amount feature."
        )
    else:
        with st.form("transaction_form"):
            amount = st.number_input(
                "Transaction amount",
                min_value=0.0,
                value=100.0,
                step=10.0,
            )

            time_value = st.number_input(
                "Transaction time (seconds, if applicable)",
                min_value=0.0,
                value=1000.0,
                step=100.0,
            )

            st.caption(
                "For this demo form, the V1–V28 features are set to zero "
                "because actual transformed feature values are not supplied."
            )

            submitted = st.form_submit_button(
                "Analyze Transaction",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            transaction = {col: 0.0 for col in feature_columns}

            transaction["Amount"] = amount

            if "Time" in transaction:
                transaction["Time"] = time_value

            input_df = pd.DataFrame(
                [transaction],
                columns=feature_columns,
            )

            probability = float(model.predict_proba(input_df)[0, 1])
            prediction = int(probability >= 0.5)

            st.markdown("---")
            st.subheader("Prediction Result")

            result_col1, result_col2 = st.columns(2)

            result_col1.metric(
                "Estimated fraud probability",
                f"{probability:.2%}",
            )

            if prediction == 1:
                result_col2.error("Model prediction: Potential fraud")
            else:
                result_col2.success("Model prediction: Likely legitimate")

            if probability < 0.30:
                risk_level = "Lower model-estimated risk"
            elif probability < 0.70:
                risk_level = "Moderate model-estimated risk"
            else:
                risk_level = "Higher model-estimated risk"

            st.info(f"Risk category: **{risk_level}**")

            st.warning(
                "Important limitation: this is not a reliable assessment "
                "of a real transaction. The V1–V28 features were set to zero, "
                "and the probability is not necessarily calibrated. Do not "
                "use this form to approve, reject, or investigate real payments."
            )

# -------------------- ABOUT PAGE --------------------

elif page == "About the Project":
    st.subheader("About FraudShield AI")

    st.markdown(
        """
        **FraudShield AI** is an educational machine-learning project
        designed to demonstrate how a classification model can help analyze
        labeled credit card transaction data.

        ### Project objectives
        - Load and validate transaction data from a CSV file.
        - Train a Random Forest classifier.
        - Explore class imbalance between legitimate and fraudulent records.
        - Evaluate the model using precision, recall, F1 score, and ROC-AUC.
        - Visualize a confusion matrix and feature importance.
        - Demonstrate an example prediction through an interactive dashboard.

        ### Technology stack
        - Python
        - Pandas and NumPy
        - Scikit-learn
        - Streamlit
        - Plotly

        ### Dataset notes
        When the Kaggle credit card fraud dataset is used, the `Class` column
        is the target label: `0` represents a legitimate transaction and
        `1` represents a fraudulent transaction. The dataset's `V1`–`V28`
        columns are transformed features; they are not ordinary human-readable
        transaction details.

        ### Limitations
        This is a portfolio and learning prototype. Its performance depends
        on the dataset, sampling, feature processing, and evaluation method.
        It has not been validated for real banking operations, and it should
        not be used as a production fraud prevention system.
        """
    )

    st.markdown("---")
    st.write("**Developer:** Abeeha Zeeshan")
    st.write("**Project:** Credit Card Fraud Detection using Machine Learning")

# -------------------- FOOTER --------------------

st.markdown("---")
st.caption(
    "FraudShield AI | Educational prototype | "
    "Predictions are experimental and must not be used for real financial decisions."
)
