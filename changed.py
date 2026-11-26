# ============================================================
# BURNINWATCH
# AI-Based Burn-In Anomaly & Degradation Detection
#
# This file is the STREAMLIT APPLICATION.
#
# IMPORTANT:
# - Do NOT replace burninwatch.py with this file.
# - burninwatch.py performs the ML analysis.
# - app.py displays and analyzes those results.
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BurnInWatch",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTS
# ============================================================

DATA_PATH = "data/synthetic_burnin_dataset.csv"

RESULTS_PATH = "outputs/burninwatch_results.csv"

FEATURES_PATH = "outputs/component_features.csv"

COMPLETE_PATH = "outputs/complete_burninwatch_dataset.csv"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .risk-box {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, possible_names):

    """
    Find a column using exact names first and
    then partial keyword matching.
    """

    if df is None or df.empty:
        return None

    lower_map = {
        str(col).lower().strip(): col
        for col in df.columns
    }

    # Exact match
    for name in possible_names:

        name_lower = name.lower()

        if name_lower in lower_map:
            return lower_map[name_lower]

    # Partial match
    for col in df.columns:

        col_lower = str(col).lower()

        for name in possible_names:

            if name.lower() in col_lower:
                return col

    return None


def detect_component_column(df):

    return find_column(
        df,
        [
            "component_id",
            "component",
            "device_id",
            "device",
            "part_id",
            "part",
            "unit_id",
            "unit"
        ]
    )


def detect_time_column(df):

    return find_column(
        df,
        [
            "burn_in_hours",
            "burnin_hours",
            "burn_in_time",
            "burnin_time",
            "burn_in",
            "time",
            "hours",
            "hour",
            "timestamp"
        ]
    )


def detect_status_column(df):

    return find_column(
        df,
        [
            "final_status",
            "status",
            "anomaly_status",
            "classification"
        ]
    )


def detect_risk_column(df):

    return find_column(
        df,
        [
            "risk_level",
            "risk",
            "risk_category"
        ]
    )


def detect_ground_truth_column(df):

    return find_column(
        df,
        [
            "ground_truth",
            "true_label",
            "actual_label",
            "fault_type",
            "fault",
            "failure",
            "defect",
            "class",
            "label"
        ]
    )


def detect_anomaly_columns(df):

    detected = []

    for col in df.columns:

        name = str(col).lower()

        if (
            "isolation" in name
            or "lof" in name
            or "svm" in name
            or "anomaly_vote" in name
            or "anomaly" in name
        ):

            detected.append(col)

    return detected


def get_status_counts(df, status_column):

    if status_column is None:
        return {}

    values = (
        df[status_column]
        .astype(str)
        .str.strip()
    )

    return values.value_counts().to_dict()


def classify_status(text):

    """
    Convert status text into a broad category.
    """

    text = str(text).lower()

    if (
        "critical" in text
        or "anomalous" in text
        or "abnormal" in text
    ):
        return "Anomalous"

    if (
        "suspicious" in text
        or "moderate" in text
        or "high" in text
    ):
        return "Suspicious"

    if (
        "normal" in text
        or "low" in text
    ):
        return "Normal"

    return "Unknown"


def calculate_percentage_change(initial, final):

    if pd.isna(initial) or pd.isna(final):
        return np.nan

    if initial == 0:
        return np.nan

    return ((final - initial) / abs(initial)) * 100


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_csv(path):

    return pd.read_csv(path)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">BurnInWatch</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Based Burn-In Anomaly & Degradation Detection System'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CHECK RESULTS
# ============================================================

if not os.path.exists(RESULTS_PATH):

    st.error(
        "ML results were not found."
    )

    st.warning(
        "Run the original burninwatch.py first."
    )

    st.code(
        "python burninwatch.py",
        language="bash"
    )

    st.stop()


# ============================================================
# LOAD RESULT FILE
# ============================================================

try:

    results = load_csv(RESULTS_PATH)

except Exception as e:

    st.error(
        f"Could not load ML results: {e}"
    )

    st.stop()


# ============================================================
# LOAD FEATURE FILE
# ============================================================

features = None

if os.path.exists(FEATURES_PATH):

    try:

        features = load_csv(FEATURES_PATH)

    except Exception:

        features = None


# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

raw_data = None

if os.path.exists(DATA_PATH):

    try:

        raw_data = load_csv(DATA_PATH)

    except Exception:

        raw_data = None


# ============================================================
# DETECT IMPORTANT COLUMNS
# ============================================================

component_column = detect_component_column(results)

status_column = detect_status_column(results)

risk_column = detect_risk_column(results)

ground_truth_column = detect_ground_truth_column(results)

anomaly_columns = detect_anomaly_columns(results)


# ============================================================
# COMPONENT COLUMN FALLBACK
# ============================================================

if component_column is None:

    if len(results.columns) > 0:

        component_column = results.columns[0]

    else:

        st.error(
            "No columns were found in the result file."
        )

        st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("BurnInWatch")

st.sidebar.markdown(
    "### Navigation"
)

page = st.sidebar.radio(
    "",
    [
        "Dashboard",
        "Component Analysis",
        "Anomaly Results",
        "Burn-In Trends",
        "Model Evaluation",
        "Dataset"
    ]
)

st.sidebar.divider()

st.sidebar.write(
    "Unsupervised ML screening system"
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("System Dashboard")

    st.write(
        """
        BurnInWatch learns normal behavioral patterns from burn-in
        measurements and identifies components whose behavior differs
        significantly from the learned population.
        """
    )

    # --------------------------------------------------------
    # BASIC STATISTICS
    # --------------------------------------------------------

    total_components = results[component_column].nunique()

    total_records = len(results)

    if status_column:

        broad_status = results[
            status_column
        ].apply(classify_status)

        normal_count = (
            broad_status == "Normal"
        ).sum()

        suspicious_count = (
            broad_status == "Suspicious"
        ).sum()

        anomalous_count = (
            broad_status == "Anomalous"
        ).sum()

    else:

        normal_count = 0
        suspicious_count = 0
        anomalous_count = 0

    # --------------------------------------------------------
    # METRIC CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Components",
        total_components
    )

    c2.metric(
        "Normal",
        int(normal_count)
    )

    c3.metric(
        "Suspicious",
        int(suspicious_count)
    )

    c4.metric(
        "Anomalous",
        int(anomalous_count)
    )

    st.divider()

    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Risk Distribution")

    if risk_column:

        risk_counts = (
            results[risk_column]
            .astype(str)
            .value_counts()
        )

        st.bar_chart(risk_counts)

    elif status_column:

        status_counts = (
            results[status_column]
            .astype(str)
            .value_counts()
        )

        st.bar_chart(status_counts)

    else:

        st.info(
            "No risk/status column was detected."
        )

    st.divider()

    # --------------------------------------------------------
    # SYSTEM WORKFLOW
    # --------------------------------------------------------

    st.subheader("BurnInWatch Workflow")

    workflow = """

    Burn-in Measurements
            ↓
    Data Cleaning
            ↓
    Feature Engineering
            ↓
    Component Behavioral Fingerprint
            ↓
    Feature Scaling
            ↓
    PCA
            ↓
    ┌─────────────────────────────┐
    │ Isolation Forest            │
    │ Local Outlier Factor        │
    │ One-Class SVM               │
    └─────────────────────────────┘
            ↓
    Anomaly Detection
            ↓
    Model Agreement
            ↓
    Risk Assessment
            ↓
    Normal / Suspicious / Anomalous

    """

    st.code(
        workflow,
        language="text"
    )


# ============================================================
# COMPONENT ANALYSIS
# ============================================================

elif page == "Component Analysis":

    st.header("Component-Level Analysis")

    component_list = sorted(
        results[component_column]
        .astype(str)
        .unique()
        .tolist()
    )

    selected_component = st.selectbox(
        "Select a component",
        component_list
    )

    selected_result = results[
        results[component_column]
        .astype(str)
        == selected_component
    ]

    if selected_result.empty:

        st.warning(
            "No result found for this component."
        )

        st.stop()

    st.subheader(
        f"Component: {selected_component}"
    )

    # --------------------------------------------------------
    # STATUS / RISK
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    if status_column:

        status_value = (
            selected_result[
                status_column
            ].iloc[0]
        )

        col1.metric(
            "Final Status",
            str(status_value)
        )

    if risk_column:

        risk_value = (
            selected_result[
                risk_column
            ].iloc[0]
        )

        col2.metric(
            "Risk Level",
            str(risk_value)
        )

    # --------------------------------------------------------
    # COMPLETE RESULT
    # --------------------------------------------------------

    st.subheader("ML Result")

    st.dataframe(
        selected_result,
        use_container_width=True
    )

    # --------------------------------------------------------
    # ANOMALY MODEL RESULTS
    # --------------------------------------------------------

    st.subheader(
        "Anomaly Detection Models"
    )

    if anomaly_columns:

        st.dataframe(
            selected_result[
                anomaly_columns
            ],
            use_container_width=True
        )

    else:

        st.info(
            "Individual model columns were not detected."
        )

    # --------------------------------------------------------
    # BEHAVIORAL FEATURES
    # --------------------------------------------------------

    st.subheader(
        "Behavioral Fingerprint"
    )

    if features is not None:

        feature_component_column = (
            detect_component_column(features)
        )

        if feature_component_column:

            selected_features = features[
                features[
                    feature_component_column
                ].astype(str)
                == selected_component
            ]

            if not selected_features.empty:

                st.dataframe(
                    selected_features,
                    use_container_width=True
                )

            else:

                st.info(
                    "No feature record found for this component."
                )

        else:

            st.dataframe(
                features.head(1),
                use_container_width=True
            )

    else:

        st.warning(
            "component_features.csv was not found."
        )


# ============================================================
# ANOMALY RESULTS
# ============================================================

elif page == "Anomaly Results":

    st.header("Anomaly Detection Results")

    st.write(
        """
        The following table contains the component-level results
        generated by the unsupervised ML pipeline.
        """
    )

    display_data = results.copy()

    # --------------------------------------------------------
    # STATUS FILTER
    # --------------------------------------------------------

    if status_column:

        available_statuses = sorted(
            display_data[
                status_column
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        selected_statuses = st.multiselect(
            "Filter by final status",
            available_statuses,
            default=available_statuses
        )

        display_data = display_data[
            display_data[
                status_column
            ]
            .astype(str)
            .isin(selected_statuses)
        ]

    # --------------------------------------------------------
    # RISK FILTER
    # --------------------------------------------------------

    if risk_column:

        available_risks = sorted(
            display_data[
                risk_column
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        selected_risks = st.multiselect(
            "Filter by risk level",
            available_risks,
            default=available_risks
        )

        display_data = display_data[
            display_data[
                risk_column
            ]
            .astype(str)
            .isin(selected_risks)
        ]

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    st.write(
        f"Showing {len(display_data)} records."
    )

    st.dataframe(
        display_data,
        use_container_width=True,
        height=550
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    csv_data = display_data.to_csv(
        index=False
    )

    st.download_button(
        label="Download Anomaly Report",
        data=csv_data,
        file_name="burninwatch_anomaly_report.csv",
        mime="text/csv"
    )


# ============================================================
# BURN-IN TRENDS
# ============================================================

elif page == "Burn-In Trends":

    st.header(
        "Burn-In Measurement Trends"
    )

    if raw_data is None:

        st.error(
            "Original dataset was not found."
        )

        st.stop()

    raw_component_column = (
        detect_component_column(raw_data)
    )

    raw_time_column = (
        detect_time_column(raw_data)
    )

    if raw_component_column is None:

        st.error(
            "Could not identify the component ID column."
        )

        st.write(
            "Available columns:"
        )

        st.write(
            list(raw_data.columns)
        )

        st.stop()

    if raw_time_column is None:

        st.error(
            "Could not identify the burn-in time column."
        )

        st.write(
            "Available columns:"
        )

        st.write(
            list(raw_data.columns)
        )

        st.stop()

    # --------------------------------------------------------
    # SELECT COMPONENT
    # --------------------------------------------------------

    components = sorted(
        raw_data[
            raw_component_column
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    selected_component = st.selectbox(
        "Select component",
        components
    )

    component_data = raw_data[
        raw_data[
            raw_component_column
        ].astype(str)
        == selected_component
    ].copy()

    # --------------------------------------------------------
    # CONVERT TIME
    # --------------------------------------------------------

    component_data["_time_numeric"] = pd.to_numeric(
        component_data[
            raw_time_column
        ],
        errors="coerce"
    )

    component_data = component_data.dropna(
        subset=["_time_numeric"]
    )

    component_data = component_data.sort_values(
        "_time_numeric"
    )

    if component_data.empty:

        st.warning(
            "No valid numeric burn-in time values found."
        )

        st.stop()

    # --------------------------------------------------------
    # MEASUREMENT COLUMNS
    # --------------------------------------------------------

    excluded = {
        raw_component_column,
        raw_time_column,
        "_time_numeric"
    }

    measurement_columns = []

    for col in component_data.columns:

        if col in excluded:
            continue

        if pd.api.types.is_numeric_dtype(
            component_data[col]
        ):

            measurement_columns.append(col)

    # Remove likely labels
    measurement_columns = [
        col
        for col in measurement_columns
        if not any(
            word in str(col).lower()
            for word in [
                "label",
                "fault",
                "failure",
                "defect",
                "class",
                "status",
                "target"
            ]
        )
    ]

    if not measurement_columns:

        st.warning(
            "No numeric measurement columns were detected."
        )

        st.stop()

    # --------------------------------------------------------
    # SELECT PARAMETER
    # --------------------------------------------------------

    selected_parameter = st.selectbox(
        "Select burn-in parameter",
        measurement_columns
    )

    values = pd.to_numeric(
        component_data[
            selected_parameter
        ],
        errors="coerce"
    )

    valid = pd.DataFrame(
        {
            "Time": component_data[
                "_time_numeric"
            ],
            "Value": values
        }
    ).dropna()

    if valid.empty:

        st.warning(
            "No valid measurements available."
        )

        st.stop()

    # --------------------------------------------------------
    # LINE GRAPH
    # --------------------------------------------------------

    st.subheader(
        f"{selected_parameter} vs Burn-In Time"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        valid["Time"],
        valid["Value"],
        marker="o"
    )

    ax.set_xlabel(
        "Burn-in time (hours)"
    )

    ax.set_ylabel(
        selected_parameter
    )

    ax.set_title(
        f"{selected_component} - {selected_parameter}"
    )

    ax.grid(True)

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    # --------------------------------------------------------
    # DEGRADATION STATISTICS
    # --------------------------------------------------------

    st.subheader(
        "Degradation Indicators"
    )

    initial_value = valid[
        "Value"
    ].iloc[0]

    final_value = valid[
        "Value"
    ].iloc[-1]

    absolute_change = (
        final_value - initial_value
    )

    percentage_change = calculate_percentage_change(
        initial_value,
        final_value
    )

    if len(valid) >= 2:

        slope = np.polyfit(
            valid["Time"],
            valid["Value"],
            1
        )[0]

    else:

        slope = np.nan

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Initial Value",
        f"{initial_value:.4f}"
    )

    c2.metric(
        "Final Value",
        f"{final_value:.4f}"
    )

    c3.metric(
        "Absolute Change",
        f"{absolute_change:.4f}"
    )

    if pd.notna(percentage_change):

        c4.metric(
            "% Change",
            f"{percentage_change:.2f}%"
        )

    else:

        c4.metric(
            "% Change",
            "N/A"
        )

    st.write(
        f"Estimated trend slope: **{slope:.6f} per hour**"
    )

    # --------------------------------------------------------
    # ENGINEERING INTERPRETATION
    # --------------------------------------------------------

    st.subheader(
        "Behavior Interpretation"
    )

    if pd.isna(slope):

        st.info(
            "Not enough time points to estimate a trend."
        )

    elif slope > 0:

        st.warning(
            "The selected parameter shows an increasing trend "
            "during burn-in. This may indicate degradation or "
            "behavioral drift and should be considered together "
            "with the ML anomaly result."
        )

    elif slope < 0:

        st.info(
            "The selected parameter shows a decreasing trend "
            "during burn-in."
        )

    else:

        st.success(
            "The selected parameter is approximately stable "
            "during burn-in."
        )


# ============================================================
# MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.header(
        "Unsupervised Model Evaluation"
    )

    st.write(
        """
        Ground-truth labels are NOT used to train the anomaly
        detection models. If the dataset contains an independent
        ground-truth column, it can be used here only to evaluate
        the resulting predictions.
        """
    )

    if ground_truth_column is None:

        st.info(
            "No ground-truth column was detected in the result file."
        )

        st.write(
            "Available columns:"
        )

        st.write(
            list(results.columns)
        )

        st.stop()

    st.success(
        f"Ground-truth column detected: {ground_truth_column}"
    )

    # --------------------------------------------------------
    # FIND POSSIBLE PREDICTION COLUMNS
    # --------------------------------------------------------

    prediction_columns = []

    for col in results.columns:

        name = str(col).lower()

        if (
            "isolation" in name
            or "lof" in name
            or "svm" in name
        ):

            prediction_columns.append(col)

    if not prediction_columns:

        st.warning(
            "No individual model prediction columns were detected."
        )

        st.stop()

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    y_true_raw = (
        results[ground_truth_column]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Broad binary abnormal/normal conversion
    def ground_truth_binary(value):

        abnormal_words = [
            "fault",
            "failure",
            "defect",
            "abnormal",
            "anomaly",
            "anomalous",
            "gradual",
            "sudden",
            "latent",
            "bad"
        ]

        normal_words = [
            "normal",
            "good",
            "healthy"
        ]

        for word in abnormal_words:

            if word in value:
                return 1

        for word in normal_words:

            if word in value:
                return 0

        return np.nan

    y_true = y_true_raw.apply(
        ground_truth_binary
    )

    # --------------------------------------------------------
    # MODEL EVALUATION
    # --------------------------------------------------------

    evaluation_rows = []

    for prediction_column in prediction_columns:

        values = results[
            prediction_column
        ]

        # Convert common anomaly formats
        def convert_prediction(value):

            value_string = str(value).lower()

            if (
                "anomaly" in value_string
                or "outlier" in value_string
                or "abnormal" in value_string
            ):
                return 1

            if "normal" in value_string:
                return 0

            try:

                numeric = float(value)

                # Existing pipeline commonly uses:
                # 1 = anomaly
                # -1 = anomaly
                # 0 = normal

                if numeric == -1:
                    return 1

                if numeric == 1:
                    return 1

                if numeric == 0:
                    return 0

            except:

                pass

            return np.nan

        y_pred = values.apply(
            convert_prediction
        )

        valid_mask = (
            y_true.notna()
            & y_pred.notna()
        )

        if valid_mask.sum() == 0:

            continue

        yt = y_true[
            valid_mask
        ].astype(int)

        yp = y_pred[
            valid_mask
        ].astype(int)

        evaluation_rows.append(
            {
                "Model": prediction_column,
                "Accuracy": accuracy_score(
                    yt,
                    yp
                ),
                "Precision": precision_score(
                    yt,
                    yp,
                    zero_division=0
                ),
                "Recall": recall_score(
                    yt,
                    yp,
                    zero_division=0
                ),
                "F1 Score": f1_score(
                    yt,
                    yp,
                    zero_division=0
                )
            }
        )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    if not evaluation_rows:

        st.warning(
            "Could not calculate evaluation metrics automatically. "
            "Check the ground-truth and prediction column formats."
        )

    else:

        evaluation_df = pd.DataFrame(
            evaluation_rows
        )

        st.dataframe(
            evaluation_df.style.format(
                {
                    "Accuracy": "{:.3f}",
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "F1 Score": "{:.3f}"
                }
            ),
            use_container_width=True
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        best_row = evaluation_df.loc[
            evaluation_df["F1 Score"].idxmax()
        ]

        st.success(
            f"Best F1-score among evaluated models: "
            f"{best_row['Model']} "
            f"({best_row['F1 Score']:.3f})"
        )


# ============================================================
# DATASET
# ============================================================

elif page == "Dataset":

    st.header(
        "Dataset Explorer"
    )

    if raw_data is None:

        st.error(
            "Original dataset was not found."
        )

        st.stop()

    # --------------------------------------------------------
    # DATASET INFO
    # --------------------------------------------------------

    rows = raw_data.shape[0]

    columns = raw_data.shape[1]

    missing_values = (
        raw_data.isnull()
        .sum()
        .sum()
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Rows",
        rows
    )

    c2.metric(
        "Columns",
        columns
    )

    c3.metric(
        "Missing Values",
        int(missing_values)
    )

    st.divider()

    # --------------------------------------------------------
    # DATASET PREVIEW
    # --------------------------------------------------------

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        raw_data,
        use_container_width=True,
        height=500
    )

    # --------------------------------------------------------
    # COLUMN INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "Dataset Columns"
    )

    column_info = pd.DataFrame(
        {
            "Column": raw_data.columns,
            "Data Type": [
                str(dtype)
                for dtype in raw_data.dtypes
            ],
            "Missing Values": [
                int(raw_data[col].isnull().sum())
                for col in raw_data.columns
            ],
            "Unique Values": [
                int(raw_data[col].nunique())
                for col in raw_data.columns
            ]
        }
    )

    st.dataframe(
        column_info,
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "BurnInWatch | Unsupervised ML-Based Burn-In "
    "Anomaly & Degradation Detection"
)