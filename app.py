# ============================================================
# BURNINWATCH
# Unsupervised Component Screening & Anomaly Detection
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_PATH = "data/synthetic_burnin_dataset.csv"

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("BURNINWATCH - COMPONENT ANOMALY DETECTION")
print("=" * 70)

print("\n[1] Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

print("\nColumn names:")
for column in df.columns:
    print(" -", column)


# ============================================================
# 3. BASIC DATA INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("[2] DATASET INFORMATION")
print("=" * 70)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe(include="all").T)


# ============================================================
# 4. CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

print("\nCleaned column names:")
print(df.columns.tolist())


# ============================================================
# 5. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("[3] MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

print(missing)

# Fill numeric missing values using median
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].median())

# Fill categorical missing values using mode
categorical_columns = df.select_dtypes(exclude=np.number).columns

for column in categorical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].mode()[0])


# ============================================================
# 6. REMOVE DUPLICATES
# ============================================================

duplicates = df.duplicated().sum()

print("\nDuplicate rows:", duplicates)

if duplicates > 0:
    df = df.drop_duplicates()

print("Rows after duplicate removal:", len(df))


# ============================================================
# 7. IDENTIFY IMPORTANT COLUMNS AUTOMATICALLY
# ============================================================

print("\n" + "=" * 70)
print("[4] IDENTIFYING IMPORTANT COLUMNS")
print("=" * 70)


def find_column(possible_names, columns):
    """
    Finds a column whose name contains one of the keywords.
    """

    for keyword in possible_names:

        for column in columns:

            if keyword in column.lower():
                return column

    return None


# Possible component ID columns
component_column = find_column(
    [
        "component_id",
        "component",
        "device_id",
        "device",
        "part_id",
        "part"
    ],
    df.columns
)


# Possible time columns
time_column = find_column(
    [
        "burn_in_hours",
        "burnin_hours",
        "burn_in_time",
        "burnin_time",
        "time",
        "hour",
        "hours",
        "timestamp"
    ],
    df.columns
)


print("Component column:", component_column)
print("Time column     :", time_column)


# ============================================================
# 8. IF COMPONENT ID IS NOT FOUND
# ============================================================

if component_column is None:

    print("\nWARNING:")
    print("Could not automatically identify the component ID column.")
    print("The first non-numeric column will be used.")

    non_numeric = df.select_dtypes(exclude=np.number).columns

    if len(non_numeric) > 0:
        component_column = non_numeric[0]

    else:
        # Create artificial component IDs
        df["component_id"] = np.arange(len(df))
        component_column = "component_id"


print("Using component column:", component_column)


# ============================================================
# 9. IDENTIFY NUMERIC MEASUREMENT COLUMNS
# ============================================================

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

print("\nNumeric columns:")
for column in numeric_columns:
    print(" -", column)


# Remove obvious ID/time columns from measurements

measurement_columns = numeric_columns.copy()

if component_column in measurement_columns:
    measurement_columns.remove(component_column)

if time_column in measurement_columns:
    measurement_columns.remove(time_column)


# Remove possible label columns from ML features
# We DO NOT use labels to train the unsupervised models.

possible_label_words = [
    "label",
    "fault",
    "failure",
    "defect",
    "class",
    "status",
    "target",
    "ground_truth"
]

label_columns = []

for column in measurement_columns:

    if any(word in column.lower() for word in possible_label_words):
        label_columns.append(column)

for column in label_columns:
    measurement_columns.remove(column)


print("\nMeasurement columns used for ML:")

for column in measurement_columns:
    print(" -", column)


# ============================================================
# 10. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("[5] EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Histogram of numerical measurements

for column in measurement_columns:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df[column],
        kde=True
    )

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/distribution_{column}.png"
    )

    plt.show()


# ============================================================
# 11. CORRELATION MATRIX
# ============================================================

if len(measurement_columns) > 1:

    plt.figure(
        figsize=(
            max(8, len(measurement_columns)),
            max(6, len(measurement_columns))
        )
    )

    correlation = df[measurement_columns].corr()

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Between Burn-in Measurements")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/correlation_matrix.png"
    )

    plt.show()


# ============================================================
# 12. BURN-IN TIME-SERIES VISUALIZATION
# ============================================================

print("\n" + "=" * 70)
print("[6] BURN-IN TIME-SERIES ANALYSIS")
print("=" * 70)


if component_column is not None and time_column is not None:

    unique_components = df[component_column].unique()

    # Plot first 5 components
    selected_components = unique_components[:5]

    for measurement in measurement_columns:

        plt.figure(figsize=(10, 6))

        for component in selected_components:

            component_data = df[
                df[component_column] == component
            ].sort_values(time_column)

            plt.plot(
                component_data[time_column],
                component_data[measurement],
                marker="o",
                label=str(component)
            )

        plt.title(
            f"{measurement} During Burn-in"
        )

        plt.xlabel(time_column)
        plt.ylabel(measurement)

        plt.legend()

        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            f"{OUTPUT_DIR}/burnin_{measurement}.png"
        )

        plt.show()

else:

    print(
        "\nTime/component columns were not detected."
    )

    print(
        "Skipping time-series visualization."
    )


# ============================================================
# 13. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("[7] FEATURE ENGINEERING")
print("=" * 70)


def calculate_slope(group, time_col, value_col):

    """
    Calculates linear trend/slope.
    """

    x = group[time_col].values
    y = group[value_col].values

    # Need at least 2 different time values
    if len(np.unique(x)) < 2:
        return 0

    slope = np.polyfit(x, y, 1)[0]

    return slope


features = []


if component_column is not None and time_column is not None:

    grouped = df.groupby(component_column)

    for component_id, group in grouped:

        group = group.sort_values(time_column)

        row = {
            "component_id": component_id
        }

        for column in measurement_columns:

            values = group[column].values

            # Basic statistical features

            row[f"{column}_mean"] = np.mean(values)

            row[f"{column}_std"] = np.std(values)

            row[f"{column}_min"] = np.min(values)

            row[f"{column}_max"] = np.max(values)

            # Initial and final values

            row[f"{column}_initial"] = values[0]

            row[f"{column}_final"] = values[-1]

            # Absolute change

            row[f"{column}_change"] = (
                values[-1] - values[0]
            )

            # Percentage change

            initial_value = values[0]

            if abs(initial_value) > 1e-10:

                row[f"{column}_percentage_change"] = (
                    (values[-1] - values[0])
                    / abs(initial_value)
                ) * 100

            else:

                row[f"{column}_percentage_change"] = 0

            # Trend / slope

            row[f"{column}_slope"] = calculate_slope(
                group,
                time_column,
                column
            )

        features.append(row)


else:

    print(
        "\nComponent/time structure unavailable."
    )

    print(
        "Creating features from the available numeric measurements."
    )

    # Fallback: treat each row as one observation

    feature_df = df[measurement_columns].copy()

    feature_df["component_id"] = np.arange(
        len(feature_df)
    )


# Convert feature list into DataFrame

if len(features) > 0:

    feature_df = pd.DataFrame(features)


print("\nComponent-level feature dataset:")
print(feature_df.head())

print("\nFeature dataset shape:")
print(feature_df.shape)


# ============================================================
# 14. SAVE FEATURE DATASET
# ============================================================

feature_df.to_csv(
    f"{OUTPUT_DIR}/component_features.csv",
    index=False
)

print(
    "\nFeature dataset saved to:"
    f" {OUTPUT_DIR}/component_features.csv"
)


# ============================================================
# 15. PREPARE DATA FOR MACHINE LEARNING
# ============================================================

print("\n" + "=" * 70)
print("[8] PREPARING DATA FOR MACHINE LEARNING")
print("=" * 70)


# Keep component IDs separately

component_ids = feature_df["component_id"]


# Select only numerical features

X = feature_df.drop(
    columns=["component_id"]
)

X = X.select_dtypes(
    include=np.number
)


# Replace infinite values

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# Fill remaining missing values

X = X.fillna(
    X.median()
)


print("Number of ML features:", X.shape[1])


# ============================================================
# 16. STANDARD SCALING
# ============================================================

print("\n[9] STANDARD SCALING")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Scaling completed.")


# Save scaler

joblib.dump(
    scaler,
    f"{MODEL_DIR}/scaler.pkl"
)


# ============================================================
# 17. PCA
# ============================================================

print("\n" + "=" * 70)
print("[10] PCA")
print("=" * 70)


# PCA for visualization

pca = PCA(
    n_components=2,
    random_state=42
)

X_pca = pca.fit_transform(X_scaled)

print(
    "Explained variance ratio:",
    pca.explained_variance_ratio_
)

print(
    "Total explained variance:",
    pca.explained_variance_ratio_.sum()
)


# Save PCA

joblib.dump(
    pca,
    f"{MODEL_DIR}/pca.pkl"
)


# ============================================================
# 18. PCA VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    "BurnInWatch - PCA Component Distribution"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/pca_distribution.png"
)

plt.show()


# ============================================================
# 19. ISOLATION FOREST
# ============================================================

print("\n" + "=" * 70)
print("[11] ISOLATION FOREST")
print("=" * 70)


isolation_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

isolation_forest.fit(X_scaled)


iso_prediction = isolation_forest.predict(
    X_scaled
)

iso_score = isolation_forest.decision_function(
    X_scaled
)


# Convert predictions

iso_result = np.where(
    iso_prediction == -1,
    "Anomaly",
    "Normal"
)


print(
    "Anomalies detected:",
    np.sum(iso_prediction == -1)
)

print(
    "Normal components:",
    np.sum(iso_prediction == 1)
)


# Save model

joblib.dump(
    isolation_forest,
    f"{MODEL_DIR}/isolation_forest.pkl"
)


# ============================================================
# 20. LOCAL OUTLIER FACTOR
# ============================================================

print("\n" + "=" * 70)
print("[12] LOCAL OUTLIER FACTOR")
print("=" * 70)


# Make sure n_neighbors is valid

n_samples = len(X_scaled)

n_neighbors = min(
    20,
    max(2, n_samples - 1)
)


lof = LocalOutlierFactor(
    n_neighbors=n_neighbors,
    contamination=0.05
)

lof_prediction = lof.fit_predict(
    X_scaled
)

lof_score = lof.negative_outlier_factor_


lof_result = np.where(
    lof_prediction == -1,
    "Anomaly",
    "Normal"
)


print(
    "Anomalies detected:",
    np.sum(lof_prediction == -1)
)

print(
    "Normal components:",
    np.sum(lof_prediction == 1)
)


# ============================================================
# 21. ONE-CLASS SVM
# ============================================================

print("\n" + "=" * 70)
print("[13] ONE-CLASS SVM")
print("=" * 70)


one_class_svm = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.05
)

one_class_svm.fit(X_scaled)


svm_prediction = one_class_svm.predict(
    X_scaled
)

svm_score = one_class_svm.decision_function(
    X_scaled
)


svm_result = np.where(
    svm_prediction == -1,
    "Anomaly",
    "Normal"
)


print(
    "Anomalies detected:",
    np.sum(svm_prediction == -1)
)

print(
    "Normal components:",
    np.sum(svm_prediction == 1)
)


# Save SVM

joblib.dump(
    one_class_svm,
    f"{MODEL_DIR}/one_class_svm.pkl"
)


# ============================================================
# 22. CREATE FINAL RESULTS TABLE
# ============================================================

print("\n" + "=" * 70)
print("[14] MODEL COMPARISON")
print("=" * 70)


results = pd.DataFrame({

    "component_id": component_ids,

    "isolation_forest": iso_result,

    "lof": lof_result,

    "one_class_svm": svm_result,

    "isolation_score": iso_score,

    "lof_score": lof_score,

    "svm_score": svm_score
})


# ============================================================
# 23. MODEL AGREEMENT
# ============================================================


results["anomaly_votes"] = (

    (results["isolation_forest"] == "Anomaly").astype(int)

    +

    (results["lof"] == "Anomaly").astype(int)

    +

    (results["one_class_svm"] == "Anomaly").astype(int)

)


# ============================================================
# 24. SCREENING RISK
# ============================================================


def calculate_risk(votes):

    if votes == 0:

        return "Low Risk"

    elif votes == 1:

        return "Moderate Risk"

    elif votes == 2:

        return "High Risk"

    else:

        return "Critical Risk"


results["risk_level"] = (
    results["anomaly_votes"]
    .apply(calculate_risk)
)


# ============================================================
# 25. FINAL STATUS
# ============================================================


def calculate_status(votes):

    if votes == 0:

        return "Normal"

    elif votes == 1:

        return "Suspicious"

    else:

        return "Anomalous"


results["final_status"] = (
    results["anomaly_votes"]
    .apply(calculate_status)
)


# ============================================================
# 26. ADD PCA VALUES
# ============================================================

results["PC1"] = X_pca[:, 0]

results["PC2"] = X_pca[:, 1]


# ============================================================
# 27. DISPLAY RESULTS
# ============================================================

print("\nFinal screening results:")

print(
    results.head(20).to_string(
        index=False
    )
)


# ============================================================
# 28. MODEL COMPARISON SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("[15] ANOMALY SUMMARY")
print("=" * 70)


print(
    "\nIsolation Forest anomalies:",
    (results["isolation_forest"] == "Anomaly").sum()
)

print(
    "LOF anomalies:",
    (results["lof"] == "Anomaly").sum()
)

print(
    "One-Class SVM anomalies:",
    (results["one_class_svm"] == "Anomaly").sum()
)


print("\nRisk distribution:")

print(
    results["risk_level"].value_counts()
)


print("\nFinal status distribution:")

print(
    results["final_status"].value_counts()
)


# ============================================================
# 29. PCA WITH ANOMALY STATUS
# ============================================================

plt.figure(figsize=(10, 7))


normal = results[
    results["final_status"] == "Normal"
]

suspicious = results[
    results["final_status"] == "Suspicious"
]

anomalous = results[
    results["final_status"] == "Anomalous"
]


plt.scatter(
    normal["PC1"],
    normal["PC2"],
    label="Normal",
    alpha=0.6
)


plt.scatter(
    suspicious["PC1"],
    suspicious["PC2"],
    label="Suspicious",
    alpha=0.8
)


plt.scatter(
    anomalous["PC1"],
    anomalous["PC2"],
    marker="x",
    s=80,
    label="Anomalous"
)


plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.title(
    "BurnInWatch - Unsupervised Anomaly Detection"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/final_anomaly_pca.png"
)

plt.show()


# ============================================================
# 30. SAVE FINAL RESULTS
# ============================================================

results.to_csv(
    f"{OUTPUT_DIR}/burninwatch_results.csv",
    index=False
)


print(
    "\nFinal results saved to:"
    " outputs/burninwatch_results.csv"
)


# ============================================================
# 31. SAVE COMPLETE FEATURE + RESULT DATASET
# ============================================================

complete_results = feature_df.copy()

complete_results = complete_results.merge(
    results,
    on="component_id",
    how="left"
)


complete_results.to_csv(
    f"{OUTPUT_DIR}/complete_burninwatch_dataset.csv",
    index=False
)


# ============================================================
# 32. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("BURNINWATCH PIPELINE COMPLETED")
print("=" * 70)

print("\nGenerated files:")

print("\nOutputs:")
print(" - component_features.csv")
print(" - burninwatch_results.csv")
print(" - complete_burninwatch_dataset.csv")
print(" - PCA visualization")
print(" - burn-in measurement plots")
print(" - correlation matrix")
print(" - distribution plots")

print("\nModels:")
print(" - scaler.pkl")
print(" - pca.pkl")
print(" - isolation_forest.pkl")
print(" - one_class_svm.pkl")

print("\nNext stage:")
print("Build the Streamlit BurnInWatch dashboard.")