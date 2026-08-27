import math
import pandas as pd
from scipy.stats import ks_2samp

# ============================================================
# BASELINE DATA
# ============================================================

baseline_df = pd.read_csv("model/titanic.csv")

# Keep only numeric columns for drift detection
baseline_df = baseline_df[
    ["Pclass", "Age", "SibSp", "Parch", "Fare"]
]

# ============================================================
# PRODUCTION REQUEST BUFFER
# ============================================================

production_data = []


# ============================================================
# LOG INCOMING REQUESTS
# ============================================================

def log_request(data: dict):
    """
    Store incoming prediction requests
    for drift analysis.
    """
    production_data.append(data)


# ============================================================
# KS DRIFT DETECTOR
# ============================================================

def detect_drift():
    """
    Compare production data against the
    baseline training data using the
    Kolmogorov-Smirnov Test.
    """

    if len(production_data) < 10:
        return {
            "drift_detected": False,
            "message": "Not enough production data collected.",
            "samples_received": len(production_data)
        }

    production_df = pd.DataFrame(production_data)

    drifted_features = []
    statistics = {}

    for column in baseline_df.columns:

        baseline_values = baseline_df[column].dropna()
        production_values = production_df[column].dropna()

        # Skip if not enough production samples
        if len(production_values) < 2:
            statistics[column] = {
                "ks_statistic": 0.0,
                "p_value": 1.0
            }
            continue

        statistic, p_value = ks_2samp(
            baseline_values,
            production_values
        )

        # Prevent NaN values from breaking JSON serialization
        if math.isnan(statistic):
            statistic = 0.0

        if math.isnan(p_value):
            p_value = 1.0

        statistics[column] = {
            "ks_statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 4)
        }

        if p_value < 0.05:
            drifted_features.append(column)

    return {
        "drift_detected": len(drifted_features) > 0,
        "drifted_features": drifted_features,
        "statistics": statistics,
        "production_samples": len(production_df)
    }