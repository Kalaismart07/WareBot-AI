# ml_models.py
# WareBot AI - Machine Learning Engine

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# 1. ROBOT TELEMETRY GENERATOR
# ============================================================

def generate_telemetry(num_robots=20, records_per_robot=100):
    """
    Generate simulated real-time telemetry for warehouse robots.
    """

    np.random.seed(42)

    rows = []

    for robot_id in range(1, num_robots + 1):

        battery = np.random.uniform(40, 100)

        for record in range(records_per_robot):

            battery = max(
                5,
                battery - np.random.uniform(0.05, 0.30)
            )

            motor_current = np.random.normal(8, 1.2)
            temperature = np.random.normal(42, 5)
            vibration = np.random.normal(2.5, 0.7)
            speed = np.random.normal(1.5, 0.25)
            operating_hours = record * 0.15
            navigation_errors = np.random.poisson(1)

            rows.append({
                "robot_id": f"R{robot_id:02d}",
                "timestamp": record,
                "battery": round(battery, 2),
                "motor_current": round(motor_current, 2),
                "temperature": round(temperature, 2),
                "vibration": round(vibration, 2),
                "speed": round(speed, 2),
                "operating_hours": round(operating_hours, 2),
                "navigation_errors": navigation_errors
            })

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Simulate abnormal robot behaviour
    # --------------------------------------------------------

    anomaly_count = int(len(df) * 0.08)

    anomaly_indices = np.random.choice(
        df.index,
        size=anomaly_count,
        replace=False
    )

    df.loc[anomaly_indices, "motor_current"] *= np.random.uniform(
        1.4, 2.0, anomaly_count
    )

    df.loc[anomaly_indices, "temperature"] += np.random.uniform(
        10, 20, anomaly_count
    )

    df.loc[anomaly_indices, "vibration"] *= np.random.uniform(
        1.5, 2.5, anomaly_count
    )

    df.loc[anomaly_indices, "navigation_errors"] += np.random.randint(
        3, 8, anomaly_count
    )

    # --------------------------------------------------------
    # Maintenance risk label
    # --------------------------------------------------------

    df["maintenance_risk"] = (
        (df["battery"] < 25) |
        (df["motor_current"] > 12) |
        (df["temperature"] > 55) |
        (df["vibration"] > 4) |
        (df["navigation_errors"] > 4)
    ).astype(int)

    return df


# ============================================================
# 2. RULE-BASED ANOMALY SCORE
# ============================================================

def calculate_anomaly_score(df):
    """
    Calculate an anomaly score from robot telemetry.
    """

    features = [
        "battery",
        "motor_current",
        "temperature",
        "vibration",
        "speed",
        "operating_hours",
        "navigation_errors"
    ]

    scaler = StandardScaler()

    X = scaler.fit_transform(df[features])

    # Distance from normal behaviour
    anomaly_score = np.sqrt(np.sum(X ** 2, axis=1))

    df = df.copy()

    df["anomaly_score"] = np.round(anomaly_score, 3)

    threshold = df["anomaly_score"].quantile(0.95)

    df["anomaly"] = (
        df["anomaly_score"] > threshold
    ).astype(int)

    return df, threshold


# ============================================================
# 3. PREDICTIVE MAINTENANCE MODEL
# ============================================================

def train_maintenance_model(df):
    """
    Train a Random Forest model to predict maintenance risk.
    """

    features = [
        "battery",
        "motor_current",
        "temperature",
        "vibration",
        "speed",
        "operating_hours",
        "navigation_errors"
    ]

    X = df[features]
    y = df["maintenance_risk"]

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X, y)

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)

    return model, accuracy, features


# ============================================================
# 4. ROBOT HEALTH SCORE
# ============================================================

def calculate_health_score(row):
    """
    Convert telemetry into a simple robot health score.
    """

    score = 100

    if row["battery"] < 30:
        score -= 20

    if row["motor_current"] > 12:
        score -= 20

    if row["temperature"] > 55:
        score -= 20

    if row["vibration"] > 4:
        score -= 20

    if row["navigation_errors"] > 4:
        score -= 15

    return max(0, min(100, score))


# ============================================================
# 5. ROBOT STATUS
# ============================================================

def get_robot_status(health_score):
    """
    Convert health score into operational status.
    """

    if health_score >= 80:
        return "Healthy"

    elif health_score >= 60:
        return "Warning"

    else:
        return "Critical"


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    telemetry = generate_telemetry()

    telemetry, threshold = calculate_anomaly_score(
        telemetry
    )

    model, accuracy, features = train_maintenance_model(
        telemetry
    )

    telemetry["health_score"] = telemetry.apply(
        calculate_health_score,
        axis=1
    )

    telemetry["status"] = telemetry["health_score"].apply(
        get_robot_status
    )

    print("\n===================================")
    print("       WAREBOT AI ML ENGINE")
    print("===================================")

    print(f"\nTotal telemetry records : {len(telemetry)}")
    print(f"Anomaly threshold       : {threshold:.3f}")
    print(f"Anomalies detected      : {telemetry['anomaly'].sum()}")
    print(f"Maintenance accuracy    : {accuracy * 100:.2f}%")

    print("\nRobot Health Preview:")
    print(
        telemetry[
            [
                "robot_id",
                "battery",
                "temperature",
                "vibration",
                "health_score",
                "status"
            ]
        ].tail(10)
    )

    telemetry.to_csv("data/robot_telemetry.csv", index=False)

    print("\nTelemetry saved successfully!")