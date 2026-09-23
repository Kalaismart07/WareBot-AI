# api.py
# WareBot AI - FastAPI Backend

from fastapi import FastAPI

import shap
import json
import threading
from rl_task_allocator import WarehouseTaskAllocator
from optimization import optimize_rl_robot

import paho.mqtt.client as mqtt

from ml_models import (
    generate_telemetry,
    calculate_anomaly_score,
    train_maintenance_model,
    calculate_health_score,
    get_robot_status,
)

from wms import (
    get_inventory,
    get_orders,
    get_low_stock_items,
    inventory_summary,
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="WareBot AI API",
    description="Warehouse Robotics Monitoring and Optimization API",
    version="1.0.0",
)


# ============================================================
# MQTT CONFIGURATION
# ============================================================

MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "warebot/robots/telemetry"

mqtt_telemetry = []

mqtt_lock = threading.Lock()


# ============================================================
# MQTT CONNECT
# ============================================================

def mqtt_on_connect(
    client,
    userdata,
    flags,
    reason_code,
    properties=None
):

    if reason_code == 0:

        print("===================================")
        print("       WAREBOT AI MQTT")
        print("===================================")
        print("MQTT Status : Connected")
        print(f"Broker      : {MQTT_BROKER}")
        print(f"Topic       : {MQTT_TOPIC}")
        print("-----------------------------------")

        client.subscribe(MQTT_TOPIC)

        print("MQTT telemetry listener started.")

    else:

        print(
            f"MQTT connection failed: {reason_code}"
        )


# ============================================================
# MQTT MESSAGE RECEIVER
# ============================================================

def mqtt_on_message(
    client,
    userdata,
    message
):

    try:

        payload = message.payload.decode(
            "utf-8"
        )

        telemetry = json.loads(payload)

        with mqtt_lock:

            mqtt_telemetry.append(
                telemetry
            )

            # Keep latest 500 messages
            if len(mqtt_telemetry) > 500:

                mqtt_telemetry.pop(0)

        print(
            f"MQTT Telemetry Received | "
            f"{telemetry.get('robot_id')} | "
            f"Battery: {telemetry.get('battery')}% | "
            f"Motor: {telemetry.get('motor_current')}A"
        )

    except Exception as e:

        print(
            f"MQTT telemetry error: {e}"
        )


# ============================================================
# START MQTT LISTENER
# ============================================================

def start_mqtt():

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="warebot-fastapi"
    )

    client.on_connect = mqtt_on_connect

    client.on_message = mqtt_on_message

    try:

        client.connect(
            MQTT_BROKER,
            MQTT_PORT,
            60
        )

        client.loop_start()

        print(
            "WareBot MQTT listener started."
        )

    except Exception as e:

        print(
            f"MQTT startup error: {e}"
        )


# ============================================================
# START MQTT IN BACKGROUND
# ============================================================

mqtt_thread = threading.Thread(
    target=start_mqtt,
    daemon=True
)

mqtt_thread.start()


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {

        "project": "WareBot AI",

        "status": "Online",

        "message":
        "Warehouse Robotics AI Backend is running."

    }


# ============================================================
# MQTT TELEMETRY
# ============================================================

@app.get("/mqtt/telemetry")
def mqtt_data():

    with mqtt_lock:

        latest_data = list(
            mqtt_telemetry[-50:]
        )

    return {

        "source": "MQTT",

        "broker": MQTT_BROKER,

        "topic": MQTT_TOPIC,

        "records": len(latest_data),

        "data": latest_data

    }


# ============================================================
# ROBOT TELEMETRY
# ============================================================

@app.get("/robots")
def robot_data():

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

    telemetry["status"] = telemetry[
        "health_score"
    ].apply(
        get_robot_status
    )

    latest = (

        telemetry

        .sort_values("timestamp")

        .groupby("robot_id")

        .tail(1)

    )

    return latest.to_dict(
        orient="records"
    )


# ============================================================
# ANOMALIES
# ============================================================

@app.get("/anomalies")
def anomalies():

    telemetry = generate_telemetry()

    telemetry, threshold = calculate_anomaly_score(
        telemetry
    )

    anomalies = telemetry[
        telemetry["anomaly"] == 1
    ]

    return {

        "threshold":
        round(float(threshold), 3),

        "total_anomalies":
        len(anomalies),

        "data":
        anomalies.to_dict(
            orient="records"
        ),

    }


# ============================================================
# INVENTORY
# ============================================================

@app.get("/inventory")
def inventory():

    return get_inventory().to_dict(
        orient="records"
    )


# ============================================================
# LOW STOCK
# ============================================================

@app.get("/inventory/low-stock")
def low_stock():

    return get_low_stock_items().to_dict(
        orient="records"
    )


# ============================================================
# ORDERS
# ============================================================

@app.get("/orders")
def orders():

    return get_orders().to_dict(
        orient="records"
    )


# ============================================================
# WMS SUMMARY
# ============================================================

@app.get("/wms-summary")
def wms_summary():

    return inventory_summary()

# ============================================================
# MQTT ROBOT AI ANALYSIS
# ============================================================

@app.get("/mqtt/robots")
def mqtt_robot_analysis():

    import pandas as pd

    with mqtt_lock:
        data = list(mqtt_telemetry)

    if not data:
        return {
            "source": "MQTT",
            "status": "No telemetry received",
            "robots": []
        }

    telemetry = pd.DataFrame(data)

    telemetry["maintenance_risk"] = (
        (telemetry["battery"] < 25) |
        (telemetry["motor_current"] > 12) |
        (telemetry["temperature"] > 55) |
        (telemetry["vibration"] > 4) |
        (telemetry["navigation_errors"] > 4)
    ).astype(int)

    # --------------------------------------------------------
    # Anomaly Detection
    # --------------------------------------------------------

    telemetry, threshold = calculate_anomaly_score(
        telemetry
    )

    # --------------------------------------------------------
    # Maintenance Risk Model
    # --------------------------------------------------------

    model, accuracy, features = train_maintenance_model(
        telemetry
    )

    # --------------------------------------------------------
    # Robot Health
    # --------------------------------------------------------

    telemetry["health_score"] = telemetry.apply(
        calculate_health_score,
        axis=1
    )

    telemetry["status"] = telemetry[
        "health_score"
    ].apply(
        get_robot_status
    )

    # --------------------------------------------------------
    # Latest telemetry per robot
    # --------------------------------------------------------

    latest = (
        telemetry
        .sort_values("timestamp")
        .groupby("robot_id")
        .tail(1)
    )

    return {
        "source": "MQTT",
        "robots": latest.to_dict(
            orient="records"
        )
    }

# ============================================================
# MQTT SHAP EXPLAINABILITY
# ============================================================

@app.get("/mqtt/explain")
def mqtt_explain():

    import pandas as pd
    import shap

    with mqtt_lock:
        data = list(mqtt_telemetry)

    if not data:
        return {
            "source": "MQTT",
            "status": "No telemetry received",
            "explanation": []
        }

    telemetry = pd.DataFrame(data)

    telemetry["maintenance_risk"] = (
        (telemetry["battery"] < 25) |
        (telemetry["motor_current"] > 12) |
        (telemetry["temperature"] > 55) |
        (telemetry["vibration"] > 4) |
        (telemetry["navigation_errors"] > 4)
    ).astype(int)

    model, accuracy, features = train_maintenance_model(
        telemetry
    )

    X = telemetry[features]

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        values = shap_values[1]
    else:
        values = shap_values

    importance = abs(values).mean(axis=0)

    explanation = []

    for feature, value in zip(
        features,
        importance
    ):
        explanation.append({
            "feature": feature,
            "importance": round(
                float(value),
                4
            )
        })

    explanation.sort(
        key=lambda x: x["importance"],
        reverse=True
    )

    return {
        "source": "MQTT",
        "model": "XGBoost",
        "model_accuracy": round(
            float(accuracy),
            4
        ),
        "explanation": explanation
    }

# ============================================================
# RL TASK ALLOCATION + DIJKSTRA ROUTE OPTIMIZATION
# ============================================================

@app.get("/mqtt/allocate-task")
def mqtt_allocate_task():

    import pandas as pd

    with mqtt_lock:
        data = list(mqtt_telemetry)

    if not data:
        return {
            "source": "MQTT",
            "status": "No telemetry received",
            "allocation": None,
            "route": None
        }

    telemetry = pd.DataFrame(data)

    # --------------------------------------------------------
    # Maintenance Risk
    # --------------------------------------------------------

    telemetry["maintenance_risk"] = (
        (telemetry["battery"] < 25) |
        (telemetry["motor_current"] > 12) |
        (telemetry["temperature"] > 55) |
        (telemetry["vibration"] > 4) |
        (telemetry["navigation_errors"] > 4)
    ).astype(int)

    # --------------------------------------------------------
    # Anomaly Detection
    # --------------------------------------------------------

    telemetry, threshold = calculate_anomaly_score(
        telemetry
    )

    # --------------------------------------------------------
    # Robot Health
    # --------------------------------------------------------

    telemetry["health_score"] = telemetry.apply(
        calculate_health_score,
        axis=1
    )

    telemetry["status"] = telemetry[
        "health_score"
    ].apply(
        get_robot_status
    )

    # --------------------------------------------------------
    # Latest Robot State
    # --------------------------------------------------------

    latest = (
        telemetry
        .sort_values("timestamp")
        .groupby("robot_id")
        .tail(1)
        .copy()
    )

    # --------------------------------------------------------
    # Warehouse Stations
    # --------------------------------------------------------

    stations = [
        "P1",
        "P2",
        "P3"
    ]

    # --------------------------------------------------------
    # Robot Position Simulation
    # --------------------------------------------------------

    positions = {
        "R01": (0, 0),
        "R02": (2, 2),
        "R03": (4, 2),
        "R04": (6, 0),
        "R05": (8, 1),
        "R06": (1, 4),
        "R07": (3, 5),
        "R08": (5, 4),
        "R09": (7, 3),
        "R10": (9, 0),
        "R11": (0, 6),
        "R12": (2, 7),
        "R13": (4, 6),
        "R14": (6, 5),
        "R15": (8, 4),
        "R16": (1, 8),
        "R17": (3, 8),
        "R18": (5, 8),
        "R19": (7, 8),
        "R20": (9, 5)
    }

    latest["position"] = latest[
        "robot_id"
    ].map(positions)

    # Remove robots without a warehouse position
    latest = latest[
        latest["position"].notna()
    ]

    robots = latest.to_dict(
        orient="records"
    )

    if not robots:
        return {
            "source": "MQTT",
            "status": "No valid robot positions",
            "allocation": None,
            "route": None
        }

    # --------------------------------------------------------
    # RL Task Allocator
    # --------------------------------------------------------

    allocator = WarehouseTaskAllocator(
        robots,
        stations
    )

    allocator.train(
        episodes=500
    )

    allocation = allocator.allocate_task(
        robots
    )

    if not allocation:
        return {
            "source": "MQTT",
            "status": "No robot could be allocated",
            "allocation": None,
            "route": None
        }

    # --------------------------------------------------------
    # Select Target Station
    # --------------------------------------------------------

    target_station = stations[
        len(allocation["robot_id"]) % len(stations)
    ]

    selected_robot = next(
        (
            robot
            for robot in robots
            if robot["robot_id"] ==
            allocation["robot_id"]
        ),
        None
    )

    # --------------------------------------------------------
    # Dijkstra Route Optimization
    # --------------------------------------------------------

    route = optimize_rl_robot(
        selected_robot,
        target_station
    )

    # Use the actual Dijkstra route distance
    if route.get("status") == "Route optimized":
        allocation["distance"] = route["distance"]

    # --------------------------------------------------------
    # Final Response
    # --------------------------------------------------------

    return {
        "source": "MQTT",
        "algorithm": "Q-Learning + Dijkstra",
        "training_episodes": 500,
        "stations": stations,
        "allocation": allocation,
        "route_optimization": route
    }