# api.py
# WareBot AI - FastAPI Backend

from fastapi import FastAPI

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
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "project": "WareBot AI",
        "status": "Online",
        "message": "Warehouse Robotics AI Backend is running."
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

    telemetry["status"] = telemetry["health_score"].apply(
        get_robot_status
    )

    latest = (
        telemetry
        .sort_values("timestamp")
        .groupby("robot_id")
        .tail(1)
    )

    return latest.to_dict(orient="records")


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
        "threshold": round(float(threshold), 3),
        "total_anomalies": len(anomalies),
        "data": anomalies.to_dict(
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