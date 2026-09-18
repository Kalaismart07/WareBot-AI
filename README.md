# 🤖 WareBot AI

## Real-Time Warehouse Robotics Anomaly Detection & Inventory Flow Optimization

WareBot AI is an intelligent warehouse operations platform designed to monitor autonomous warehouse robots, detect abnormal behavior, predict maintenance requirements, optimize robot task allocation and routes, and monitor warehouse inventory and orders.

---

## 🎯 Problem Statement

Modern warehouses depend on autonomous mobile robots to move inventory and fulfill orders. Robot failures, abnormal telemetry, inefficient task allocation, and warehouse congestion can cause delays and reduce operational efficiency.

WareBot AI provides a software-based warehouse intelligence system that uses robot telemetry, machine learning, route optimization, and warehouse management data to support real-time fleet operations.

---

## 🚀 Key Features

### 🛰️ Fleet Monitoring
- Real-time robot telemetry monitoring
- Battery monitoring
- Motor current monitoring
- Temperature and vibration tracking
- Navigation error monitoring
- Robot health status

### 🛠️ Predictive Maintenance
- Robot anomaly detection
- Maintenance risk identification
- Robot health scoring
- Critical robot identification
- Telemetry-based maintenance insights

### 🗺️ Route Optimization
- Warehouse grid simulation
- Obstacle-aware path planning
- Dynamic robot task allocation
- Target station selection
- Route distance calculation
- Optimized path generation

### 📦 Warehouse Management
- Inventory monitoring
- Product stock levels
- Reorder-level tracking
- Active order monitoring
- Order priority and status

### 🤖 AI Operations Assistant
- Robot health queries
- Maintenance status queries
- Anomaly information
- Fleet status assistance

---

## 🧠 Machine Learning

The project demonstrates multiple AI/ML components:

- **Autoencoder** — Robot telemetry anomaly detection
- **XGBoost** — Predictive maintenance risk scoring
- **SHAP** — Explainable maintenance insights
- **LSTM** — Battery degradation forecasting
- **Classical optimization** — Robot task and route optimization

> The telemetry dataset used in this prototype is simulated for demonstrating the complete warehouse intelligence workflow.

---

## 🏗️ System Architecture

```text
Robot Telemetry
      │
      ▼
Data Processing
      │
      ├──────────────► Anomaly Detection
      │
      ├──────────────► Predictive Maintenance
      │
      └──────────────► Battery Forecasting
                              │
                              ▼
                     Warehouse Intelligence
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       Route Optimization   WMS Data      AI Assistant
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                    Streamlit Dashboard