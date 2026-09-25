# ============================================================
# WAREBOT AI - AERO-CROP STYLE COMMAND CENTER
# ============================================================
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

from ml_models import (
    generate_telemetry,
    calculate_anomaly_score,
    train_maintenance_model,
    calculate_health_score,
    get_robot_status,
)
from wms import get_inventory, get_orders, get_low_stock_items
from optimization import WAREHOUSE_GRID, STATIONS

st.set_page_config(
    page_title="WareBot AI | Command Center",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Data
# -----------------------------
@st.cache_data
def load_data():
    telemetry = generate_telemetry(num_robots=20, records_per_robot=100)
    telemetry, threshold = calculate_anomaly_score(telemetry)
    model, accuracy, features = train_maintenance_model(telemetry)
    telemetry["health_score"] = telemetry.apply(calculate_health_score, axis=1)
    telemetry["status"] = telemetry["health_score"].apply(get_robot_status)
    return telemetry, threshold, accuracy

telemetry, anomaly_threshold, model_accuracy = load_data()
latest = telemetry.sort_values("timestamp").groupby("robot_id").tail(1).copy()
latest = latest.sort_values("robot_id")

TOTAL = len(latest)
HEALTHY = int((latest["status"] == "Healthy").sum())
WARNING = int((latest["status"] == "Warning").sum())
CRITICAL = int((latest["status"] == "Critical").sum())
ATTENTION = WARNING + CRITICAL
ANOMALIES = int(telemetry["anomaly"].sum())
LOW_STOCK_DF = get_low_stock_items()
LOW_STOCK = len(LOW_STOCK_DF)
AVG_BATTERY = float(latest["battery"].mean())
AVG_HEALTH = float(latest["health_score"].mean())

# -----------------------------
# Session navigation
# -----------------------------
if "active_page" not in st.session_state:
    st.session_state.active_page = "Dashboard"

pages = {
    "Dashboard": "Dashboard",
    "Fleet Monitoring": "Fleet Monitoring",
    "Predictive Maintenance": "Predictive Maintenance",
    "Route Optimization": "Route Optimization",
    "Inventory & Orders": "Inventory & Orders",
    "Analytics": "Analytics",
    "AI Assistant": "AI Assistant",
}

# -----------------------------
# Bright premium CSS
# -----------------------------
CSS = r"""
<style>
:root {
    --navy:#10244a;
    --blue:#1687f8;
    --blue2:#0b74e5;
    --cyan:#16b7d7;
    --green:#19c77a;
    --orange:#f7a51b;
    --red:#ef4b58;
    --purple:#8b5cf6;
    --ink:#14213d;
    --muted:#58708f;
    --line:#dce8f5;
    --bg:#f4f9ff;
    --card:#ffffff;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--ink)!important;}
[data-testid="stAppViewContainer"] .main{padding-top:0!important;}
/* Hide Streamlit's own top toolbar/Deploy strip so it never overlaps the WareBot header */
header[data-testid="stHeader"]{display:none!important;}
[data-testid="stToolbar"]{display:none!important;}
[data-testid="stSidebar"]{display:none!important;}
.block-container{max-width:1660px!important;padding:8px 28px 44px!important;margin-top:0!important;}
#MainMenu,footer{visibility:hidden!important;}
*{box-sizing:border-box;}

/* top navigation */
.wb-nav-shell{height:74px;background:rgba(255,255,255,.98);border:1px solid #d9e7f4;border-radius:22px;box-shadow:0 10px 30px rgba(37,91,142,.10);display:flex;align-items:center;padding:8px 12px;margin-bottom:12px;overflow:hidden;}
.wb-brand{display:flex;align-items:center;gap:10px;min-width:300px;padding-left:2px;}
.wb-logo{width:48px;height:48px;border-radius:15px;background:linear-gradient(145deg,#dcefff,#eef8ff);display:flex;align-items:center;justify-content:center;font-size:25px;box-shadow:inset 0 1px 0 #fff;}
.wb-brand-title{font-size:24px;font-weight:900;line-height:1;color:#10244a;letter-spacing:-.5px;}
.wb-brand-title span{color:#1687f8;}
.wb-brand-sub{font-size:10px;color:#5d7898;margin-top:5px;font-weight:650;white-space:nowrap;}
.wb-live{display:flex;align-items:center;justify-content:center;gap:7px;padding:10px 13px;border:1px solid #bfeedd;border-radius:999px;background:#f1fff9;color:#087c50;font-size:11px;font-weight:850;white-space:nowrap;min-width:112px;}
.wb-live-dot{width:9px;height:9px;border-radius:50%;background:#15c47a;box-shadow:0 0 0 5px rgba(21,196,122,.12);animation:pulse 1.8s infinite;}
.wb-date{padding:0 10px;border-left:1px solid #dce8f5;font-size:10px;color:#4d6685;line-height:1.35;text-align:center;min-width:98px;white-space:nowrap;}
@keyframes pulse{50%{box-shadow:0 0 0 9px rgba(21,196,122,0);}}

/* Streamlit nav buttons */
.wb-nav-buttons{margin:0 3px;}
.wb-nav-buttons .stButton>button{border:1px solid transparent!important;background:transparent!important;color:#1e3b63!important;border-radius:14px!important;font-size:10px!important;font-weight:780!important;min-height:48px!important;padding:0 6px!important;box-shadow:none!important;transition:.2s!important;white-space:nowrap!important;}
.wb-nav-buttons .stButton>button:hover{background:#edf6ff!important;color:#0876e7!important;border-color:#d5e8f8!important;transform:translateY(-1px)!important;}
.wb-nav-buttons.active .stButton>button{background:linear-gradient(135deg,#1687f8,#0876e7)!important;color:white!important;border-color:#1687f8!important;box-shadow:0 8px 18px rgba(22,135,248,.22)!important;}
.wb-nav-buttons.active .stButton>button:hover{color:white!important;}

/* hide refresh row spacing */
.refresh-row{display:none;}

/* hero */
.wb-hero{
    height:220px;
    border-radius:24px;
    overflow:hidden;
    position:relative;
    background:
        url('https://www.atpress.ne.jp/releases/219292/LL_img_219292_2.png')
        center/cover no-repeat;
    box-shadow:0 14px 38px rgba(41,100,154,.14);
    border:1px solid #d6e8f8;
    margin-bottom:14px;
}
.wb-hero:before{
    content:"";
    position:absolute;
    inset:0;
    background:linear-gradient(
        90deg,
        rgba(255,255,255,.90) 0%,
        rgba(255,255,255,.68) 32%,
        rgba(255,255,255,.30) 58%,
        rgba(255,255,255,.08) 100%
    );
}
.wb-hero-copy{position:absolute;left:34px;top:28px;max-width:680px;z-index:2;}
.wb-eyebrow{color:#0985e9;font-size:12px;letter-spacing:1.6px;font-weight:900;margin-bottom:7px;}
.wb-hero-title{font-size:48px;line-height:.98;font-weight:950;letter-spacing:-2.1px;color:#10264b;margin-bottom:10px;}
.wb-hero-title span{color:#147ff0;}
.wb-hero-sub{font-size:14px;line-height:1.55;color:#416384;font-weight:560;max-width:720px;}
.wb-fleet-float{position:absolute;right:20px;bottom:22px;z-index:3;background:rgba(255,255,255,.94);border:1px solid rgba(214,229,243,.95);border-radius:17px;padding:13px 18px;display:flex;align-items:center;gap:22px;box-shadow:0 12px 30px rgba(29,80,125,.16);backdrop-filter:blur(14px);}
.wb-fleet-online{font-weight:900;color:#1b2e4d;font-size:13px;}.wb-fleet-online small{display:block;color:#6d8098;font-weight:600;font-size:10px;margin-top:2px;}.wb-fleet-dot{display:inline-block;width:10px;height:10px;background:#18c77b;border-radius:50%;margin-right:7px;box-shadow:0 0 0 5px rgba(24,199,123,.11);}
.wb-float-stat{border-left:1px solid #dce7f1;padding-left:18px;min-width:82px;text-align:center;}.wb-float-stat b{font-size:18px;color:#152a4c;display:block;}.wb-float-stat span{font-size:9px;color:#71849a;font-weight:700;}

/* KPI cards */
.kpi-card{min-height:126px;border-radius:18px;border:1px solid #dbe8f4;background:white;box-shadow:0 10px 28px rgba(33,86,133,.08);padding:16px;position:relative;overflow:hidden;transition:.22s;}
.kpi-card:hover{transform:translateY(-3px);box-shadow:0 16px 32px rgba(33,86,133,.13);}
.kpi-card:before{content:"";position:absolute;left:0;right:0;top:0;height:3px;background:var(--accent);}
.kpi-icon{width:46px;height:46px;border-radius:14px;background:var(--soft);display:flex;align-items:center;justify-content:center;font-size:23px;float:left;margin-right:11px;}
.kpi-label{font-size:11px;color:#526b89;font-weight:850;letter-spacing:.35px;padding-top:2px;}.kpi-value{font-size:29px;color:#112747;font-weight:950;line-height:1.05;margin-top:5px;}.kpi-note{font-size:10px;color:#647b98;margin-top:7px;font-weight:650;}.kpi-note b{color:var(--accent);}.kpi-ring{position:absolute;right:16px;top:20px;width:52px;height:52px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:900;color:#153052;background:conic-gradient(var(--accent) var(--pct),#edf2f7 0);}.kpi-ring:after{content:"";position:absolute;width:39px;height:39px;background:#fff;border-radius:50%;}.kpi-ring span{position:relative;z-index:1;}

/* section */
.section-title{font-size:21px;font-weight:900;color:#132744;margin:12px 0 2px;letter-spacing:-.5px;}.section-sub{font-size:11px;color:#71849c;margin-bottom:10px;font-weight:600;}
.panel{background:#fff;border:1px solid #dce8f4;border-radius:18px;box-shadow:0 9px 28px rgba(32,82,128,.07);padding:16px;transition:.2s;}.panel:hover{box-shadow:0 13px 32px rgba(32,82,128,.10);}
.panel-title{font-size:15px;font-weight:900;color:#142947;}.panel-sub{font-size:10px;color:#72869f;margin-top:3px;}

/* plotly */
.js-plotly-plot{border-radius:14px!important;}
[data-testid="stPlotlyChart"]{border-radius:14px!important;overflow:hidden!important;}

/* buttons and inputs */
.stButton>button{border-radius:11px!important;border:1px solid #cfe0ef!important;background:#fff!important;color:#234263!important;font-weight:750!important;min-height:38px!important;transition:.2s!important;}
.stButton>button:hover{border-color:#79b9ef!important;background:#f3f9ff!important;color:#0d79df!important;transform:translateY(-1px)!important;}
[data-testid="stSelectbox"]>div>div{border-radius:11px!important;border:1px solid #9fc1df!important;background:#fff!important;min-height:44px!important;}
/* Selectbox readability: dark selected value, label, arrow and dropdown options */
[data-testid="stSelectbox"] label{
    color:#294b6d!important;
    font-size:13px!important;
    font-weight:800!important;
}
/* HARD FIX for Streamlit BaseWeb selectbox text visibility */
[data-testid="stSelectbox"] div[data-baseweb="select"],
[data-testid="stSelectbox"] div[data-baseweb="select"] > div{
    background:#ffffff!important;
    border-color:#a9c8e4!important;
    color:#173858!important;
    opacity:1!important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] span,
[data-testid="stSelectbox"] div[data-baseweb="select"] div,
[data-testid="stSelectbox"] div[data-baseweb="select"] p,
[data-testid="stSelectbox"] div[data-baseweb="select"] input{
    color:#173858!important;
    -webkit-text-fill-color:#173858!important;
    opacity:1!important;
    font-size:13px!important;
    font-weight:700!important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] svg{
    fill:#315675!important;
    color:#315675!important;
    opacity:1!important;
}
[data-baseweb="popover"],
[data-baseweb="menu"]{
    background:#ffffff!important;
    color:#173858!important;
}
[data-baseweb="popover"] [role="option"],
[data-baseweb="menu"] [role="option"]{
    color:#173858!important;
    -webkit-text-fill-color:#173858!important;
    background:#ffffff!important;
    font-size:13px!important;
    font-weight:700!important;
    opacity:1!important;
}
[data-baseweb="popover"] [role="option"] *,
[data-baseweb="menu"] [role="option"] *{
    color:#173858!important;
    -webkit-text-fill-color:#173858!important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="menu"] [role="option"]:hover{
    background:#edf6ff!important;
    color:#0876e7!important;
}

/* Streamlit BaseWeb selected value — force visible dark text */
div[data-testid="stSelectbox"] div[data-baseweb="select"] div[role="button"],
div[data-testid="stSelectbox"] div[data-baseweb="select"] div[role="combobox"]{
    color:#102f50!important;
    -webkit-text-fill-color:#102f50!important;
    opacity:1!important;
    font-size:14px!important;
    font-weight:800!important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] div[role="button"] > div,
div[data-testid="stSelectbox"] div[data-baseweb="select"] div[role="combobox"] > div{
    color:#102f50!important;
    -webkit-text-fill-color:#102f50!important;
    opacity:1!important;
    font-size:14px!important;
    font-weight:800!important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] input{
    color:#102f50!important;
    -webkit-text-fill-color:#102f50!important;
    opacity:1!important;
    caret-color:#102f50!important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] span{
    color:#102f50!important;
    -webkit-text-fill-color:#102f50!important;
    opacity:1!important;
    font-size:14px!important;
    font-weight:800!important;
}

/* BaseWeb's actual selected-value classes */
div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="singleValue"],
div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="ValueContainer"],
div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="placeholder"]{
    color:#102f50!important;
    -webkit-text-fill-color:#102f50!important;
    opacity:1!important;
    font-size:14px!important;
    font-weight:800!important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="singleValue"] *,
div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="ValueContainer"] *{
    color:#102f50!important;
    -webkit-text-fill-color:#102f50!important;
    opacity:1!important;
}

/* Make the whole control unambiguously light/readable */
div[data-testid="stSelectbox"] [data-baseweb="select"]{
    color-scheme:light!important;
    filter:none!important;
    opacity:1!important;
}

/* Force Streamlit/BaseWeb selected text and input text to dark */
div[data-testid="stSelectbox"] [data-baseweb="select"] input{
    color:#102f50!important;
    -webkit-text-fill-color:#102f50!important;
    caret-color:#102f50!important;
    opacity:1!important;
    font-size:14px!important;
    font-weight:800!important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] input::placeholder{
    color:#102f50!important;
    -webkit-text-fill-color:#102f50!important;
    opacity:1!important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"]{
    color:#102f50!important;
    -webkit-text-fill-color:#102f50!important;
    opacity:1!important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] [aria-selected="true"]{
    color:#102f50!important;
    -webkit-text-fill-color:#102f50!important;
    opacity:1!important;
    font-weight:800!important;
}

/* Selectbox label */
div[data-testid="stSelectbox"] label{
    color:#163758!important;
    opacity:1!important;
    font-size:13px!important;
    font-weight:750!important;
}

/* Dropdown menu items */
[role="listbox"] [role="option"]{
    color:#102f50!important;
    -webkit-text-fill-color:#102f50!important;
    opacity:1!important;
    font-size:14px!important;
    font-weight:700!important;
    background:#ffffff!important;
}
[role="listbox"] [role="option"][aria-selected="true"]{
    color:#102f50!important;
    -webkit-text-fill-color:#102f50!important;
    background:#edf6ff!important;
}

/* BaseWeb closed Select: force the actual single-value node dark.
   Streamlit may render this as generated class names, so use partial class selectors. */
div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="singleValue"],
div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="SingleValue"],
div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="ValueContainer"] > div{
    color:#102f50!important;
    -webkit-text-fill-color:#102f50!important;
    opacity:1!important;
    visibility:visible!important;
    font-size:14px!important;
    font-weight:800!important;
}

/* Do not let a disabled-looking parent fade the selected value */
div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="ValueContainer"]{
    opacity:1!important;
    color:#102f50!important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="Control"]{
    opacity:1!important;
}

/* Arrow/icon remains visible */
div[data-testid="stSelectbox"] [data-baseweb="select"] svg{
    opacity:1!important;
    color:#486581!important;
    fill:#486581!important;
}

/* Plotly axis titles/ticks: force dark readable text */
[data-testid="stPlotlyChart"] .xtitle,
[data-testid="stPlotlyChart"] .ytitle{
    fill:#315675!important;
    font-size:13px!important;
}
[data-testid="stPlotlyChart"] .xtick text,
[data-testid="stPlotlyChart"] .ytick text{
    fill:#315675!important;
    font-size:12px!important;
}

/* alerts */
.alert-card{border-radius:13px;border:1px solid #e0eaf4;background:#fff;padding:10px 12px;margin-bottom:8px;display:flex;gap:10px;align-items:flex-start;box-shadow:0 5px 16px rgba(35,84,128,.05);}.alert-icon{width:32px;height:32px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:16px;flex:none;}.alert-title{font-size:11px;font-weight:900;}.alert-msg{font-size:10px;color:#667d97;margin-top:3px;}.alert-time{font-size:9px;color:#94a6ba;margin-left:auto;white-space:nowrap;}

/* map legend */
.legend-row{display:flex;flex-wrap:wrap;gap:13px;padding:7px 3px 0;color:#5f7590;font-size:10px;font-weight:700;}.legend-dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:4px;vertical-align:-1px;}.route-line{display:inline-block;width:22px;border-top:2px dashed #58a6f5;margin-right:5px;vertical-align:middle;}

/* table */
[data-testid="stDataFrame"]{border:1px solid #dce8f4!important;border-radius:12px!important;overflow:hidden!important;}
.wb-table-wrap{border:1px solid #dce8f4;border-radius:12px;overflow:hidden;background:#fff;margin-top:10px;}
.wb-table{width:100%;border-collapse:collapse;font-size:10px;color:#29425f;}
.wb-table th{background:#f3f8fd;color:#5b718b;text-align:left;font-size:9px;letter-spacing:.25px;font-weight:850;padding:9px 7px;border-bottom:1px solid #dce8f4;}
.wb-table td{padding:9px 7px;border-bottom:1px solid #edf2f7;white-space:nowrap;}
.wb-table tr:last-child td{border-bottom:0;}
.wb-status{display:inline-flex;align-items:center;gap:4px;padding:4px 7px;border-radius:999px;font-size:9px;font-weight:850;}
.wb-status.healthy{background:#e7faf2;color:#0b9c62;}
.wb-status.warning{background:#fff4dd;color:#c47b00;}
.wb-status.critical{background:#ffebee;color:#d93545;}
.wb-bar{height:6px;width:54px;border-radius:99px;background:#e9f0f6;display:inline-block;overflow:hidden;vertical-align:middle;margin-right:4px;}
.wb-bar i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#19c77a,#4dd9a0);}
/* final operations polish */
.wb-table-wrap{box-shadow:inset 0 1px 0 rgba(255,255,255,.8),0 6px 18px rgba(34,84,126,.05);}
.wb-table tbody tr:hover{background:#f7fbff;}
.wb-map-panel{background:linear-gradient(180deg,#fbfdff 0%,#f5faff 100%);border:1px solid #d9e8f5;border-radius:14px;padding:4px;}
.wb-map-badge{display:inline-flex;align-items:center;gap:5px;background:#fff;border:1px solid #dce9f5;border-radius:10px;padding:6px 9px;color:#33506f;font-size:9px;font-weight:800;box-shadow:0 4px 12px rgba(37,86,126,.06);}
.ai-box{margin-top:12px;border:1px solid #f0dfb6;border-radius:14px;background:linear-gradient(135deg,#fffaf0,#fff6df);padding:15px;box-shadow:0 7px 18px rgba(204,143,35,.08);}
.ai-box-title{color:#db8500;font-weight:900;font-size:12px;letter-spacing:.2px;}
.ai-main{display:flex;justify-content:space-between;align-items:center;margin-top:12px;padding:10px 11px;background:#fff;border:1px solid #f3e4c1;border-radius:11px;}
.ai-main strong{font-size:19px;color:#132947;}
.ai-stat{font-size:10px;color:#73859b;font-weight:700;}
.ai-stat b{display:block;font-size:15px;color:#132947;margin-top:3px;}
.alert-card{transition:transform .18s,box-shadow .18s;}.alert-card:hover{transform:translateY(-2px);box-shadow:0 8px 20px rgba(35,84,128,.09);}

/* ===== Readability upgrade: darker text + larger type ===== */
.wb-brand-title{font-size:26px;color:#0b1f3a;}
.wb-brand-sub{font-size:11px;color:#385878;font-weight:700;}
.wb-nav-buttons .stButton>button{font-size:12px!important;color:#16385f!important;font-weight:820!important;min-height:50px!important;}
.wb-live{font-size:12px;}
.wb-date{font-size:11px;color:#294969;font-weight:700;}
.wb-eyebrow{font-size:13px;color:#0676d8;}
.wb-hero-title{font-size:50px;color:#0b2245;}
.wb-hero-sub{font-size:15px;color:#294f74;font-weight:650;}
.wb-fleet-online{font-size:14px;color:#122947;}
.wb-fleet-online small{font-size:11px;color:#4f6884;}
.wb-float-stat b{font-size:20px;}
.wb-float-stat span{font-size:10px;color:#4e6885;}
.kpi-label{font-size:12px;color:#304e70;font-weight:900;}
.kpi-value{font-size:31px;color:#0c2343;}
.kpi-note{font-size:11px;color:#405d7b;font-weight:700;}
.kpi-ring{font-size:11px;}
.section-title{font-size:24px;color:#0b2343;}
.section-sub{font-size:12px;color:#496783;font-weight:650;}
.panel-title{font-size:17px;color:#0b2545;}
.panel-sub{font-size:11px;color:#4c6985;font-weight:600;}
.stButton>button{font-size:12px!important;}
[data-testid="stSelectbox"] label,
[data-testid="stTextInput"] label{font-size:12px!important;color:#294b6d!important;font-weight:750!important;}
.wb-table{font-size:11px;color:#213f5f;}
.wb-table th{font-size:10px;color:#3e5c78;padding:10px 8px;}
.wb-table td{font-size:11px;padding:10px 8px;}
.wb-status{font-size:10px;}
.alert-title{font-size:12px;}
.alert-msg{font-size:11px;color:#405d78;}
.alert-time{font-size:10px;color:#647b94;}
.legend-row{font-size:11px;color:#405d78;}
.ai-box-title{font-size:13px;}
.ai-stat{font-size:11px;color:#49657f;}
.ai-stat b{font-size:16px;color:#0c2343;}
.plotly .xtick text,.plotly .ytick text{font-size:11px!important;}
@media(max-width:1100px){.wb-brand{min-width:220px}.wb-hero-title{font-size:39px}.wb-fleet-float{right:10px;gap:10px}.wb-nav-buttons .stButton>button{font-size:10px!important;}.wb-date{display:none;}}

/* FINAL SELECTBOX TEXT OVERRIDE — black only, no layout changes */
div[data-testid="stSelectbox"] [data-baseweb="select"] * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
    text-shadow: none !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] input {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
    caret-color: #000000 !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] svg {
    color: #4b6078 !important;
    fill: #4b6078 !important;
    -webkit-text-fill-color: initial !important;
}

/* Dropdown popup itself — option text black */
[data-baseweb="menu"] *,
[data-baseweb="popover"] *,
[role="listbox"] *,
[role="option"] {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
}
[role="option"][aria-selected="true"] {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}


/* v9: selected Streamlit BaseWeb value only — black, fully opaque */
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child {
    opacity: 1 !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child > div:first-child {
    opacity: 1 !important;
    color: #000 !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child > div:first-child > div {
    opacity: 1 !important;
    color: #000 !important;
    -webkit-text-fill-color: #000 !important;
    visibility: visible !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child > div:first-child > div > div {
    opacity: 1 !important;
    color: #000 !important;
    -webkit-text-fill-color: #000 !important;
    visibility: visible !important;
    font-weight: 800 !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child input {
    opacity: 1 !important;
    color: #000 !important;
    -webkit-text-fill-color: #000 !important;
    visibility: visible !important;
}

/* v10: final visibility override for the selected value.
   Force opacity on the whole Streamlit widget and use a zero-blur black
   text shadow so even a BaseWeb light/placeholder text color becomes visible. */
div[data-testid="stSelectbox"]{
    opacity:1 !important;
    filter:none !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"],
div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
div[data-testid="stSelectbox"] [data-baseweb="select"] > div > div,
div[data-testid="stSelectbox"] [data-baseweb="select"] > div > div > div,
div[data-testid="stSelectbox"] [data-baseweb="select"] > div > div > div > div{
    opacity:1 !important;
    visibility:visible !important;
    color:#000000 !important;
    -webkit-text-fill-color:#000000 !important;
    text-shadow:0 0 0 #000000 !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="singleValue"],
div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="placeholder"],
div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="ValueContainer"] *{
    opacity:1 !important;
    visibility:visible !important;
    color:#000000 !important;
    -webkit-text-fill-color:#000000 !important;
    text-shadow:0 0 0 #000000 !important;
}


/* FINAL SELECTBOX FIX
   Keep the native Streamlit selectbox exactly as-is.
   Only force the selected value to render as solid black. */
div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="singleValue"] {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
    visibility: visible !important;
    font-weight: 800 !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="SingleValue"] {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
    visibility: visible !important;
    font-weight: 800 !important;
}

/* BaseWeb value container */
div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="ValueContainer"] {
    color: #000000 !important;
    opacity: 1 !important;
}

/* Text nodes inside the value container */
div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="ValueContainer"] span,
div[data-testid="stSelectbox"] [data-baseweb="select"] [class*="ValueContainer"] div {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* Closed selectbox fallback: target the first value area only */
div[data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child > div:first-child {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child > div:first-child > div {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
    visibility: visible !important;
    font-weight: 800 !important;
}


/* v12 - final selectbox readability fix */
div[data-testid="stSelectbox"] [data-baseweb="select"] {
    background: #ffffff !important;
    color: #000000 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #000000 !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    color: #000000 !important;
    opacity: 1 !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"] {
    color: #000000 !important;
    opacity: 1 !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"] > div {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
    visibility: visible !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] input,
div[data-testid="stSelectbox"] [data-baseweb="select"] [aria-autocomplete] {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
    visibility: visible !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] svg {
    opacity: 1 !important;
    color: #526b86 !important;
    fill: #526b86 !important;
}

/* Strong fallback: every text-bearing descendant in the closed control */
div[data-testid="stSelectbox"] [data-baseweb="select"] span,
div[data-testid="stSelectbox"] [data-baseweb="select"] p {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
    visibility: visible !important;
}


/* FINAL - STREAMLIT BASEWEB SELECTBOX VALUE VISIBILITY */
div[data-testid="stSelectbox"] [data-baseweb="select"] {
    color: #000000 !important;
    opacity: 1 !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
    visibility: visible !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] [aria-selected="true"] {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] input {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] svg {
    color: #243b5a !important;
    fill: #243b5a !important;
    opacity: 1 !important;
}


/* ============================================================
   WAREBOT BLUE PREMIUM HERO
   Original blue UI preserved — background image only
   ============================================================ */
.wb-nav-shell{height:82px!important;border-radius:24px!important;background:rgba(255,255,255,.96)!important;border:1px solid #d9e7f4!important;box-shadow:0 14px 40px rgba(34,91,142,.10)!important;padding:9px 16px!important;}
.wb-logo{width:52px!important;height:52px!important;border-radius:16px!important;background:linear-gradient(145deg,#dcefff,#eef8ff)!important;box-shadow:0 8px 20px rgba(22,135,248,.12)!important;}
.wb-brand-title{font-size:25px!important;color:#10244a!important;}
.wb-brand-title span{color:#1687f8!important;}
.wb-brand-sub{color:#5d7898!important;font-size:10px!important;}
.wb-live{background:#f1fff9!important;border-color:#bfeedd!important;color:#087c50!important;min-width:145px!important;padding:11px 16px!important;}
.wb-date{border-left-color:#dce8f5!important;color:#4d6685!important;}
.wb-nav-buttons .stButton>button{color:#1e3b63!important;border-radius:999px!important;font-size:10px!important;font-weight:800!important;}
.wb-nav-buttons .stButton>button:hover{background:#edf6ff!important;color:#0876e7!important;border-color:#d5e8f8!important;}
.wb-nav-buttons.active .stButton>button{background:linear-gradient(135deg,#1687f8,#0876e7)!important;border-color:#1687f8!important;color:white!important;box-shadow:0 8px 20px rgba(22,135,248,.22)!important;}
.wb-hero{height:250px!important;border-radius:26px!important;margin-bottom:8px!important;border:1px solid #d6e8f8!important;background:url('https://www.atpress.ne.jp/releases/219292/LL_img_219292_2.png') center/cover no-repeat!important;box-shadow:0 18px 45px rgba(37,91,142,.13)!important;}
.wb-hero:before{background:linear-gradient(90deg,rgba(244,249,255,.97) 0%,rgba(244,249,255,.90) 27%,rgba(244,249,255,.62) 48%,rgba(244,249,255,.20) 72%,rgba(244,249,255,.04) 100%)!important;}
.wb-hero-copy{left:42px!important;top:38px!important;max-width:730px!important;}
.wb-eyebrow{color:#0876d8!important;letter-spacing:2px!important;font-size:11px!important;}
.wb-hero-title{font-size:52px!important;letter-spacing:-2.5px!important;color:#10244a!important;}
.wb-hero-title span{color:#1687f8!important;}
.wb-hero-sub{color:#416384!important;font-size:14px!important;max-width:700px!important;}
.wb-fleet-float{right:24px!important;bottom:22px!important;border-radius:20px!important;border-color:rgba(214,229,243,.95)!important;box-shadow:0 15px 35px rgba(39,91,142,.15)!important;}
.wb-fleet-dot{background:#19c77a!important;}
.kpi-card{border-radius:20px!important;border-color:#d9e8f4!important;box-shadow:0 12px 32px rgba(38,91,142,.075)!important;}
.kpi-card:hover{box-shadow:0 18px 38px rgba(38,91,142,.13)!important;}
.section-title{color:#132744!important;font-size:22px!important;}
.section-sub{color:#71849c!important;}
.panel{border-radius:20px!important;border-color:#d9e8f4!important;box-shadow:0 11px 30px rgba(38,91,142,.065)!important;}
.panel-title{color:#142947!important;}
[data-testid="stSelectbox"]>div>div{border-radius:13px!important;border-color:#b9d1e7!important;}



/* FINAL RADIO TEXT VISIBILITY FIX */
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] label *,
div[data-testid="stRadio"] [role="radio"] + div,
div[data-testid="stRadio"] [role="radio"] ~ div,
div[data-testid="stRadio"] p,
div[data-testid="stRadio"] span {
    color:#173858 !important;
    -webkit-text-fill-color:#173858 !important;
    opacity:1 !important;
    visibility:visible !important;
    text-shadow:none !important;
    font-weight:750 !important;
}

div[data-testid="stRadio"] label > div:last-child,
div[data-testid="stRadio"] label > div:last-child * {
    color:#173858 !important;
    -webkit-text-fill-color:#173858 !important;
    opacity:1 !important;
    visibility:visible !important;
}

div[data-testid="stRadio"] [role="radio"] {
    opacity:1 !important;
}

div[data-testid="stRadio"] [role="radio"][aria-checked="true"] {
    opacity:1 !important;
}


/* PREMIUM AI RECOMMENDATION CARD */
.ai-box{
    background:linear-gradient(145deg,#fffdf7 0%,#fff7df 100%) !important;
    border:1px solid #f3d58a !important;
    border-radius:20px !important;
    padding:18px !important;
    box-shadow:0 10px 28px rgba(214,164,55,.12) !important;
}

.ai-box-title{
    color:#c47a00 !important;
    font-weight:900 !important;
    font-size:16px !important;
    margin-bottom:14px !important;
}

.ai-main{
    background:#ffffff !important;
    border:1px solid #f0dfb4 !important;
    border-radius:14px !important;
    padding:15px 14px !important;
}

.ai-stat{
    color:#58708d !important;
    font-size:11px !important;
    font-weight:800 !important;
    letter-spacing:.2px !important;
}

.ai-main .ai-stat{
    width:48% !important;
}

.ai-main strong{
    display:block !important;
    margin-top:5px !important;
    font-size:20px !important;
    line-height:1.1 !important;
    color:#132947 !important;
    font-weight:900 !important;
}


/* ROUTE OPTIMIZATION METRIC VISIBILITY */
[data-testid="stMetric"] {
    background:#ffffff !important;
    border:1px solid #d6e8f8 !important;
    border-radius:16px !important;
    opacity:1 !important;
}

[data-testid="stMetricLabel"] {
    color:#58708d !important;
    -webkit-text-fill-color:#58708d !important;
    opacity:1 !important;
    font-weight:800 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] *,
[data-testid="stMetricValue"] div {
    color:#132947 !important;
    -webkit-text-fill-color:#132947 !important;
    opacity:1 !important;
    visibility:visible !important;
    font-weight:900 !important;
}


/* OPTIMIZED PATH CODE BOX ALIGNMENT */
[data-testid="stCodeBlock"] {
    margin-top:0 !important;
}

[data-testid="stCodeBlock"] pre {
    margin:0 !important;
    padding:12px 16px !important;
    line-height:1.35 !important;
    font-size:14px !important;
    box-sizing:border-box !important;
}

[data-testid="stCodeBlock"] code {
    padding:0 !important;
    margin:0 !important;
}


/* AI ASSISTANT CHAT UI */
div[data-testid="stChatInput"] {
    background:#ffffff !important;
    border:1px solid #cfe2f5 !important;
    border-radius:14px !important;
    box-shadow:0 4px 14px rgba(31,105,170,.08) !important;
}

div[data-testid="stChatInput"] textarea {
    background:#ffffff !important;
    color:#173858 !important;
    -webkit-text-fill-color:#173858 !important;
    caret-color:#1687f8 !important;
    font-size:15px !important;
    font-weight:600 !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color:#71859b !important;
    -webkit-text-fill-color:#71859b !important;
    opacity:1 !important;
}

div[data-testid="stChatInput"] button {
    background:#1687f8 !important;
    color:#ffffff !important;
    border-radius:10px !important;
}

div[data-testid="stChatMessage"] {
    background:#ffffff !important;
    border:1px solid #d7e8f8 !important;
    border-radius:14px !important;
    color:#173858 !important;
}

div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] span,
div[data-testid="stChatMessage"] div {
    color:#173858 !important;
    -webkit-text-fill-color:#173858 !important;
    opacity:1 !important;
}


/* FINAL AI CHAT WHITE UI */
div[data-testid="stChatInput"],
div[data-testid="stChatInput"] > div,
div[data-testid="stChatInput"] form {
    background:#ffffff !important;
    border:1px solid #cfe2f5 !important;
    border-radius:14px !important;
    box-shadow:0 3px 12px rgba(31,105,170,.06) !important;
}

div[data-testid="stChatInput"] textarea {
    background:#ffffff !important;
    color:#173858 !important;
    -webkit-text-fill-color:#173858 !important;
    caret-color:#1687f8 !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color:#71859b !important;
    -webkit-text-fill-color:#71859b !important;
    opacity:1 !important;
}

div[data-testid="stChatMessage"] {
    background:#ffffff !important;
    border:1px solid #d6e8f8 !important;
    border-radius:14px !important;
    box-shadow:0 3px 12px rgba(31,105,170,.05) !important;
}

div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] span {
    color:#173858 !important;
    -webkit-text-fill-color:#173858 !important;
}


/* =========================================================
   FINAL PREMIUM HEADER NAVIGATION
   ========================================================= */

.wb-nav-buttons {
    position:relative !important;
    margin:0 4px 24px !important;
    overflow:visible !important;
}

.wb-nav-buttons .stButton {
    width:100% !important;
}

.wb-nav-buttons .stButton > button {
    min-height:48px !important;
    height:48px !important;
    padding:0 14px !important;
    border:1px solid #d7e6f4 !important;
    border-radius:999px !important;
    background:rgba(255,255,255,.96) !important;
    color:#173858 !important;
    -webkit-text-fill-color:#173858 !important;
    font-size:12px !important;
    font-weight:800 !important;
    box-shadow:0 5px 15px rgba(39,91,142,.07) !important;
    transition:all .22s ease !important;
}

.wb-nav-buttons .stButton > button:hover {
    background:#edf7ff !important;
    color:#0876e7 !important;
    -webkit-text-fill-color:#0876e7 !important;
    border-color:#b9dafa !important;
    transform:translateY(-2px) !important;
    box-shadow:0 8px 20px rgba(22,135,248,.14) !important;
}

/* SELECTED PAGE */
.wb-nav-buttons.active .stButton > button {
    background:linear-gradient(135deg,#1687f8,#0876e7) !important;
    color:#ffffff !important;
    -webkit-text-fill-color:#ffffff !important;
    border-color:#1687f8 !important;
    box-shadow:
        0 10px 24px rgba(22,135,248,.28),
        0 0 0 3px rgba(22,135,248,.08) !important;
    transform:translateY(-1px) !important;
}

.wb-nav-buttons.active .stButton > button:hover {
    background:linear-gradient(135deg,#1687f8,#0876e7) !important;
    color:#ffffff !important;
    -webkit-text-fill-color:#ffffff !important;
}

/* CURRENT VIEW BADGE */
.wb-nav-buttons.active::after {
    content:"CURRENT VIEW";
    position:absolute;
    left:50%;
    top:54px;
    transform:translateX(-50%);
    background:#eaf6ff;
    color:#1687f8;
    border:1px solid #b9dcfa;
    border-radius:999px;
    padding:4px 11px;
    font-size:9px;
    font-weight:900;
    letter-spacing:.5px;
    white-space:nowrap;
    box-shadow:0 5px 14px rgba(22,135,248,.12);
    z-index:20;
}

/* SMALL BLUE POINTER */
.wb-nav-buttons.active::before {
    content:"";
    position:absolute;
    left:50%;
    top:48px;
    transform:translateX(-50%);
    width:0;
    height:0;
    border-left:6px solid transparent;
    border-right:6px solid transparent;
    border-bottom:6px solid #eaf6ff;
    z-index:21;
}

/* BRAND */
.wb-brand-title {
    font-size:25px !important;
    font-weight:950 !important;
    letter-spacing:-.7px !important;
}

.wb-logo {
    width:52px !important;
    height:52px !important;
    border-radius:16px !important;
    box-shadow:0 8px 22px rgba(22,135,248,.12) !important;
}

/* LIVE STATUS */
.wb-live {
    min-width:145px !important;
    padding:11px 16px !important;
    box-shadow:0 7px 18px rgba(25,199,122,.08) !important;
}


/* FINAL WHITE FLEET TABLE */
.wb-fleet-table-wrap {
    background:#ffffff !important;
    border:1px solid #d7e6f4 !important;
    border-radius:12px !important;
    overflow:hidden !important;
    box-shadow:0 5px 18px rgba(31,105,170,.06) !important;
}

.wb-fleet-table {
    width:100% !important;
    border-collapse:collapse !important;
    background:#ffffff !important;
    color:#173858 !important;
    font-size:13px !important;
}

.wb-fleet-table th {
    background:#f7fbff !important;
    color:#526b84 !important;
    font-weight:800 !important;
    text-align:left !important;
    padding:11px 10px !important;
    border-bottom:1px solid #dce9f5 !important;
}

.wb-fleet-table td {
    background:#ffffff !important;
    color:#173858 !important;
    padding:10px !important;
    border-bottom:1px solid #e8f0f7 !important;
}

.wb-fleet-table tbody tr:hover td {
    background:#f7fbff !important;
}

.wb-fleet-table td.healthy {
    color:#159b68 !important;
    font-weight:700 !important;
}

.wb-fleet-table td.critical {
    color:#ef4d59 !important;
    font-weight:800 !important;
}



/* FINAL WHITE DATA TABLES */
[data-testid="stDataFrame"] {
    background:#ffffff !important;
    border:1px solid #d7e6f4 !important;
    border-radius:12px !important;
    overflow:hidden !important;
}

[data-testid="stDataFrame"] [role="grid"] {
    background:#ffffff !important;
    color:#173858 !important;
}

[data-testid="stDataFrame"] [role="columnheader"] {
    background:#f7fbff !important;
    color:#526b84 !important;
    border-color:#dce9f5 !important;
}

[data-testid="stDataFrame"] [role="gridcell"] {
    background:#ffffff !important;
    color:#173858 !important;
    border-color:#e5eef6 !important;
}

[data-testid="stDataFrame"] canvas {
    background:#ffffff !important;
}


/* REAL WHITE HTML DATA TABLES */
.wb-white-table-box{
    width:100%;
    max-height:500px;
    overflow:auto;
    background:#ffffff !important;
    border:1px solid #d7e6f4;
    border-radius:12px;
    box-shadow:0 5px 18px rgba(31,105,170,.06);
}

.wb-white-html-table{
    width:100%;
    border-collapse:collapse;
    background:#ffffff !important;
    color:#173858 !important;
    font-size:13px;
}

.wb-white-html-table thead th{
    position:sticky;
    top:0;
    z-index:2;
    background:#f5faff !important;
    color:#526b84 !important;
    font-weight:800;
    text-align:left;
    padding:11px 10px;
    border-bottom:1px solid #dce9f5;
    white-space:nowrap;
}

.wb-white-html-table tbody td{
    background:#ffffff !important;
    color:#173858 !important;
    padding:10px;
    border-bottom:1px solid #e7eef5;
    white-space:nowrap;
}

.wb-white-html-table tbody tr:hover td{
    background:#f7fbff !important;
}

</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------
def nav_button(label, page, active=False):
    cls = "active" if active else ""
    st.markdown(f'<div class="wb-nav-buttons {cls}">', unsafe_allow_html=True)
    if st.button(label, key=f"nav_{page}", width="stretch"):
        st.session_state.active_page = page
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def metric_card(icon, label, value, note, accent, soft, pct=None):
    ring = ""
    if pct is not None:
        ring = f'<div class="kpi-ring" style="--accent:{accent};--pct:{pct}%"><span>{pct:.0f}%</span></div>'
    st.markdown(f'''
    <div class="kpi-card" style="--accent:{accent};--soft:{soft}">
      <div class="kpi-icon">{icon}</div>
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-note"><b>↑</b> {note}</div>
      {ring}
    </div>''', unsafe_allow_html=True)

def plot_layout(fig, height=280):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=10,r=10,t=8,b=10),
        font=dict(color="#173253",family="Arial",size=12),
        hoverlabel=dict(bgcolor="#ffffff",font_color="#14213d"),
    )
    fig.update_xaxes(
        gridcolor="#dfeaf5",
        zerolinecolor="#dfeaf5",
        tickfont=dict(size=12,color="#315675"),
        title_font=dict(size=13,color="#315675"),
    )
    fig.update_yaxes(
        gridcolor="#dfeaf5",
        zerolinecolor="#dfeaf5",
        tickfont=dict(size=12,color="#315675"),
        title_font=dict(size=13,color="#315675"),
    )
    return fig

# -----------------------------
# Top navigation
# -----------------------------
nav_cols = st.columns([2.55, .82, 1.02, 1.20, 1.05, 1.00, .72, .88, .72], gap="small")
with nav_cols[0]:
    st.markdown('''<div class=\"wb-brand\"><div class=\"wb-logo\">🤖</div><div><div class=\"wb-brand-title\">WareBot <span>AI</span></div><div class=\"wb-brand-sub\">Autonomous Warehouse Intelligence Platform</div></div></div>''', unsafe_allow_html=True)

nav_items = [
    ("🏠 Dashboard", "Dashboard"),
    ("🤖 Fleet Monitoring", "Fleet Monitoring"),
    ("🛠️ Predictive Maintenance", "Predictive Maintenance"),
    ("📍 Route Optimization", "Route Optimization"),
    ("📦 Inventory & Orders", "Inventory & Orders"),
    ("📊 Analytics", "Analytics"),
    ("💬 AI Assistant", "AI Assistant"),
]

for idx, (label, page) in enumerate(nav_items, start=1):
    if idx >= len(nav_cols)-1:
        break
    with nav_cols[idx]:
        nav_button(label, page, st.session_state.active_page == page)

with nav_cols[-2]:
    st.markdown('<div class="wb-live"><span class="wb-live-dot"></span>SYSTEM LIVE</div>', unsafe_allow_html=True)
with nav_cols[-1]:
    india_now = datetime.now(ZoneInfo("Asia/Kolkata"))
    live_date = india_now.strftime("%b %d, %Y")
    live_time = india_now.strftime("%I:%M %p")
    st.markdown(
        f'<div class="wb-date"><b>{live_date}</b><br>{live_time}</div>',
        unsafe_allow_html=True,
    )

# -----------------------------
# Hero
# -----------------------------
st.markdown(f'''
<div class="wb-hero">
  <div class="wb-hero-copy">
    <div class="wb-eyebrow">WAREBOT AI · WAREHOUSE OPERATIONS</div>
    <div class="wb-hero-title">Warehouse <span>Intelligence</span></div>
    <div class="wb-hero-sub">Real-time fleet health, predictive maintenance, AI task allocation and optimized inventory flow for autonomous warehouse robotics.</div>
  </div>
  <div class="wb-fleet-float">
    <div class="wb-fleet-online"><span class="wb-fleet-dot"></span>SYSTEM OPERATIONAL<small>20 robots monitored</small></div>
    <div class="wb-float-stat"><b>20</b><span>Robots</span></div>
    <div class="wb-float-stat"><b>{HEALTHY}</b><span>Healthy</span></div>
    <div class="wb-float-stat"><b>{ANOMALIES}</b><span>Anomalies</span></div>
  </div>
</div>
<div style="margin:-1px 0 14px;border:1px solid #e5dcae;background:#fff8df;border-radius:13px;padding:9px 14px;color:#806d35;font-size:10px;font-weight:750;box-shadow:0 5px 15px rgba(120,105,55,.05)">⚠️ DEMO MODE · Warehouse telemetry, robot positions and WMS records are simulated for project demonstration.</div>
''', unsafe_allow_html=True)

# -----------------------------
# Dashboard content
# -----------------------------
# WHITE HTML TABLE HELPER
def white_data_table(df, height=500):
    table_html = df.to_html(
        index=False,
        classes="wb-white-html-table",
        border=0
    )

    st.markdown(
        f"""
        <div class="wb-white-table-box" style="max-height:{height}px;">
            {table_html}
        </div>
        """,
        unsafe_allow_html=True
    )




if st.session_state.active_page == "Dashboard":
    k = st.columns(5, gap="small")
    with k[0]: metric_card("🤖","TOTAL ROBOTS",TOTAL,"5% vs last hour","#1687f8","#e8f4ff")
    with k[1]: metric_card("🛡️","HEALTHY ROBOTS",HEALTHY,f"{HEALTHY/TOTAL*100:.1f}% fleet health","#19bf7a","#e6fbf2",HEALTHY/TOTAL*100)
    with k[2]: metric_card("⚠️","ATTENTION",ATTENTION,f"{ATTENTION/TOTAL*100:.1f}% requires review","#f4a313","#fff5df",ATTENTION/TOTAL*100)
    with k[3]: metric_card("〽️","ANOMALIES",ANOMALIES,f"{ANOMALIES/max(len(telemetry),1)*100:.1f}% anomaly rate","#ef4b58","#fff0f1",min(100,ANOMALIES/max(len(telemetry),1)*100))
    with k[4]: metric_card("📦","LOW STOCK ITEMS",LOW_STOCK,f"{LOW_STOCK/max(TOTAL,1)*100:.1f}% reorder needed","#8b5cf6","#f4efff",LOW_STOCK/max(TOTAL,1)*100)

    st.markdown('<div class="section-title">Fleet Intelligence</div><div class="section-sub">Live health, battery distribution and robot-level operating state.</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1.12,1.12,1.08], gap="small")

    # health donut
    with c1:
        st.markdown('<div class="panel"><div class="panel-title">🤖 Fleet Health Distribution</div><div class="panel-sub">Health status of all warehouse robots</div>', unsafe_allow_html=True)
        fig=go.Figure(go.Pie(labels=["Healthy","Needs Attention","Critical"],values=[HEALTHY,WARNING,CRITICAL],hole=.68,marker=dict(colors=["#24c985","#f6a719","#ef4d59"],line=dict(color="white",width=3)),textinfo="percent",textfont=dict(size=11,color="white")))
        fig.add_annotation(text=f"<b>{TOTAL}</b><br><span style='font-size:11px'>Robots</span>",showarrow=False,font=dict(size=22,color="#142947"))
        fig.update_layout(showlegend=True,legend=dict(orientation="v",x=.96,y=.5,xanchor="left",font=dict(size=10,color="#36516f")),margin=dict(l=5,r=90,t=10,b=5),height=260,paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,width="stretch",config={"displayModeBar":False})
        st.markdown('</div>',unsafe_allow_html=True)

    # battery distribution
    with c2:
        st.markdown('<div class="panel"><div class="panel-title">🔋 Battery Level Distribution</div><div class="panel-sub">Current battery levels across fleet</div>', unsafe_allow_html=True)
        bins=[0,20,40,60,80,100]
        labels=["0–20%","20–40%","40–60%","60–80%","80–100%"]
        cats=pd.cut(latest["battery"],bins=bins,labels=labels,include_lowest=True)
        counts=cats.value_counts().reindex(labels,fill_value=0)
        fig=go.Figure(go.Bar(x=labels,y=counts.values,text=counts.values,textposition="outside",marker_color=["#ef4d59","#f59e0b","#f8c94b","#4d99f5","#19c77a"],marker_line_width=0))
        fig=plot_layout(fig,285); fig.update_yaxes(title="Number of Robots",dtick=1,title_font=dict(size=14,color="#315675"),tickfont=dict(size=12,color="#315675"),title_standoff=10); fig.update_xaxes(title="Battery Level",title_font=dict(size=14,color="#315675"),tickfont=dict(size=12,color="#315675"),title_standoff=10)
        st.plotly_chart(fig,width="stretch",config={"displayModeBar":False})
        st.markdown('</div>',unsafe_allow_html=True)

    # status table
    with c3:
        st.markdown('<div class="panel"><div class="panel-title">🤖 Robot Status Overview</div><div class="panel-sub">Current status of warehouse robots</div>', unsafe_allow_html=True)
        show=latest.head(5)[["robot_id","battery","health_score","status"]].copy()
        show["task"]=["Picking","Transport","Idle","Picking","Maintenance"][:len(show)]
        show.columns=["ROBOT ID","BATTERY","HEALTH","STATUS","TASK"]
        show["BATTERY"]=show["BATTERY"].round(0).astype(int).astype(str)+"%"
        show["HEALTH"]=show["HEALTH"].round(0).astype(int).astype(str)+"%"
        rows=[]
        for _,r in show.iterrows():
            status=str(r["STATUS"])
            cls={"Healthy":"healthy","Warning":"warning","Critical":"critical"}.get(status,"healthy")
            battery=float(str(r["BATTERY"]).replace("%",""))
            rows.append(
                f"<tr><td><b>{r['ROBOT ID']}</b></td>"
                f"<td><span class='wb-bar'><i style='width:{battery:.0f}%'></i></span>{r['BATTERY']}</td>"
                f"<td>{r['HEALTH']}</td>"
                f"<td><span class='wb-status {cls}'>● {status}</span></td>"
                f"<td>{r['TASK']}</td></tr>"
            )
        table_html = (
            '<div class="wb-table-wrap"><table class="wb-table">'
            '<thead><tr><th>ROBOT ID</th><th>BATTERY</th><th>HEALTH</th><th>STATUS</th><th>TASK</th></tr></thead>'
            '<tbody>' + ''.join(rows) + '</tbody></table></div>'
        )
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    # Digital twin / allocation / alerts
    st.markdown('<div class="section-title">Warehouse Operations</div><div class="section-sub">Live digital twin, AI task allocation and operational alerts.</div>', unsafe_allow_html=True)
    mcol,tcol,acol=st.columns([1.65,1.0,.82],gap="small")

    with mcol:
        st.markdown('<div class="panel"><div class="panel-title">📍 Warehouse Digital Twin</div><div class="panel-sub">Live robot positions and warehouse layout</div>',unsafe_allow_html=True)
        robot_pos={"R01":(0,0),"R02":(2,2),"R03":(4,2),"R04":(6,0),"R05":(8,1),"R06":(1,4),"R07":(3,5),"R08":(5,4),"R09":(7,3),"R10":(9,0),"R11":(0,6),"R12":(2,7),"R13":(4,6),"R14":(6,5),"R15":(8,4),"R16":(1,8),"R17":(3,8),"R18":(5,8),"R19":(7,8),"R20":(9,5)}
        fig=go.Figure()
        obsx=[];obsy=[]
        for y,row in enumerate(WAREHOUSE_GRID):
            for x,v in enumerate(row):
                if v==1: obsx.append(x);obsy.append(y)
        fig.add_trace(go.Scatter(x=obsx,y=obsy,mode="markers",marker=dict(size=23,color="#b8c9dc",symbol="square",line=dict(color="#9fb5cc",width=1)),name="Shelf / Rack",hoverinfo="skip"))
        station_colors={"P1":"#1687f8","P2":"#1687f8","P3":"#1687f8","PACK":"#8b5cf6"}
        for s,pos in STATIONS.items():
            fig.add_trace(go.Scatter(x=[pos[0]],y=[pos[1]],mode="markers+text",text=[s],textposition="top center",marker=dict(size=22,color=station_colors.get(s,"#1687f8"),symbol="diamond",line=dict(color="white",width=2)),name=s))
        colors={"Healthy":"#18c77a","Warning":"#f5a623","Critical":"#ef4d59"}
        for rid,pos in robot_pos.items():
            row=latest[latest.robot_id==rid]
            status=row.iloc[0]["status"] if not row.empty else "Healthy"
            fig.add_trace(go.Scatter(x=[pos[0]],y=[pos[1]],mode="markers+text",text=[rid],textposition="bottom center",marker=dict(size=17,color=colors.get(status,"#1687f8"),line=dict(color="white",width=2)),name=rid,showlegend=False,hovertemplate=f"<b>{rid}</b><br>Status: {status}<extra></extra>"))
        # planned route to P3
        route=[(6,0),(6,1),(6,2),(7,2),(8,2),(8,3),(9,3),(9,4),(9,5)]
        fig.add_trace(go.Scatter(x=[p[0] for p in route],y=[p[1] for p in route],mode="lines",line=dict(color="#1687f8",width=4,dash="dot"),name="Planned Route"))
        fig=plot_layout(fig,300);fig.update_xaxes(showgrid=True,dtick=1,range=[-1,10],showticklabels=False);fig.update_yaxes(showgrid=True,dtick=1,range=[9,-1],showticklabels=False)
        fig.update_layout(showlegend=False,margin=dict(l=4,r=4,t=8,b=4))
        st.markdown('<div class="wb-map-panel">',unsafe_allow_html=True)
        st.plotly_chart(fig,width="stretch",config={"displayModeBar":False})
        st.markdown('''<div class="legend-row"><span><i class="legend-dot" style="background:#18c77a"></i>Healthy</span><span><i class="legend-dot" style="background:#f5a623"></i>Attention</span><span><i class="legend-dot" style="background:#ef4d59"></i>Critical</span><span><i class="route-line"></i>Planned Route</span><span>▣ Shelf / Rack</span></div></div></div>''',unsafe_allow_html=True)

    with tcol:
        st.markdown('<div class="panel" style="padding:18px"><div class="panel-title">🎯 Task Allocation</div><div class="panel-sub">AI recommended task assignment</div>',unsafe_allow_html=True)
        robot_ids=latest.robot_id.tolist()
        chosen_robot=st.radio("Select Robot",robot_ids,index=robot_ids.index("R04") if "R04" in robot_ids else 0,key="alloc_robot",label_visibility="visible",horizontal=True)

        target=st.radio("Select Target Station",["P1","P2","P3"],index=2,key="alloc_target",label_visibility="visible",horizontal=True)
        if st.button("🔍 Get AI Recommendation",key="ai_recommend",width="stretch"):
            try:
                resp=requests.get("https://warebot-ai.onrender.com/mqtt/allocate-task",timeout=10)
                data=resp.json() if resp.ok else {}
                alloc=data.get("allocation",{}); route_data=data.get("route_optimization",{})
                rec_robot=alloc.get("robot_id",chosen_robot); distance=route_data.get("distance",9); score=route_data.get("score",36.6)
            except Exception:
                rec_robot=chosen_robot; distance=9; score=36.6
            st.session_state.alloc_result=(rec_robot,target,distance,score)
        rec=st.session_state.get("alloc_result",("R04","P3",9,36.6))
        st.markdown(f'''<div class="ai-box"><div class="ai-box-title">⭐ AI RECOMMENDATION</div><div class="ai-main"><span class="ai-stat">RECOMMENDED ROBOT<strong>{rec[0]}</strong></span><span class="ai-stat">TARGET STATION<strong>{rec[1]}</strong></span></div><div style="display:flex;justify-content:space-between;margin-top:12px"><span class="ai-stat">ESTIMATED STEPS<b>{rec[2]} steps</b></span><span class="ai-stat">OPTIMIZATION SCORE<b>{rec[3]}</b></span></div></div>''',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    with acol:
        st.markdown('<div class="panel"><div style="display:flex;justify-content:space-between;align-items:center"><div><div class="panel-title">🔔 Recent Alerts</div><div class="panel-sub">Latest operational events</div></div><span style="color:#1687f8;font-size:10px;font-weight:800">View All</span></div>',unsafe_allow_html=True)
        alerts=[
            ("#ef4d59","🔴","R05 - Low Battery","Battery level at 12%. Please recharge.","5 min ago"),
            ("#f5a623","⚠️","R03 - High Temperature","Temperature at 68°C (threshold: 60°C).","12 min ago"),
            ("#1687f8","🔵","Congestion Detected","High traffic at P2 station.","18 min ago"),
            ("#8b5cf6","🟣","Low Stock Alert","Item A1001 below reorder level.","25 min ago"),
            ("#ef4d59","🔴","R01 - Vibration Anomaly","Unusual vibration pattern detected.","32 min ago"),
        ]
        for color,icon,title,msg,t in alerts:
            st.markdown(f'''<div class="alert-card"><div class="alert-icon" style="background:{color}16">{icon}</div><div><div class="alert-title" style="color:{color}">{title}</div><div class="alert-msg">{msg}</div></div><div class="alert-time">{t}</div></div>''',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

elif st.session_state.active_page == "Fleet Monitoring":
    st.markdown('<div class="section-title">Fleet Monitoring</div><div class="section-sub">Robot-level telemetry, battery and health status.</div>',unsafe_allow_html=True)
    a,b=st.columns([1.1,1])
    with a:
        st.markdown('<div class="panel"><div class="panel-title">Robot Status Overview</div>',unsafe_allow_html=True)
        df=latest[["robot_id","battery","health_score","status","temperature","vibration","motor_current","navigation_errors"]].copy()
        df.columns=["Robot","Battery %","Health","Status","Temperature","Vibration","Motor Current","Nav Errors"]
        table_rows = ""
        for _, row in df.round(2).iterrows():
            status = str(row["Status"])
            status_class = "critical" if status == "Critical" else "healthy"

            table_rows += f"""
            <tr>
                <td>{row["Robot"]}</td>
                <td>{row["Battery %"]}</td>
                <td>{row["Health"]}</td>
                <td class="{status_class}">{status}</td>
                <td>{row["Temperature"]}</td>
                <td>{row["Vibration"]}</td>
                <td>{row["Motor Current"]}</td>
                <td>{row["Nav Errors"]}</td>
            </tr>
            """

        st.html(f"""
        <div class="wb-fleet-table-wrap">
            <table class="wb-fleet-table">
                <thead>
                    <tr>
                        <th>Robot</th>
                        <th>Battery %</th>
                        <th>Health</th>
                        <th>Status</th>
                        <th>Temperature</th>
                        <th>Vibration</th>
                        <th>Motor Current</th>
                        <th>Nav Errors</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        """)
        st.markdown('</div>',unsafe_allow_html=True)
    with b:
        selected=st.radio("Select Robot",sorted(telemetry.robot_id.unique()),index=0,key="fleet_robot",horizontal=True)
        hist=telemetry[telemetry.robot_id==selected].sort_values("timestamp")
        fig=go.Figure(go.Scatter(x=hist.timestamp,y=hist.battery,mode="lines+markers",line=dict(color="#1687f8",width=3),marker=dict(size=4)))
        fig=plot_layout(fig,360);fig.update_yaxes(title="Battery %",range=[0,100]);fig.update_xaxes(title="Time")
        st.markdown('<div class="panel"><div class="panel-title">Battery Performance</div>',unsafe_allow_html=True);st.plotly_chart(fig,width="stretch",config={"displayModeBar":False});st.markdown('</div>',unsafe_allow_html=True)
        fig2=go.Figure(go.Scatter(x=hist.timestamp,y=hist.health_score,mode="lines",line=dict(color="#19bf7a",width=3),fill="tozeroy",fillcolor="rgba(25,191,122,.10)"));fig2=plot_layout(fig2,250);fig2.update_yaxes(title="Health Score",range=[0,100]);st.plotly_chart(fig2,width="stretch",config={"displayModeBar":False})


elif st.session_state.active_page == "Predictive Maintenance":
    st.markdown('<div class="section-title">Predictive Maintenance</div><div class="section-sub">AI-driven robot health and maintenance risk intelligence.</div>',unsafe_allow_html=True)
    p1,p2,p3,p4=st.columns(4)
    for col,label,val,sub in [(p1,"MODEL ACCURACY",f"{model_accuracy*100:.1f}%","XGBoost maintenance model"),(p2,"ANOMALY THRESHOLD",f"{anomaly_threshold:.3f}","Detected from telemetry"),(p3,"MAINTENANCE FLAGS",ATTENTION,"Requires review"),(p4,"AVG HEALTH",f"{AVG_HEALTH:.1f}/100","Fleet health score")]:
        with col: st.markdown(f'<div class="kpi-card" style="--accent:#1687f8;--soft:#eaf5ff"><div class="kpi-label">{label}</div><div class="kpi-value">{val}</div><div class="kpi-note">{sub}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="panel" style="margin-top:12px"><div class="panel-title">Maintenance Risk Overview</div>',unsafe_allow_html=True)
    risk=latest[["robot_id","battery","temperature","vibration","motor_current","navigation_errors","health_score","status"]].copy();risk["risk"]=100-risk["health_score"];white_data_table(risk.round(2),500);st.markdown('</div>',unsafe_allow_html=True)

elif st.session_state.active_page == "Route Optimization":
    st.markdown('<div class="section-title">Route Optimization</div><div class="section-sub">AI task allocation with live backend recommendation and shortest-path routing.</div>',unsafe_allow_html=True)
    target=st.radio("Target Station",["P1","P2","P3"],index=2,key="route_target",horizontal=True)
    if st.button("⚡ Run AI Allocation",key="route_run"):
        try:
            resp=requests.get("https://warebot-ai.onrender.com/mqtt/allocate-task",timeout=10)
            data=resp.json(); alloc=data.get("allocation",{});route=data.get("route_optimization",{})
            st.session_state.route_result=(alloc,route)
        except Exception as e: st.error(f"Backend unavailable: {e}")
    alloc,route=st.session_state.get("route_result",({},{}))
    q1,q2,q3=st.columns(3)
    with q1: st.metric("Recommended Robot",alloc.get("robot_id","R04"))
    with q2: st.metric("Route Distance",f"{route.get('distance',9)} steps")
    with q3: st.metric("Optimization Score",route.get("score",36.6))
    st.markdown('<div class="panel" style="margin-top:12px"><div class="panel-title">Optimized Path</div>',unsafe_allow_html=True)
    st.code(str(route.get("path",[(6,0),(6,1),(6,2),(7,2),(8,2),(8,3),(9,3),(9,4),(9,5)])),language="text");st.markdown('</div>',unsafe_allow_html=True)

elif st.session_state.active_page == "Inventory & Orders":
    st.markdown('<div class="section-title">Inventory & Orders</div><div class="section-sub">Warehouse inventory, low-stock items and active order flow.</div>',unsafe_allow_html=True)
    inv=get_inventory();orders=get_orders()
    a,b=st.columns(2)
    with a:
        st.markdown('<div class="panel"><div class="panel-title">Inventory</div>',unsafe_allow_html=True);white_data_table(inv,500);st.markdown('</div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><div class="panel-title">Active Orders</div>',unsafe_allow_html=True);white_data_table(orders,500);st.markdown('</div>',unsafe_allow_html=True)

elif st.session_state.active_page == "Analytics":
    st.markdown('<div class="section-title">Warehouse Analytics</div><div class="section-sub">Fleet, anomaly and operational distribution insights.</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        status=latest.status.value_counts().rename_axis("Status").reset_index(name="Robots")
        fig=px.bar(status,x="Status",y="Robots",text="Robots",color="Status",color_discrete_map={"Healthy":"#19c77a","Warning":"#f5a623","Critical":"#ef4d59"});fig=plot_layout(fig,340);st.markdown('<div class="panel"><div class="panel-title">Robot Status</div>',unsafe_allow_html=True);st.plotly_chart(fig,width="stretch",config={"displayModeBar":False});st.markdown('</div>',unsafe_allow_html=True)
    with c2:
        fig=px.scatter(latest,x="battery",y="health_score",size="health_score",color="status",hover_name="robot_id",color_discrete_map={"Healthy":"#19c77a","Warning":"#f5a623","Critical":"#ef4d59"});fig=plot_layout(fig,340);fig.update_xaxes(title="Battery %");fig.update_yaxes(title="Health Score");st.markdown('<div class="panel"><div class="panel-title">Battery vs Health</div>',unsafe_allow_html=True);st.plotly_chart(fig,width="stretch",config={"displayModeBar":False});st.markdown('</div>',unsafe_allow_html=True)

elif st.session_state.active_page == "AI Assistant":
    st.markdown('<div class="section-title">AI Operations Assistant</div><div class="section-sub">Ask about robot health, maintenance, anomalies, inventory or fleet status.</div>',unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="panel-title">🤖 WareBot Operations Assistant</div>',unsafe_allow_html=True)
        question=st.chat_input("Ask WareBot about fleet operations...")
        if question:
            q=question.lower()
            if "maintenance" in q or "repair" in q or "problem" in q:
                crit=latest[latest.status=="Critical"]
                answer=f"{len(crit)} critical robot(s) need attention: {', '.join(crit.robot_id.tolist())}. Check battery, vibration, temperature and motor current." if len(crit) else "No robots are currently classified as Critical."
            elif "inventory" in q or "stock" in q:
                answer=f"{LOW_STOCK} inventory item(s) are below reorder level."
            elif "anomaly" in q:
                answer=f"The AI telemetry pipeline detected {ANOMALIES} anomalies."
            elif "robot" in q or "fleet" in q:
                answer=f"Fleet status: {TOTAL} robots monitored, {HEALTHY} healthy, {ATTENTION} requiring attention."
            else:
                answer="I can help with robot health, maintenance, anomalies, inventory and fleet status."
            st.chat_message("assistant",avatar="🤖").write(answer)

st.markdown(
    f'<div style="text-align:center;color:#5d7590;font-size:10px;font-weight:600;padding-top:18px">'
    f'WareBot AI v1.0 · Autonomous Warehouse Intelligence · Prototype using simulated warehouse telemetry · '
    f'Live time: {datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%I:%M:%S %p")}</div>',
    unsafe_allow_html=True,
)
