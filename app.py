# ============================================================
# WAREBOT AI
# Premium Warehouse Intelligence Command Center
# ============================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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
)

from optimization import (
    WAREHOUSE_GRID,
    STATIONS,
    allocate_task,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="WareBot AI | Command Center",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# THEME
# ============================================================

if "theme" not in st.session_state:
    st.session_state.theme = "dark"


theme_choice = st.sidebar.radio(
    "APPEARANCE",
    ["🌙 Dark", "☀️ Light"],
    index=0 if st.session_state.theme == "dark" else 1,
)

st.session_state.theme = (
    "dark"
    if theme_choice == "🌙 Dark"
    else "light"
)

is_dark = st.session_state.theme == "dark"


# ============================================================
# THEME COLORS
# ============================================================

if is_dark:

    BG = "#070b14"
    SIDEBAR = "#090e18"
    CARD = "#0d1421"
    CARD_2 = "#101827"

    TEXT = "#f8fafc"
    TEXT_MUTED = "#94a3b8"

    BORDER = "rgba(148,163,184,0.11)"
    BORDER_HOVER = "rgba(34,211,238,0.40)"

    ACCENT = "#22d3ee"
    ACCENT_2 = "#6366f1"

    GRID = "rgba(148,163,184,0.08)"
    PLOT_BG = "#060b15"

else:

    BG = "#f4f7fb"
    SIDEBAR = "#ffffff"
    CARD = "#ffffff"
    CARD_2 = "#f8fafc"

    TEXT = "#172033"
    TEXT_MUTED = "#475569"

    BORDER = "rgba(15,23,42,0.10)"
    BORDER_HOVER = "rgba(8,145,178,0.40)"

    ACCENT = "#0891b2"
    ACCENT_2 = "#4f46e5"

    GRID = "rgba(15,23,42,0.10)"
    PLOT_BG = "#ffffff"


# ============================================================
# PREMIUM CSS
# ============================================================

CSS = """
<style>

/* =========================================================
   WAREBOT AI — SIDEBAR INDEPENDENT SCROLL FIX
   ========================================================= */

[data-testid="stSidebar"] {
    height: 100vh !important;
    overflow: hidden !important;
}

[data-testid="stSidebar"] > div:first-child {
    height: 100vh !important;
    max-height: 100vh !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    scrollbar-width: thin !important;
    scrollbar-color: rgba(100,116,139,.55) transparent !important;
}

[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar {
    width: 6px !important;
}

[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-track {
    background: transparent !important;
}

[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb {
    background: rgba(100,116,139,.45) !important;
    border-radius: 10px !important;
}

[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb:hover {
    background: rgba(34,211,238,.80) !important;
}

[data-testid="stSidebarContent"] {
    height: auto !important;
    min-height: 100% !important;
    overflow: visible !important;
}

[data-testid="stSidebarUserContent"] {
    padding-bottom: 24px !important;
}

/* Compact sidebar spacing so all controls remain comfortable at 100% zoom */
[data-testid="stSidebar"] .stButton > button {
    min-height: 42px !important;
    margin-bottom: 6px !important;
}

[data-testid="stSidebar"] hr {
    margin: 18px 0 !important;
}

/* Light mode */
@media (prefers-color-scheme: light) {
    [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb {
        background: rgba(71,85,105,.35) !important;
    }
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
    [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb {
        background: rgba(148,163,184,.35) !important;
    }
}



html, body {
    color-scheme: __COLOR_SCHEME__ !important;
    background: __BG__ !important;
}

body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background:
        radial-gradient(circle at 12% 0%, rgba(34,211,238,0.06), transparent 28%),
        radial-gradient(circle at 90% 5%, rgba(99,102,241,0.06), transparent 28%),
        __BG__ !important;
    color: __TEXT__ !important;
}

/* Streamlit top header / black strip fix */
header[data-testid="stHeader"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    background: __BG__ !important;
    background-color: __BG__ !important;
    color: __TEXT__ !important;
    border: 0 !important;
    box-shadow: none !important;
}

/* Main layout */
.block-container {
    max-width: 1500px !important;
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
}

#MainMenu, footer {
    visibility: hidden !important;
}

h1, h2, h3, h4, h5, h6 {
    color: __TEXT__ !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"],
label {
    color: __TEXT_MUTED__ !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: __SIDEBAR__ !important;
    background-color: __SIDEBAR__ !important;
    border-right: 1px solid __BORDER__ !important;
}

[data-testid="stSidebar"] * {
    color: __TEXT__ !important;
}

[data-testid="stSidebar"] .stButton > button,
.stButton > button {
    background: __CARD_2__ !important;
    background-color: __CARD_2__ !important;
    color: __TEXT__ !important;
    border: 1px solid __BORDER__ !important;
    border-radius: 10px !important;
    font-weight: 650 !important;
}

[data-testid="stSidebar"] .stButton > button:hover,
.stButton > button:hover {
    color: __ACCENT__ !important;
    border-color: __BORDER_HOVER__ !important;
}

/* ============================================================
   PREMIUM GLOW / DEPTH
   ============================================================ */

[data-testid="stMetric"] {
    position: relative;
    transition: transform 0.22s ease, box-shadow 0.22s ease,
                border-color 0.22s ease !important;
    box-shadow:
        0 8px 24px rgba(15,23,42,0.06),
        0 0 0 1px rgba(34,211,238,0.025) !important;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: rgba(34,211,238,0.30) !important;
    box-shadow:
        0 12px 30px rgba(15,23,42,0.10),
        0 0 22px rgba(34,211,238,0.10) !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    position: relative;
    transition: box-shadow 0.22s ease, border-color 0.22s ease !important;
    box-shadow: 0 8px 25px rgba(15,23,42,0.045) !important;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(34,211,238,0.24) !important;
    box-shadow:
        0 10px 30px rgba(15,23,42,0.08),
        0 0 24px rgba(34,211,238,0.06) !important;
}

[data-testid="stPlotlyChart"] {
    transition: box-shadow 0.22s ease, border-color 0.22s ease !important;
}

[data-testid="stPlotlyChart"]:hover {
    border-color: rgba(34,211,238,0.25) !important;
    box-shadow: 0 0 24px rgba(34,211,238,0.07) !important;
}

/* Sidebar system status cards */
.wb-status-card {
    display: flex;
    align-items: center;
    gap: 7px;
    width: 100%;
    box-sizing: border-box;
    margin: 0 0 0.42rem 0;
    padding: 0.68rem 0.78rem;
    border-radius: 11px;
    border: 1px solid rgba(34,197,94,0.16);
    background: rgba(34,197,94,0.12);
    color: __TEXT__;
    box-shadow: 0 5px 16px rgba(15,23,42,0.04);
    transition: transform 0.16s ease, box-shadow 0.16s ease;
}

.wb-status-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(15,23,42,0.08);
}

.wb-status-dot {
    color: #16a34a;
    font-size: 12px;
}

.wb-status-name, .wb-status-state {
    color: __TEXT__;
    font-size: 0.92rem;
}

.wb-status-name { font-weight: 500; }
.wb-status-state { font-weight: 650; }
.wb-status-dash { color: __TEXT_MUTED__; }

.wb-sidebar-footer { padding: 0.15rem 0 0.25rem 0; }
.wb-version {
    color: __TEXT_MUTED__;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 0.35rem;
}
.wb-footer-sub { color: __TEXT_MUTED__; font-size: 0.78rem; }

/* Premium LIVE badge */
[data-testid="stAlert"] {
    box-shadow: 0 0 18px rgba(34,197,94,0.08) !important;
}

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button,
.stButton > button {
    transition: all 0.2s ease !important;
}

[data-testid="stSidebar"] .stButton > button:hover,
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 0 18px rgba(34,211,238,0.10) !important;
}

/* ============================================================
   SELECTBOX — FULL WHITE CONTROL + PREMIUM GLOW
   ============================================================ */

[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div[role="button"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div[aria-haspopup="listbox"] {
    background: __SELECT_BG__ !important;
    background-color: __SELECT_BG__ !important;
    color: __SELECT_TEXT__ !important;
    border-color: __SELECT_BORDER__ !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    min-height: 44px !important;
    transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
}

/* Remove the black arrow segment in Light mode */
[data-testid="stSelectbox"] [data-baseweb="select"] > div > div:last-child,
[data-testid="stSelectbox"] [data-baseweb="select"] [aria-hidden="true"] {
    background: __SELECT_BG__ !important;
    background-color: __SELECT_BG__ !important;
}

/* Premium focus glow */
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div {
    border-color: #06b6d4 !important;
    box-shadow:
        0 0 0 2px rgba(6,182,212,0.13),
        0 0 22px rgba(6,182,212,0.13) !important;
}

/* ============================================================
   DROPDOWN POPUP — PREMIUM FLOATING PANEL
   ============================================================ */

div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="menu"],
div[role="listbox"],
ul[role="listbox"] {
    border-radius: 14px !important;
    box-shadow:
        0 18px 45px rgba(15,23,42,0.16),
        0 0 24px rgba(6,182,212,0.07) !important;
}

div[role="option"],
li[role="option"] {
    min-height: 42px !important;
    border-radius: 9px !important;
    margin: 3px 6px !important;
    transition: background 0.15s ease, padding-left 0.15s ease !important;
}

div[role="option"]:hover,
li[role="option"]:hover {
    padding-left: 14px !important;
    box-shadow: inset 3px 0 0 #06b6d4 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, __CARD_2__, __CARD__) !important;
    border: 1px solid __BORDER__ !important;
    border-radius: 16px !important;
    padding: 18px 20px !important;
    min-height: 125px;
}

[data-testid="stMetricLabel"] {
    color: __TEXT_MUTED__ !important;
}

[data-testid="stMetricValue"] {
    color: __TEXT__ !important;
}

/* Containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: __CARD__ !important;
    background-color: __CARD__ !important;
    border: 1px solid __BORDER__ !important;
    border-radius: 16px !important;
}

/* ============================================================
   SELECTBOX — LIGHT MODE / DARK MODE HARD FIX
   ============================================================ */

[data-testid="stSelectbox"],
[data-testid="stSelectbox"] label {
    color: __TEXT__ !important;
}

/* Closed selectbox outer control */
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div[role="button"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div[aria-haspopup="listbox"],
[data-testid="stSelectbox"] [role="combobox"] {
    background: __SELECT_BG__ !important;
    background-color: __SELECT_BG__ !important;
    color: __SELECT_TEXT__ !important;
    border-radius: 12px !important;
    border: 1px solid __SELECT_BORDER__ !important;
    box-shadow: none !important;
    opacity: 1 !important;
    -webkit-text-fill-color: __SELECT_TEXT__ !important;
}

/* Every child inside closed selectbox */
[data-testid="stSelectbox"] [data-baseweb="select"] > div *,
[data-testid="stSelectbox"] [data-baseweb="select"] > div[role="button"] *,
[data-testid="stSelectbox"] [role="combobox"] * {
    background: transparent !important;
    background-color: transparent !important;
    color: __SELECT_TEXT__ !important;
    -webkit-text-fill-color: __SELECT_TEXT__ !important;
    opacity: 1 !important;
}

/* Selected text / input */
[data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] input {
    background: transparent !important;
    color: __SELECT_TEXT__ !important;
    -webkit-text-fill-color: __SELECT_TEXT__ !important;
    caret-color: __SELECT_TEXT__ !important;
}

/* Dropdown arrow */
[data-testid="stSelectbox"] [data-baseweb="select"] svg {
    color: __SELECT_TEXT__ !important;
    fill: __SELECT_TEXT__ !important;
    stroke: __SELECT_TEXT__ !important;
}

/* Premium cyan focus */
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus,
[data-testid="stSelectbox"] [role="combobox"]:focus {
    background: __SELECT_BG__ !important;
    background-color: __SELECT_BG__ !important;
    color: __SELECT_TEXT__ !important;
    border: 1px solid #06b6d4 !important;
    box-shadow: 0 0 0 2px rgba(6,182,212,0.16) !important;
    outline: none !important;
}

/* ============================================================
   SELECTBOX POPUP — WHITE IN LIGHT / DARK IN DARK
   ============================================================ */

div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="menu"],
div[role="listbox"],
ul[role="listbox"] {
    background: __SELECT_BG__ !important;
    background-color: __SELECT_BG__ !important;
    color: __SELECT_TEXT__ !important;
    border-color: __SELECT_BORDER__ !important;
    color-scheme: __COLOR_SCHEME__ !important;
}

div[data-baseweb="menu"] *,
div[role="listbox"] *,
ul[role="listbox"] * {
    color: __SELECT_TEXT__ !important;
    -webkit-text-fill-color: __SELECT_TEXT__ !important;
}

div[role="option"],
li[role="option"] {
    background: __SELECT_BG__ !important;
    background-color: __SELECT_BG__ !important;
    color: __SELECT_TEXT__ !important;
    -webkit-text-fill-color: __SELECT_TEXT__ !important;
}

div[role="option"] *,
li[role="option"] * {
    background: transparent !important;
    color: __SELECT_TEXT__ !important;
    -webkit-text-fill-color: __SELECT_TEXT__ !important;
}

div[role="option"]:hover,
li[role="option"]:hover {
    background: __OPTION_HOVER__ !important;
    background-color: __OPTION_HOVER__ !important;
    color: __SELECT_TEXT__ !important;
}

div[role="option"][aria-selected="true"],
li[role="option"][aria-selected="true"] {
    background: __OPTION_SELECTED__ !important;
    background-color: __OPTION_SELECTED__ !important;
    color: __SELECT_TEXT__ !important;
}

/* Normal text inputs */
div[data-baseweb="input"] > div {
    background: __CARD__ !important;
    background-color: __CARD__ !important;
    border-color: __BORDER__ !important;
}

input {
    color: __TEXT__ !important;
    -webkit-text-fill-color: __TEXT__ !important;
}

/* Charts / tables */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid __BORDER__ !important;
}

[data-testid="stPlotlyChart"] {
    border-radius: 14px !important;
    overflow: hidden;
    border: 1px solid __BORDER__ !important;
}

[data-testid="stAlert"] {
    border-radius: 11px !important;
}

hr {
    border-color: __BORDER__ !important;
}

/* ============================================================
   CHATBOT — CLEAN INPUT, NO EXTRA DARK BOX
   ============================================================ */

[data-testid="stChatInput"] {
    color-scheme: __COLOR_SCHEME__ !important;
    background: transparent !important;
    background-color: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
}

[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] form,
[data-testid="stChatInput"] [data-baseweb="textarea"] {
    background: __CHAT_BG__ !important;
    background-color: __CHAT_BG__ !important;
    border: 1px solid __BORDER__ !important;
    border-radius: 14px !important;
    box-shadow: none !important;
}

[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus {
    background: transparent !important;
    background-color: transparent !important;
    color: __TEXT__ !important;
    -webkit-text-fill-color: __TEXT__ !important;
    box-shadow: none !important;
    outline: none !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: __TEXT_MUTED__ !important;
    -webkit-text-fill-color: __TEXT_MUTED__ !important;
    opacity: 1 !important;
}

[data-testid="stChatInput"] button {
    background: __ACCENT__ !important;
    color: #ffffff !important;
    border: 0 !important;
    box-shadow: none !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: 0 !important;
    border-radius: 12px !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 7px;
}

::-webkit-scrollbar-track {
    background: __BG__;
}

::-webkit-scrollbar-thumb {
    background: #64748b;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: __ACCENT__;
}



/* ============================================================
   WAREBOT AI — EXTRA PREMIUM POLISH
   ============================================================ */

/* Premium page depth */
[data-testid="stAppViewContainer"] {
    position: relative !important;
}

.block-container {
    position: relative !important;
}

/* Subtle top accent line */
.block-container::before {
    content: "";
    display: block;
    height: 2px;
    width: 100%;
    margin-bottom: 22px;
    border-radius: 999px;
    background: linear-gradient(
        90deg,
        __ACCENT__,
        __ACCENT_2__,
        transparent 88%
    ) !important;
    opacity: 0.9;
}

/* Header typography */
[data-testid="stAppViewContainer"] h1 {
    font-weight: 850 !important;
    letter-spacing: -1.5px !important;
    text-shadow: 0 3px 18px rgba(34,211,238,0.08);
}

/* Premium section headings */
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3 {
    font-weight: 760 !important;
    letter-spacing: -0.5px !important;
}

/* KPI cards — stronger premium treatment */
[data-testid="stMetric"] {
    overflow: hidden !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
}

[data-testid="stMetric"]::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(
        90deg,
        __ACCENT__,
        __ACCENT_2__
    ) !important;
    opacity: 0.95;
}

[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.7px !important;
}

[data-testid="stMetricValue"] {
    font-size: 2.05rem !important;
    font-weight: 850 !important;
    letter-spacing: -1px !important;
    line-height: 1.05 !important;
}

/* Delta/status pill */
[data-testid="stMetricDelta"] {
    font-weight: 650 !important;
    border-radius: 999px !important;
}

/* Premium bordered content cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    overflow: hidden !important;
}

[data-testid="stVerticalBlockBorderWrapper"]::before {
    content: "";
    display: block;
    height: 1px;
    margin: -1px 18px 0 18px;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(34,211,238,0.45),
        transparent
    );
    opacity: 0.65;
}

/* Smooth card lift */
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-2px) !important;
}

/* Premium sidebar spacing + full status visibility */
[data-testid="stSidebar"] .block-container {
    padding-top: 0.85rem !important;
    padding-bottom: 0.8rem !important;
}

[data-testid="stSidebar"] hr {
    margin: 0.65rem 0 !important;
}

[data-testid="stSidebar"] [data-testid="stButton"] {
    margin-bottom: 0.32rem !important;
}

[data-testid="stSidebar"] [data-testid="stButton"] > button {
    min-height: 42px !important;
    padding-top: 0.35rem !important;
    padding-bottom: 0.35rem !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] {
    margin-bottom: 0.15rem !important;
}

[data-testid="stSidebar"] [data-testid="stAlert"] {
    margin-bottom: 0.45rem !important;
    padding: 0.55rem 0.7rem !important;
    border-radius: 10px !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    margin-bottom: 0.35rem !important;
}

/* Premium sidebar branding */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-weight: 800 !important;
}

[data-testid="stSidebar"] [data-testid="stButton"] > button {
    min-height: 44px !important;
    letter-spacing: 0.1px !important;
}

/* Premium LIVE badge */
[data-testid="stAlert"] {
    border: 1px solid rgba(34,197,94,0.22) !important;
    border-radius: 999px !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
}

/* Selectbox premium depth without changing its colors */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    box-shadow:
        0 6px 18px rgba(15,23,42,0.06),
        inset 0 1px 0 rgba(255,255,255,0.08) !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {
    border-color: __ACCENT__ !important;
    box-shadow:
        0 8px 22px rgba(8,145,178,0.10),
        0 0 0 1px rgba(8,145,178,0.08) !important;
}

/* Dropdown floating animation feel */
div[data-baseweb="popover"] {
    animation: warebotDropdown 0.14s ease-out !important;
}

@keyframes warebotDropdown {
    from {
        opacity: 0;
        transform: translateY(-5px) scale(0.99);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

/* Buttons */
.stButton > button {
    transition:
        transform 0.18s ease,
        box-shadow 0.18s ease,
        border-color 0.18s ease,
        background 0.18s ease !important;
}

.stButton > button:active {
    transform: translateY(0) scale(0.985) !important;
}

/* Charts get a polished frame */
[data-testid="stPlotlyChart"] {
    background: transparent !important;
    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease !important;
}

[data-testid="stPlotlyChart"]:hover {
    transform: translateY(-1px) !important;
}

/* Tables */
[data-testid="stDataFrame"] {
    box-shadow: 0 8px 24px rgba(15,23,42,0.045) !important;
}

/* Chatbot card */
[data-testid="stChatMessage"] {
    padding-top: 0.35rem !important;
    padding-bottom: 0.35rem !important;
}

[data-testid="stChatInput"] {
    margin-top: 4px !important;
}

/* Remove accidental browser/Streamlit input dark fill in light mode */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] [data-baseweb="textarea"] > div {
    color-scheme: __COLOR_SCHEME__ !important;
}

/* Premium footer */
[data-testid="stAppViewContainer"] hr {
    opacity: 0.65 !important;
}


/* ============================================================
   SIDEBAR NAVIGATION — PREMIUM INTERACTION
   ============================================================ */

[data-testid="stSidebar"] .stButton > button {
    position: relative !important;
    overflow: hidden !important;
}

[data-testid="stSidebar"] .stButton > button::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 2px;
    background: linear-gradient(180deg, __ACCENT__, __ACCENT_2__);
    opacity: 0;
    transition: opacity 0.18s ease;
}

[data-testid="stSidebar"] .stButton > button:hover::before {
    opacity: 1;
}


/* ===== COMPACT SIDEBAR — KEEP SYSTEM STATUS VISIBLE ===== */
[data-testid="stSidebar"] .block-container {
    padding-top: 0.45rem !important;
    padding-bottom: 0.35rem !important;
}
[data-testid="stSidebar"] hr {
    margin: 0.35rem 0 !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] {
    margin-bottom: 0.18rem !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    min-height: 36px !important;
    height: 36px !important;
    padding: 0.15rem 0.5rem !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    margin-top: 0.25rem !important;
    margin-bottom: 0.3rem !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    margin-top: 0.12rem !important;
    margin-bottom: 0.2rem !important;
}
.wb-status-card {
    min-height: 34px !important;
    padding: 0.42rem 0.65rem !important;
    margin: 0.22rem 0 !important;
    border-radius: 10px !important;
}
.wb-sidebar-footer {
    padding: 0.1rem 0 !important;
}



/* ============================================================
   FINAL UI CLEANUP
   Fix the two remaining unfinished-looking controls:
   1) LIVE status
   2) Select Robot / Target Station
   ============================================================ */

/* ---------- LIVE: one clean pill ---------- */
[data-testid="stAlert"] {
    width: fit-content !important;
    min-width: 118px !important;
    min-height: 44px !important;
    margin-left: auto !important;
    padding: 0.65rem 1.05rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 999px !important;
    border: 1px solid rgba(16,185,129,0.28) !important;
    background: rgba(16,185,129,0.10) !important;
    box-shadow:
        0 8px 26px rgba(16,185,129,0.10),
        inset 0 1px 0 rgba(255,255,255,0.28) !important;
}

[data-testid="stAlert"] > div {
    width: auto !important;
    padding: 0 !important;
    background: transparent !important;
    border: 0 !important;
}

[data-testid="stAlert"] [data-testid="stAlertContent"] {
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
    border: 0 !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {
    background: transparent !important;
    border: 0 !important;
}

[data-testid="stAlert"] svg {
    width: 10px !important;
    height: 10px !important;
    color: #10b981 !important;
    fill: #10b981 !important;
}

[data-testid="stAlert"] p {
    color: #047857 !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
}

/* ---------- SELECTBOX: seamless single control ---------- */

/* Outer control */
[data-testid="stSelectbox"] [data-baseweb="select"] {
    width: 100% !important;
    background: transparent !important;
    border: 0 !important;
    border-radius: 14px !important;
    box-shadow: none !important;
}

/* BaseWeb's actual clickable wrapper */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    min-height: 50px !important;
    width: 100% !important;
    box-sizing: border-box !important;
    background: __SELECT_BG__ !important;
    background-color: __SELECT_BG__ !important;
    color: __SELECT_TEXT__ !important;
    border: 1px solid __SELECT_BORDER__ !important;
    border-radius: 14px !important;
    box-shadow: 0 6px 18px rgba(15,23,42,0.055) !important;
    overflow: hidden !important;
}

/* ALL inner BaseWeb layers must be transparent.
   This removes the dark arrow rectangle in Light mode. */
[data-testid="stSelectbox"] [data-baseweb="select"] > div > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div > div > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div > div > div > div {
    background: transparent !important;
    background-color: transparent !important;
    color: __SELECT_TEXT__ !important;
    border: 0 !important;
}

/* Value text */
[data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] input {
    background: transparent !important;
    color: __SELECT_TEXT__ !important;
    -webkit-text-fill-color: __SELECT_TEXT__ !important;
    font-weight: 500 !important;
}

/* Arrow area — no separate block */
[data-testid="stSelectbox"] [data-baseweb="select"] svg {
    background: transparent !important;
    color: __SELECT_TEXT__ !important;
    fill: __SELECT_TEXT__ !important;
}

/* Hover */
[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {
    border-color: __ACCENT__ !important;
    box-shadow:
        0 8px 24px rgba(8,145,178,0.09),
        0 0 0 1px rgba(8,145,178,0.07) !important;
}

/* Focus — cyan only */
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
    border: 1px solid #22d3ee !important;
    box-shadow:
        0 0 0 3px rgba(34,211,238,0.13),
        0 8px 24px rgba(34,211,238,0.08) !important;
}

/* Remove Streamlit red invalid-looking border for these normal controls */
[data-testid="stSelectbox"] [aria-invalid="true"] > div {
    border-color: __SELECT_BORDER__ !important;
}

/* ---------- LIGHT: white from edge to edge ---------- */
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div * {
    box-sizing: border-box !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"] > div[role="button"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #172033 !important;
    border-color: #cbd5e1 !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"] > div[role="button"] > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div[role="button"] > div > div {
    background: transparent !important;
    background-color: transparent !important;
}

/* Dropdown popup */
div[data-baseweb="popover"] {
    z-index: 999999 !important;
}

div[data-baseweb="popover"] > div,
div[data-baseweb="menu"] {
    background: __SELECT_BG__ !important;
    border: 1px solid __SELECT_BORDER__ !important;
    border-radius: 12px !important;
    box-shadow: 0 18px 45px rgba(15,23,42,0.14) !important;
}

div[data-baseweb="menu"] li,
div[data-baseweb="menu"] [role="option"] {
    background: __SELECT_BG__ !important;
    color: __SELECT_TEXT__ !important;
}

div[data-baseweb="menu"] li:hover,
div[data-baseweb="menu"] [role="option"]:hover {
    background: __CARD_2__ !important;
    color: __SELECT_TEXT__ !important;
}



/* ============================================================
   FINAL SELECTBOX GEOMETRY FIX
   One border only — remove BaseWeb nested/double borders.
   ============================================================ */

/* The Streamlit wrapper itself must never draw a border */
[data-testid="stSelectbox"],
[data-testid="stSelectbox"] > div,
[data-testid="stSelectbox"] > div > div {
    border: 0 !important;
    outline: 0 !important;
    box-shadow: none !important;
}

/* BaseWeb outer shell */
[data-testid="stSelectbox"] [data-baseweb="select"] {
    border: 0 !important;
    outline: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    border-radius: 14px !important;
}

/* THE ONLY visible border: actual clickable select surface */
[data-testid="stSelectbox"] [data-baseweb="select"] > div[role="button"] {
    width: 100% !important;
    min-height: 50px !important;
    box-sizing: border-box !important;
    border: 1px solid #cbd5e1 !important;
    outline: 0 !important;
    border-radius: 14px !important;
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #172033 !important;
    box-shadow: 0 5px 18px rgba(15,23,42,.055) !important;
    overflow: hidden !important;
}

/* Every child inside the clickable surface is transparent and borderless.
   This is what removes the dark right-hand rectangle. */
[data-testid="stSelectbox"] [data-baseweb="select"] > div[role="button"] > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div[role="button"] > div > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div[role="button"] > div > div > div {
    border: 0 !important;
    outline: 0 !important;
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
}

/* Value */
[data-testid="stSelectbox"] [data-baseweb="select"] > div[role="button"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] > div[role="button"] input {
    border: 0 !important;
    outline: 0 !important;
    background: transparent !important;
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
}

/* Arrow */
[data-testid="stSelectbox"] [data-baseweb="select"] > div[role="button"] svg {
    border: 0 !important;
    outline: 0 !important;
    background: transparent !important;
    color: #172033 !important;
    fill: #172033 !important;
}

/* Hover — still one border */
[data-testid="stSelectbox"] [data-baseweb="select"] > div[role="button"]:hover {
    border: 1px solid #0891b2 !important;
    box-shadow: 0 8px 22px rgba(8,145,178,.09) !important;
}

/* Focus — one cyan border + outer glow, never a second border */
[data-testid="stSelectbox"] [data-baseweb="select"] > div[role="button"]:focus,
[data-testid="stSelectbox"] [data-baseweb="select"] > div[role="button"]:focus-visible,
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div[role="button"] {
    border: 1px solid #22d3ee !important;
    outline: 0 !important;
    box-shadow: 0 0 0 3px rgba(34,211,238,.14) !important;
}

/* Streamlit error/invalid state must not create a red second border */
[data-testid="stSelectbox"] [data-baseweb="select"][aria-invalid="true"] > div[role="button"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div[aria-invalid="true"] {
    border: 1px solid #cbd5e1 !important;
    outline: 0 !important;
}

/* Open menu */
div[data-baseweb="popover"] {
    z-index: 999999 !important;
}

div[data-baseweb="popover"] > div,
div[data-baseweb="menu"] {
    background: #ffffff !important;
    border: 1px solid #d7e0ea !important;
    border-radius: 12px !important;
    box-shadow: 0 18px 45px rgba(15,23,42,.14) !important;
}

div[data-baseweb="menu"] li,
div[data-baseweb="menu"] [role="option"] {
    background: #ffffff !important;
    color: #172033 !important;
    border: 0 !important;
}

div[data-baseweb="menu"] li:hover,
div[data-baseweb="menu"] [role="option"]:hover,
div[data-baseweb="menu"] [aria-selected="true"] {
    background: #eef2f7 !important;
    color: #172033 !important;
}

/* Dark mode — same single-border geometry */
@media (prefers-color-scheme: dark) {
    [data-testid="stSelectbox"] [data-baseweb="select"] > div[role="button"] {
        border: 1px solid #334155 !important;
        background: #111827 !important;
        background-color: #111827 !important;
        color: #f8fafc !important;
        box-shadow: 0 6px 20px rgba(0,0,0,.18) !important;
    }

    [data-testid="stSelectbox"] [data-baseweb="select"] > div[role="button"] span,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div[role="button"] input {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
    }

    [data-testid="stSelectbox"] [data-baseweb="select"] > div[role="button"] svg {
        color: #e2e8f0 !important;
        fill: #e2e8f0 !important;
    }

    [data-testid="stSelectbox"] [data-baseweb="select"] > div[role="button"]:hover {
        border-color: #22d3ee !important;
    }

    [data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div[role="button"] {
        border: 1px solid #22d3ee !important;
        box-shadow: 0 0 0 3px rgba(34,211,238,.13) !important;
    }

    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"],
    div[data-baseweb="menu"] li,
    div[data-baseweb="menu"] [role="option"] {
        background: #111827 !important;
        color: #f8fafc !important;
    }

    div[data-baseweb="menu"] li:hover,
    div[data-baseweb="menu"] [role="option"]:hover,
    div[data-baseweb="menu"] [aria-selected="true"] {
        background: #1e293b !important;
        color: #ffffff !important;
    }
}

</style>
"""

CSS = (
    CSS
    .replace("__COLOR_SCHEME__", "dark" if is_dark else "light")
    .replace("__BG__", BG)
    .replace("__SIDEBAR__", SIDEBAR)
    .replace("__CARD__", CARD)
    .replace("__CARD_2__", CARD_2)
    .replace("__TEXT__", TEXT)
    .replace("__TEXT_MUTED__", TEXT_MUTED)
    .replace("__BORDER__", BORDER)
    .replace("__BORDER_HOVER__", BORDER_HOVER)
    .replace("__ACCENT__", ACCENT)
    .replace("__ACCENT_2__", ACCENT_2)
    .replace("__SELECT_BG__", "#111827" if is_dark else "#ffffff")
    .replace("__SELECT_TEXT__", "#f8fafc" if is_dark else "#172033")
    .replace("__SELECT_BORDER__", "#334155" if is_dark else "#cbd5e1")
    .replace("__OPTION_HOVER__", "#1e293b" if is_dark else "#f1f5f9")
    .replace("__OPTION_SELECTED__", "#263449" if is_dark else "#e2e8f0")
    .replace("__CHAT_BG__", CARD_2)
)

st.markdown(CSS, unsafe_allow_html=True)

# LOAD ML DATA
# ============================================================

@st.cache_data
def load_data():

    telemetry = generate_telemetry(
        num_robots=20,
        records_per_robot=100
    )

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

    return (
        telemetry,
        threshold,
        accuracy
    )


# ============================================================
# REFRESH FLEET
# ============================================================

if st.sidebar.button(
    "↻  Refresh Fleet",
    width="stretch"
):

    st.cache_data.clear()
    st.rerun()


# ============================================================
# DATA
# ============================================================

telemetry, anomaly_threshold, model_accuracy = load_data()


latest = (
    telemetry
    .sort_values("timestamp")
    .groupby("robot_id")
    .tail(1)
    .copy()
)


# ============================================================
# KPI DATA
# ============================================================

total_robots = len(latest)

healthy = len(
    latest[
        latest["status"] == "Healthy"
    ]
)

warning = len(
    latest[
        latest["status"] == "Warning"
    ]
)

critical = len(
    latest[
        latest["status"] == "Critical"
    ]
)

anomalies = int(
    telemetry["anomaly"].sum()
)

low_stock = get_low_stock_items()


# ============================================================
# NAVIGATION STATE
# ============================================================

if "active_page" not in st.session_state:
    st.session_state.active_page = "All Details"

active_page = st.session_state.active_page

def nav_button(label, page):
    if st.sidebar.button(label, width="stretch"):
        st.session_state.active_page = page
        st.rerun()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🤖 WareBot AI"
)

st.sidebar.caption(
    "WAREHOUSE INTELLIGENCE PLATFORM"
)

st.sidebar.divider()

st.sidebar.subheader(
    "CONTROL CENTER"
)

nav_button(
    "⌂  All Details",
    "All Details"
)

st.sidebar.caption("Shows the complete WareBot command center")

nav_button(
    "◉  Fleet Monitoring",
    "Fleet Monitoring"
)

nav_button(
    "△  Predictive Maintenance",
    "Predictive Maintenance"
)

nav_button(
    "◇  Route Optimization",
    "Route Optimization"
)

nav_button(
    "□  Inventory & Orders",
    "Inventory & Orders"
)

st.sidebar.caption(f"VIEW  •  {active_page}")

st.sidebar.divider()

st.sidebar.subheader(
    "SYSTEM STATUS"
)

status_items = [
    ("●", "Fleet Monitoring", "Online"),
    ("●", "ML Engine", "Online"),
    ("●", "WMS", "Connected"),
    ("●", "Route Optimizer", "Active"),
]

for dot, name, state in status_items:
    st.sidebar.markdown(
        f"""<div class=\"wb-status-card\">
            <span class=\"wb-status-dot\">{dot}</span>
            <span class=\"wb-status-name\">{name}</span>
            <span class=\"wb-status-dash\">—</span>
            <span class=\"wb-status-state\">{state}</span>
        </div>""",
        unsafe_allow_html=True,
    )

st.sidebar.divider()

st.sidebar.markdown(
    """<div class=\"wb-sidebar-footer\">
        <div class=\"wb-version\">WareBot AI v1.0</div>
        <div class=\"wb-footer-sub\">Autonomous Warehouse Intelligence</div>
    </div>""",
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns(
    [5, 1]
)

with header_left:

    st.title(
        "WAREBOT AI"
    )

    st.caption(
        "Autonomous warehouse intelligence & fleet operations"
    )


with header_right:

    st.success(
        "● LIVE"
    )


if active_page == "All Details":
    # ============================================================
    # KPI ROW
    # ============================================================

    k1, k2, k3, k4, k5 = st.columns(5)


    with k1:

        st.metric(
            "TOTAL FLEET",
            total_robots,
            "Monitored"
        )


    with k2:

        st.metric(
            "HEALTHY ROBOTS",
            healthy,
            "Operational"
        )


    with k3:

        st.metric(
            "ATTENTION",
            warning + critical,
            "Requires review"
        )


    with k4:

        st.metric(
            "ANOMALIES",
            anomalies,
            "AI detected"
        )


    with k5:

        st.metric(
            "LOW STOCK",
            len(low_stock),
            "Reorder needed"
        )



if active_page in ("All Details", "Fleet Monitoring"):
    # ============================================================
    # FLEET INTELLIGENCE
    # ============================================================

    st.subheader(
        "Fleet Intelligence"
    )

    st.divider()


    left, right = st.columns(2)


    # ============================================================
    # ROBOT HEALTH
    # ============================================================

    with left:

        with st.container(border=True):

            st.markdown(
                "#### Robot Health Distribution"
            )

            health_df = pd.DataFrame(
                {
                    "Status": [
                        "Healthy",
                        "Warning",
                        "Critical"
                    ],
                    "Robots": [
                        healthy,
                        warning,
                        critical
                    ]
                }
            )

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=health_df["Status"],
                    y=health_df["Robots"],
                    text=health_df["Robots"],
                    textposition="outside",

                    marker_color=[
                        "#22c55e",
                        "#f59e0b",
                        "#ef4444"
                    ],

                    marker_line_width=0
                )
            )

            fig.update_layout(

                template=(
                    "plotly_dark"
                    if is_dark
                    else "plotly_white"
                ),

                paper_bgcolor="rgba(0,0,0,0)",

                plot_bgcolor="rgba(0,0,0,0)",

                height=360,

                margin=dict(
                    l=20,
                    r=20,
                    t=25,
                    b=30
                ),

                showlegend=False,

                font=dict(
                    color=TEXT
                ),

                xaxis=dict(
                    title_font=dict(
                        color=TEXT
                    ),

                    tickfont=dict(
                        color=TEXT_MUTED
                    ),

                    gridcolor=GRID,

                    zerolinecolor=GRID
                ),

                yaxis=dict(
                    title_font=dict(
                        color=TEXT
                    ),

                    tickfont=dict(
                        color=TEXT_MUTED
                    ),

                    gridcolor=GRID,

                    zerolinecolor=GRID
                )
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )


    # ============================================================
    # BATTERY PERFORMANCE
    # ============================================================

    with right:

        with st.container(border=True):

            selected_robot = st.selectbox(
                "Select Robot",
                sorted(
                    telemetry[
                        "robot_id"
                    ].unique()
                )
            )

            robot_history = telemetry[
                telemetry["robot_id"]
                == selected_robot
            ]

            st.markdown(
                f"#### {selected_robot} · Battery Performance"
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=robot_history["timestamp"],
                    y=robot_history["battery"],

                    mode="lines",

                    line=dict(
                        color=ACCENT,
                        width=2.5
                    ),

                    fill="tozeroy",

                    fillcolor=(
                        "rgba(34,211,238,0.05)"
                        if is_dark
                        else "rgba(8,145,178,0.06)"
                    )
                )
            )

            fig.update_layout(

                template=(
                    "plotly_dark"
                    if is_dark
                    else "plotly_white"
                ),

                paper_bgcolor="rgba(0,0,0,0)",

                plot_bgcolor="rgba(0,0,0,0)",

                height=360,

                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=30
                ),

                showlegend=False,

                font=dict(
                    color=TEXT
                ),

                xaxis=dict(
                    title="Time",

                    title_font=dict(
                        color=TEXT
                    ),

                    tickfont=dict(
                        color=TEXT_MUTED
                    ),

                    gridcolor=GRID,

                    zerolinecolor=GRID
                ),

                yaxis=dict(
                    title="Battery %",

                    title_font=dict(
                        color=TEXT
                    ),

                    tickfont=dict(
                        color=TEXT_MUTED
                    ),

                    gridcolor=GRID,

                    zerolinecolor=GRID
                )
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )



if active_page in ("All Details", "Predictive Maintenance"):
    # ============================================================
    # PREDICTIVE MAINTENANCE
    # ============================================================

    st.subheader(
        "Predictive Maintenance"
    )

    st.divider()


    maintenance = latest[
        (
            latest["status"] != "Healthy"
        )
        |
        (
            latest["maintenance_risk"] == 1
        )
    ].sort_values(
        "health_score"
    )


    if len(maintenance) > 0:

        st.warning(
            f"⚠️ {len(maintenance)} robot(s) "
            "require maintenance attention."
        )

        st.dataframe(
            maintenance[
                [
                    "robot_id",
                    "battery",
                    "motor_current",
                    "temperature",
                    "vibration",
                    "navigation_errors",
                    "health_score",
                    "status"
                ]
            ],

            width="stretch",

            hide_index=True
        )

    else:

        st.success(
            "✓ Fleet is operating within healthy parameters."
        )



if active_page in ("All Details", "Route Optimization"):
    # ============================================================
    # WAREHOUSE DIGITAL TWIN
    # ============================================================

    st.subheader(
        "Warehouse Digital Twin"
    )

    st.divider()


    map_col, task_col = st.columns(
        [2.2, 1]
    )


    # ============================================================
    # WAREHOUSE MAP
    # ============================================================

    with map_col:

        with st.container(border=True):

            st.markdown(
                "#### Live Warehouse Map"
            )

            free_x = []
            free_y = []

            obstacle_x = []
            obstacle_y = []

            for row in range(
                len(WAREHOUSE_GRID)
            ):

                for col in range(
                    len(WAREHOUSE_GRID[0])
                ):

                    if WAREHOUSE_GRID[row][col] == 1:

                        obstacle_x.append(col)
                        obstacle_y.append(row)

                    else:

                        free_x.append(col)
                        free_y.append(row)


            fig = go.Figure()


            fig.add_trace(
                go.Scatter(
                    x=free_x,
                    y=free_y,

                    mode="markers",

                    marker=dict(
                        size=12,

                        color=(
                            "#172554"
                            if is_dark
                            else "#dbeafe"
                        ),

                        symbol="square"
                    ),

                    name="Open"
                )
            )


            fig.add_trace(
                go.Scatter(
                    x=obstacle_x,
                    y=obstacle_y,

                    mode="markers",

                    marker=dict(
                        size=17,

                        color=(
                            "#334155"
                            if is_dark
                            else "#94a3b8"
                        ),

                        symbol="square"
                    ),

                    name="Obstacle"
                )
            )


            station_colors = {
                "P1": "#22d3ee",
                "P2": "#6366f1",
                "P3": "#8b5cf6",
                "PACK": "#f59e0b"
            }


            for station, position in STATIONS.items():

                row, col = position

                fig.add_trace(
                    go.Scatter(
                        x=[col],
                        y=[row],

                        mode="markers+text",

                        text=[station],

                        textposition="top center",

                        marker=dict(
                            size=22,

                            color=station_colors.get(
                                station,
                                ACCENT
                            ),

                            symbol="diamond",

                            line=dict(
                                color="#ffffff",
                                width=1
                            )
                        ),

                        name=station
                    )
                )


            fig.update_layout(

                template=(
                    "plotly_dark"
                    if is_dark
                    else "plotly_white"
                ),

                paper_bgcolor="rgba(0,0,0,0)",

                plot_bgcolor=PLOT_BG,

                height=500,

                margin=dict(
                    l=15,
                    r=15,
                    t=20,
                    b=40
                ),

                font=dict(
                    color=TEXT
                ),

                xaxis=dict(
                    title_font=dict(
                        color=TEXT
                    ),

                    tickfont=dict(
                        color=TEXT_MUTED
                    ),

                    dtick=1,

                    gridcolor=GRID,

                    zerolinecolor=GRID
                ),

                yaxis=dict(
                    title_font=dict(
                        color=TEXT
                    ),

                    tickfont=dict(
                        color=TEXT_MUTED
                    ),

                    dtick=1,

                    autorange="reversed",

                    gridcolor=GRID,

                    zerolinecolor=GRID
                ),

                legend=dict(
                    orientation="h",

                    y=-0.08,

                    font=dict(
                        color=TEXT_MUTED
                    )
                )
            )


            st.plotly_chart(
                fig,
                width="stretch"
            )


    # ============================================================
    # AI TASK ALLOCATION
    # ============================================================

    with task_col:

        with st.container(border=True):

            st.markdown(
                "#### AI Task Allocation"
            )


            robot_options = [

                {
                    "robot_id": "R01",
                    "position": (0, 0),
                    "battery": 85,
                    "maintenance_risk": 0,
                },

                {
                    "robot_id": "R02",
                    "position": (4, 2),
                    "battery": 72,
                    "maintenance_risk": 0,
                },

                {
                    "robot_id": "R03",
                    "position": (8, 4),
                    "battery": 55,
                    "maintenance_risk": 1,
                },

                {
                    "robot_id": "R04",
                    "position": (9, 0),
                    "battery": 91,
                    "maintenance_risk": 0,
                }

            ]


            target = st.selectbox(
                "Target Station",
                list(STATIONS.keys())
            )


            selected = allocate_task(
                robot_options,
                target
            )


            if selected:

                st.metric(
                    "Recommended Robot",
                    selected["robot_id"]
                )

                st.metric(
                    "Route Distance",
                    f"{selected['distance']} steps"
                )

                st.metric(
                    "Optimization Score",
                    selected["score"]
                )

                st.markdown(
                    "##### Optimized Path"
                )

                st.code(
                    str(selected["path"]),
                    language="text"
                )



if active_page in ("All Details", "Route Optimization"):
    # ============================================================
    # CONGESTION INTELLIGENCE
    # ============================================================

    st.subheader(
        "Congestion Intelligence"
    )

    st.divider()


    congestion = pd.DataFrame(
        {
            "Station": [
                "P1",
                "P2",
                "P3",
                "PACK"
            ],

            "Robot Traffic": [
                7,
                3,
                9,
                5
            ],

            "Queue": [
                12,
                4,
                17,
                8
            ],

            "Utilization": [
                72,
                38,
                91,
                64
            ]
        }
    )


    c1, c2 = st.columns(2)


    # ============================================================
    # ROBOT TRAFFIC
    # ============================================================

    with c1:

        with st.container(border=True):

            st.markdown(
                "#### Robot Traffic"
            )

            fig = px.bar(
                congestion,
                x="Station",
                y="Robot Traffic",
                text="Robot Traffic"
            )

            fig.update_traces(
                marker_color=ACCENT_2
            )

            fig.update_layout(

                template=(
                    "plotly_dark"
                    if is_dark
                    else "plotly_white"
                ),

                paper_bgcolor="rgba(0,0,0,0)",

                plot_bgcolor="rgba(0,0,0,0)",

                height=340,

                font=dict(
                    color=TEXT
                ),

                showlegend=False,

                xaxis=dict(
                    title="Station",

                    title_font=dict(
                        color=TEXT
                    ),

                    tickfont=dict(
                        color=TEXT_MUTED
                    ),

                    gridcolor=GRID,

                    zerolinecolor=GRID
                ),

                yaxis=dict(
                    title="Robot Traffic",

                    title_font=dict(
                        color=TEXT
                    ),

                    tickfont=dict(
                        color=TEXT_MUTED
                    ),

                    gridcolor=GRID,

                    zerolinecolor=GRID
                )
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )


    # ============================================================
    # STATION UTILIZATION
    # ============================================================

    with c2:

        with st.container(border=True):

            st.markdown(
                "#### Station Utilization"
            )

            fig = px.bar(
                congestion,
                x="Station",
                y="Utilization",
                text="Utilization"
            )

            fig.update_traces(
                marker_color=ACCENT
            )

            fig.update_layout(

                template=(
                    "plotly_dark"
                    if is_dark
                    else "plotly_white"
                ),

                paper_bgcolor="rgba(0,0,0,0)",

                plot_bgcolor="rgba(0,0,0,0)",

                height=340,

                font=dict(
                    color=TEXT
                ),

                showlegend=False,

                xaxis=dict(
                    title="Station",

                    title_font=dict(
                        color=TEXT
                    ),

                    tickfont=dict(
                        color=TEXT_MUTED
                    ),

                    gridcolor=GRID,

                    zerolinecolor=GRID
                ),

                yaxis=dict(
                    title="Utilization",

                    title_font=dict(
                        color=TEXT
                    ),

                    tickfont=dict(
                        color=TEXT_MUTED
                    ),

                    gridcolor=GRID,

                    zerolinecolor=GRID,

                    range=[0, 100]
                )
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )



if active_page in ("All Details", "Inventory & Orders"):
    # ============================================================
    # WAREHOUSE MANAGEMENT
    # ============================================================

    st.subheader(
        "Warehouse Management"
    )

    st.divider()


    inventory = get_inventory()

    orders = get_orders()


    w1, w2 = st.columns(2)


    # ============================================================
    # INVENTORY
    # ============================================================

    with w1:

        with st.container(border=True):

            st.markdown(
                "#### Inventory"
            )

            st.dataframe(
                inventory,
                width="stretch",
                hide_index=True
            )


    # ============================================================
    # ACTIVE ORDERS
    # ============================================================

    with w2:

        with st.container(border=True):

            st.markdown(
                "#### Active Orders"
            )

            st.dataframe(
                orders,
                width="stretch",
                hide_index=True
            )



if active_page == "All Details":
    # ============================================================
    # OPS ASSISTANT
    # ============================================================

    st.subheader(
        "Ops Assistant"
    )

    st.divider()


    with st.container(border=True):

        st.markdown(
            "#### 🤖 WareBot Operations Assistant"
        )

        st.caption(
            "Ask about robot health, maintenance, "
            "anomalies, inventory or fleet status."
        )


        question = st.chat_input(
            "Ask WareBot..."
        )


        if question:

            q = question.lower()


            if (
                "maintenance" in q
                or "repair" in q
                or "problem" in q
            ):

                critical_robots = latest[
                    latest["status"] == "Critical"
                ]


                if len(critical_robots) > 0:

                    names = ", ".join(
                        critical_robots[
                            "robot_id"
                        ].tolist()
                    )

                    answer = (
                        f"{len(critical_robots)} critical "
                        f"robot(s) need attention: {names}. "
                        "Check battery, vibration, temperature "
                        "and motor current."
                    )

                else:

                    answer = (
                        "No robots are currently classified "
                        "as Critical."
                    )


            elif (
                "inventory" in q
                or "stock" in q
            ):

                if len(low_stock) > 0:

                    names = ", ".join(
                        low_stock[
                            "product_name"
                        ].tolist()
                    )

                    answer = (
                        f"{len(low_stock)} item(s) are below "
                        f"reorder level: {names}."
                    )

                else:

                    answer = (
                        "Inventory levels are currently healthy."
                    )


            elif "anomal" in q:

                answer = (
                    f"WareBot detected {anomalies} "
                    "telemetry anomalies."
                )


            elif (
                "robot" in q
                or "fleet" in q
            ):

                answer = (
                    f"Fleet status: {total_robots} robots, "
                    f"{healthy} healthy, "
                    f"{warning} warning, "
                    f"{critical} critical."
                )


            else:

                answer = (
                    "I can help with robot health, "
                    "maintenance, anomalies, inventory "
                    "and fleet status."
                )


            with st.chat_message(
                "assistant",
                avatar="🤖"
            ):

                st.write(
                    answer
                )



# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "WAREBOT AI  •  AUTONOMOUS WAREHOUSE INTELLIGENCE  •  v1.0"
)