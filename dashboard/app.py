# coding: utf-8
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import joblib

from src.utils import (
    MODELS_DIR, DATA_PROCESSED, DATA_EXTERNAL,
    LOM_THRESHOLD, UPM_THRESHOLD, HIGH_THRESHOLD,
    DEFAULT_APBN_PCT, DEFAULT_KUR_PCT, DEFAULT_TRANSFER_PCT,
)
from src.predict import load_model, predict_all_provinces, what_if_scenario, KOEF_APBN, KOEF_KUR, KOEF_TKD
from dashboard.ai_analysis import analyze_scenario, get_api_key

# -- Page config --
st.set_page_config(
    page_title="Regional Economic Trap Early Warning System",
    layout="wide",
    page_icon="",
)

# -- CSS Premium --
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    * { font-family: 'Inter', sans-serif !important; }

    /* -- Animated Background -- */
    .stApp {
        background: #070b14;
        position: relative;
        overflow-x: hidden;
    }
    .stApp::before {
        content: '';
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background:
            radial-gradient(ellipse at 20% 50%, rgba(15, 45, 90, 0.4) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, rgba(0, 150, 200, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 80%, rgba(100, 50, 150, 0.06) 0%, transparent 50%);
        z-index: 0;
        animation: ambientShift 20s ease-in-out infinite alternate;
    }
    @keyframes ambientShift {
        0% { transform: translate(0, 0) rotate(0deg); }
        100% { transform: translate(-2%, -1%) rotate(3deg); }
    }

    /* -- Grid Overlay -- */
    .stApp::after {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image:
            linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px);
        background-size: 60px 60px;
        z-index: 0;
        pointer-events: none;
    }

    /* -- Ensure content above backgrounds -- */
    .stApp > div:first-child { position: relative; z-index: 1; }
    .block-container { position: relative; z-index: 1; }

    /* -- Glass Header -- */
    .header-wrap {
        background: linear-gradient(135deg, rgba(10,20,40,0.92), rgba(20,40,70,0.88));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px;
        padding: 1.5rem 2rem 1.2rem;
        margin: -0.5rem 0 0.5rem 0;
        box-shadow: 0 8px 40px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
        position: relative;
        overflow: hidden;
    }
    .header-wrap::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00d4ff, #f0b429, #00d4ff, transparent);
        background-size: 200% 100%;
        animation: shimmer 4s linear infinite;
    }
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    .header-title {
        color: #fff;
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 14px;
        margin: 0;
        line-height: 1.3;
    }
    .header-title span:first-child {
        background: linear-gradient(135deg, #ffffff 40%, #80d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .header-badge {
        background: linear-gradient(135deg, #00d4ff, #0099cc);
        color: #fff;
        font-size: 0.6rem;
        font-weight: 700;
        padding: 3px 14px;
        border-radius: 20px;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        -webkit-text-fill-color: #fff;
        box-shadow: 0 2px 12px rgba(0,212,255,0.3);
        white-space: nowrap;
    }
    .header-sub {
        color: rgba(255,255,255,0.45);
        font-size: 0.8rem;
        margin-top: 4px;
        font-weight: 300;
        letter-spacing: 0.3px;
    }

    /* -- Nav Radio -- */
    div.row-widget.stRadio {
        max-width: 100% !important;
        flex-direction: row !important;
    }
    .stRadio > div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        justify-content: center;
        gap: 0.6rem;
        flex-wrap: wrap;
        padding: 0.3rem 0 0.8rem 0;
        max-width: 100% !important;
    }
    .stRadio label {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 14px !important;
        padding: 10px 28px !important;
        color: #ffffff !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        transition: all 0.4s cubic-bezier(0.25,0.46,0.45,0.94) !important;
        backdrop-filter: blur(8px) !important;
        user-select: none !important;
    }
    .stRadio label:hover {
        background: rgba(0,212,255,0.08) !important;
        border-color: rgba(0,212,255,0.25) !important;
        color: #ffffff !important;
    }
    .stRadio label[data-selected="true"] {
        background: linear-gradient(135deg, #00d4ff, #0099cc) !important;
        border-color: #00d4ff !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 20px rgba(0,212,255,0.25) !important;
    }

    /* -- Glass Card -- */
    .glass-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .glass-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 40px rgba(0,0,0,0.35);
        border-color: rgba(255,255,255,0.12);
    }

    /* -- Metric Blocks -- */
    .metric-block {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 1.2rem 1rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-block:hover {
        border-color: rgba(0,212,255,0.2);
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }
    .metric-block .icon {
        font-size: 1.4rem;
        margin-bottom: 4px;
    }
    .metric-block .label {
        color: rgba(255,255,255,0.45);
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-block .value {
        color: #fff;
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        line-height: 1.3;
    }
    .metric-block .sub {
        color: rgba(255,255,255,0.35);
        font-size: 0.7rem;
        margin-top: 2px;
    }

    /* -- Section Headers -- */
    .section-title {
        color: #f0b429;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 10px;
        letter-spacing: 0.3px;
    }
    .section-title .line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(240,180,41,0.3), transparent);
    }
    .section-icon { font-size: 1.2rem; }

    /* -- Slider -- */
    .stSlider label, .stSelectbox label, .stNumberInput label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        color: #fff !important;
        transition: border-color 0.3s ease;
    }
    .stSelectbox > div > div:hover {
        border-color: #00d4ff !important;
    }

    /* -- Dataframe -- */
    .stDataFrame {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        overflow: hidden;
    }
    .stDataFrame thead tr th {
        background: rgba(0,212,255,0.08) !important;
        color: rgba(255,255,255,0.8) !important;
        font-weight: 600 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        border-bottom: 1px solid rgba(255,255,255,0.06) !important;
    }
    .stDataFrame tbody tr td {
        color: rgba(255,255,255,0.7) !important;
        border-bottom: 1px solid rgba(255,255,255,0.03) !important;
        font-size: 0.8rem !important;
    }
    .stDataFrame tbody tr:hover td {
        background: rgba(0,212,255,0.04) !important;
    }

    /* -- Info / Success -- */
    .stAlert {
        background: rgba(0,212,255,0.08) !important;
        border: 1px solid rgba(0,212,255,0.2) !important;
        color: #80d4ff !important;
        border-radius: 14px !important;
        backdrop-filter: blur(8px);
    }
    .stSuccess {
        background: rgba(46,204,113,0.1) !important;
        border: 1px solid rgba(46,204,113,0.2) !important;
        color: #2ecc71 !important;
        border-radius: 14px !important;
        backdrop-filter: blur(8px);
    }

    /* -- Button -- */
    .stButton button {
        background: linear-gradient(135deg, #00d4ff, #0077b6) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 32px !important;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
        box-shadow: 0 4px 20px rgba(0,212,255,0.2) !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
    }
    .stButton button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 35px rgba(0,212,255,0.4) !important;
        background: linear-gradient(135deg, #33ddff, #0099cc) !important;
    }
    .stButton button:active {
        transform: translateY(0px) !important;
    }

    /* -- Footer -- */
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.2);
        font-size: 0.65rem;
        padding: 2rem 0 0.5rem 0;
        border-top: 1px solid rgba(255,255,255,0.04);
        margin-top: 2.5rem;
        letter-spacing: 0.5px;
    }
    .footer strong { color: rgba(255,255,255,0.35); }

    /* -- Caption -- */
    .caption-text {
        color: rgba(255,255,255,0.3);
        font-size: 0.7rem;
        font-style: italic;
        text-align: center;
        margin-top: -8px;
    }

    /* -- Slider Premium -- */
    .stSlider [data-baseweb="slider"] {
        padding-top: 2px;
        padding-bottom: 0;
    }
    .stSlider [data-baseweb="slider"] > div {
        height: 4px !important;
    }
    .stSlider [data-baseweb="slider"] div {
        background: rgba(255,255,255,0.08) !important;
        height: 4px !important;
        border-radius: 4px !important;
    }
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: linear-gradient(135deg, #f0b429, #f59e0b) !important;
        border: 2px solid rgba(0,0,0,0.3) !important;
        width: 18px !important;
        height: 18px !important;
        margin-top: -9px !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 8px rgba(240,180,41,0.4), 0 0 0 4px rgba(240,180,41,0.1) !important;
        transition: all 0.2s ease !important;
    }
    .stSlider [data-baseweb="slider"] div[role="slider"]:hover {
        transform: scale(1.15) !important;
        box-shadow: 0 3px 12px rgba(240,180,41,0.6), 0 0 0 6px rgba(240,180,41,0.15) !important;
    }
    .stSlider [data-baseweb="slider"] div[role="slider"]:focus {
        box-shadow: 0 0 0 3px rgba(240,180,41,0.3) !important;
    }
    .stSlider [data-testid="stSliderValue"] {
        color: #f0b429 !important;
        font-weight: 700 !important;
        font-size: 0.75rem !important;
        background: rgba(240,180,41,0.1) !important;
        padding: 1px 8px !important;
        border-radius: 8px !important;
        border: 1px solid rgba(240,180,41,0.15) !important;
        min-width: 45px !important;
        text-align: center !important;
    }
    .stSlider [data-baseweb="slider"] .st-c5 {
        background: linear-gradient(90deg, #f0b429, rgba(240,180,41,0.3)) !important;
        height: 4px !important;
        border-radius: 4px !important;
    }
    /* Slider label di dalam param-card */
    .param-card .stSlider {
        padding-top: 0 !important;
    }
    .param-card .stSlider label {
        display: none !important;
    }

    /* Responsive tweaks */
    @media (max-width: 768px) {
        .header-title { font-size: 1.1rem; flex-wrap: wrap; }
        .header-badge { font-size: 0.5rem; }
        .nav-pill { padding: 6px 16px; font-size: 0.75rem; }
    }

    /* Plotly transparent bg */
    .js-plotly-plot .plotly .main-svg,
    .js-plotly-plot .plotly .svg-container {
        background: transparent !important;
    }
    .plotly .cursor-crosshair { cursor: default; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

    /* -- Parameter Card (Slider Container) -- */
    .param-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 0.4rem 0.7rem 0.2rem;
        margin-bottom: 0.3rem;
        transition: all 0.35s cubic-bezier(0.25,0.46,0.45,0.94);
        position: relative;
        overflow: hidden;
    }
    .param-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent, #00d4ff), transparent);
        opacity: 0.6;
    }
    .param-card:hover {
        border-color: rgba(255,255,255,0.15);
        transform: translateY(-2px);
        box-shadow: 0 6px 30px rgba(0,0,0,0.3);
    }
    .param-card .param-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 2px;
    }
    .param-card .param-label {
        color: rgba(255,255,255,0.55);
        font-size: 0.6rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .param-card .param-icon {
        font-size: 0.85rem;
        opacity: 0.5;
    }
    .param-card .param-value {
        color: #f0b429;
        font-size: 0.85rem;
        font-weight: 700;
        text-align: right;
        margin-top: -6px;
    }

    /* -- Result Card -- */
    .result-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 0.5rem 0.6rem;
        margin-bottom: 0;
        transition: all 0.35s ease;
    }
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .result-card .rc-label {
        color: rgba(255,255,255,0.4);
        font-size: 0.5rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 2px;
    }
    .result-card .rc-big {
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    .result-card .rc-sub {
        color: rgba(255,255,255,0.35);
        font-size: 0.6rem;
        margin-top: 1px;
    }
    .result-card .rc-desc {
        color: rgba(255,255,255,0.65);
        font-size: 0.65rem;
        line-height: 1.4;
        margin-top: 4px;
        padding-top: 4px;
        border-top: 1px solid rgba(255,255,255,0.05);
    }

    /* -- Classification Badge -- */
    .class-badge {
        display: inline-block;
        padding: 2px 12px;
        border-radius: 20px;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .class-badge.lom { background: rgba(231,76,60,0.2); color: #e74c3c; border: 1px solid rgba(231,76,60,0.3); }
    .class-badge.upm { background: rgba(243,156,18,0.2); color: #f39c12; border: 1px solid rgba(243,156,18,0.3); }
    .class-badge.high { background: rgba(46,204,113,0.2); color: #2ecc71; border: 1px solid rgba(46,204,113,0.3); }

    /* -- Divider -- */
    .ws-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
        margin: 0.3rem 0;
    }

    /* -- Glass narrative restyle -- */
    .glass-narrative {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.8rem;
        font-size: 0.78rem;
        line-height: 1.6;
        color: rgba(255,255,255,0.75);
    }
    .glass-narrative .gn-title {
        font-weight: 700;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 6px;
    }
    .glass-narrative.gn-upm { border-left: 3px solid #2ecc71; }
    .glass-narrative.gn-upm .gn-title { color: #2ecc71; }
    .glass-narrative.gn-lom { border-left: 3px solid #e74c3c; }
    .glass-narrative.gn-lom .gn-title { color: #e74c3c; }
    .glass-narrative.gn-stable { border-left: 3px solid #f0b429; }
    .glass-narrative.gn-stable .gn-title { color: #f0b429; }
    .glass-narrative.gn-info { border-left: 3px solid #00d4ff; }
    .glass-narrative.gn-info .gn-title { color: #00d4ff; }
    .glass-narrative.gn-ai { border-left: 3px solid #a855f7; }
    .glass-narrative.gn-ai .gn-title { color: #a855f7; }

    /* -- Info strip -- */
    .info-strip {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        padding: 0.2rem 0.6rem;
        margin-bottom: 0.2rem;
    }
    .info-strip .is-label {
        color: rgba(255,255,255,0.4);
        font-size: 0.6rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .info-strip .is-value {
        color: rgba(255,255,255,0.85);
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -- Header --
st.markdown(''
    '<div class="header-wrap">'
    '<div class="header-title">'
    '<span>Regional Economic Trap<br>Early Warning System</span>'
    '<span class="header-badge">Data Analytics</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True)

# -- Navigasi --
menu = st.radio("", ["Overview", "Prediksi per Provinsi", "What-If Simulator"],
                horizontal=True, label_visibility="collapsed")


# -- Cached resources --
@st.cache_resource
def init():
    df = pd.read_parquet(DATA_PROCESSED / "data_processed.parquet")
    model, encoders = load_model()
    return df, model, encoders

@st.cache_resource
def load_geojson():
    with open(DATA_EXTERNAL / "indonesia_provinces.geojson", "r") as f:
        return json.load(f)

@st.cache_resource
def load_prov_mapping():
    with open(DATA_EXTERNAL / "provinsi_mapping.json", "r") as f:
        return json.load(f)


df, model, encoders = init()
geojson = load_geojson()
prov_mapping = load_prov_mapping()

# -- Konstanta --
WARNA = {"LOM": "#e74c3c", "UPM": "#f39c12", "HIGH": "#2ecc71", "LOW": "#3498db"}
CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="rgba(255,255,255,0.75)", family="Inter, sans-serif", size=11),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="rgba(15,25,50,0.95)", font=dict(color="#fff", family="Inter", size=12),
        bordercolor="rgba(0,212,255,0.2)"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False, showline=True,
               linecolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False, showline=True,
               linecolor="rgba(255,255,255,0.06)"),
    margin=dict(l=10, r=10, t=40, b=40),
)


# ══════════════════════════════════════════════
#  HELPER
# ══════════════════════════════════════════════
def section(title, icon=""):
    st.markdown(
        f'<div class="section-title"><span class="section-icon">{icon}</span>{title}<span class="line"></span></div>',
        unsafe_allow_html=True)


def clear_ws_results():
    keys = ["ws_result", "ws_submitted", "ws_ai_result", "ws_map_fig", "ws_map_key"]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]


# ══════════════════════════════════════════════
#  MENU 1 -- OVERVIEW
# ══════════════════════════════════════════════
if menu == "Overview":
    tahun = st.slider("Tahun Analisis", 2016, 2025, 2025)
    hasil = predict_all_provinces(model, encoders, df, tahun)
    hasil = hasil.rename(columns={"PREDIKSI_G_PDRB_IDR": "Pertumbuhan PDRB Dalam Rupiah"})

    # -- Row 1: Peta + Line Chart --
    section("Peta Klasifikasi & Indikator Ekonomi", "")
    col_map = st.columns([2, 3])

    with col_map[0]:
        prov_terpilih = st.selectbox("Pilih Provinsi", sorted(df["PROVINSI"].unique()), index=10)
        df_prov = df[df["PROVINSI"] == prov_terpilih].sort_values("TAHUN")

        fig_tren = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
            subplot_titles=("Trend PDRB", "Pertumbuhan Ekonomi", "Pertumbuhan Penduduk"),
        )
        fig_tren.add_trace(
            go.Scatter(x=df_prov["TAHUN"], y=df_prov["PDRB_IDR_MLY"],
                       mode="lines+markers", name="PDRB (miliar Rp)",
                       line=dict(color="#00d4ff", width=2.5),
                       marker=dict(color="#00d4ff", size=5, symbol="circle")),
            row=1, col=1)
        fig_tren.add_trace(
            go.Scatter(x=df_prov["TAHUN"], y=df_prov["G_PDRB_IDR"] * 100,
                       mode="lines+markers", name="Pertumbuhan Ekonomi (%)",
                       line=dict(color="#2ecc71", width=2.5),
                       marker=dict(color="#2ecc71", size=5, symbol="circle")),
            row=2, col=1)
        fig_tren.add_trace(
            go.Scatter(x=df_prov["TAHUN"], y=df_prov["G_PENDUDUK"] * 100,
                       mode="lines+markers", name="Pertumbuhan Penduduk (%)",
                       line=dict(color="#e74c3c", width=2.5),
                       marker=dict(color="#e74c3c", size=5, symbol="circle")),
            row=3, col=1)

        fig_tren.update_layout(
            title=dict(text=f"<b>{prov_terpilih}</b>", font=dict(size=13, color="#f0b429")),
            height=480, showlegend=False, **CHART_THEME)
        fig_tren.update_annotations(font=dict(size=11, color="rgba(255,255,255,0.5)"))
        fig_tren.update_xaxes(title_text="Tahun", row=3, col=1, color="rgba(255,255,255,0.4)")
        fig_tren.update_yaxes(title_text="miliar Rp", row=1, col=1, color="rgba(255,255,255,0.4)")
        fig_tren.update_yaxes(title_text="%", row=2, col=1, color="rgba(255,255,255,0.4)")
        fig_tren.update_yaxes(title_text="%", row=3, col=1, color="rgba(255,255,255,0.4)")
        st.plotly_chart(fig_tren, use_container_width=True)

    with col_map[1]:
        hasil_map = hasil.copy()
        hasil_map["location"] = hasil_map["PROVINSI"].map(prov_mapping)
        hasil_map = hasil_map.dropna(subset=["location"])
        klas_tahun = df[df["TAHUN"] == tahun][["PROVINSI", "KLASIFIKASI"]]
        hasil_map = hasil_map.merge(klas_tahun, on="PROVINSI", how="left")

        fig_choropleth = px.choropleth(
            hasil_map, geojson=geojson, locations="location",
            featureidkey="properties.Propinsi", color="KLASIFIKASI",
            color_discrete_map=WARNA, hover_name="PROVINSI",
            hover_data={"KLASIFIKASI": True, "location": False},
            title=f"Klasifikasi Provinsi {tahun}")
        fig_choropleth.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
        fig_choropleth.update_layout(
            height=510, margin={"r": 0, "t": 40, "l": 0, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.75)", family="Inter", size=11),
            title=dict(
                text=f"Klasifikasi Provinsi {tahun}",
                font=dict(size=16, color="#f0b429", family="Inter, sans-serif", weight=700),
                x=0.5, xanchor="center",
            ),
            geo=dict(showframe=False, showcoastlines=False,
                     projection_type="mercator",
                     lataxis_range=[-10, 6], lonaxis_range=[95, 141]),
            legend=dict(orientation="h", yanchor="bottom", y=0.02,
                        xanchor="center", x=0.5,
                        font=dict(color="rgba(255,255,255,0.6)", size=10)),
        )
        st.plotly_chart(fig_choropleth, use_container_width=True)

    # -- Row 2: Metric Cards --
    col_m1, col_m2, col_m3 = st.columns(3)
    rata_prediksi = hasil['Pertumbuhan PDRB Dalam Rupiah'].mean()
    with col_m1:
        st.markdown(
            '<div class="metric-block">'
            '<div class="icon"></div>'
            '<div class="label">Rata-rata Prediksi</div>'
            '<div class="value">' + f'{rata_prediksi:.4f}' + '</div>'
            '<div class="sub">Seluruh provinsi</div>'
            '</div>', unsafe_allow_html=True)
    with col_m2:
        prov_tertinggi = hasil.loc[hasil['Pertumbuhan PDRB Dalam Rupiah'].idxmax(), 'PROVINSI']
        val_tertinggi = hasil['Pertumbuhan PDRB Dalam Rupiah'].max()
        st.markdown(''
            '<div class="metric-block">'
            '<div class="icon"></div>'
            '<div class="label">Provinsi Tertinggi</div>'
            '<div class="value">' + str(prov_tertinggi) + '</div>'
            '<div class="sub" style="color:#2ecc71;">' + str(val_tertinggi) + '</div>'
            '</div>',
            unsafe_allow_html=True)
    with col_m3:
        prov_terendah = hasil.loc[hasil['Pertumbuhan PDRB Dalam Rupiah'].idxmin(), 'PROVINSI']
        val_terendah = hasil['Pertumbuhan PDRB Dalam Rupiah'].min()
        st.markdown(''
            '<div class="metric-block">'
            '<div class="icon"></div>'
            '<div class="label">Provinsi Terendah</div>'
            '<div class="value">' + str(prov_terendah) + '</div>'
            '<div class="sub" style="color:#e74c3c;">' + str(val_terendah) + '</div>'
            '</div>',
            unsafe_allow_html=True)

    # -- Row 3: Bar Chart --
    section("Prediksi Pertumbuhan PDRB per Provinsi", "")
    fig_bar = px.bar(
        hasil.sort_values("Pertumbuhan PDRB Dalam Rupiah", ascending=False),
        x="PROVINSI", y="Pertumbuhan PDRB Dalam Rupiah",
        color="Pertumbuhan PDRB Dalam Rupiah",
        color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
    )
    fig_bar.update_layout(
        xaxis_tickangle=-45, height=400,
        **CHART_THEME,
        coloraxis_colorbar=dict(
            title="", tickfont=dict(color="rgba(255,255,255,0.5)", size=10),
            thickness=12, len=0.6),
    )
    fig_bar.update_traces(marker=dict(line=dict(width=0)))
    st.plotly_chart(fig_bar, use_container_width=True)

    # -- Row 4: Klasifikasi + Sebaran --
    col_b1, col_b2 = st.columns(2)

    with col_b1:
        section("Klasifikasi LOM / UPM / HIGH / LOW", "")
        hasil_klas = df[df["TAHUN"] == tahun][["PROVINSI", "KLASIFIKASI", "G_PDRB_IDR"]].copy()
        hasil_klas = hasil_klas.sort_values("KLASIFIKASI")
        fig_klas = px.bar(
            hasil_klas, x="PROVINSI", y="G_PDRB_IDR",
            color="KLASIFIKASI", color_discrete_map=WARNA,
            labels={"G_PDRB_IDR": "Pertumbuhan PDRB", "PROVINSI": "Provinsi"})
        fig_klas.update_layout(
            xaxis_tickangle=-45, showlegend=True, height=360,
            **CHART_THEME)
        fig_klas.update_traces(marker=dict(line=dict(width=0)))
        st.plotly_chart(fig_klas, use_container_width=True)

    with col_b2:
        section("Sebaran Prediksi per Region", "")
        fig_scatter = px.scatter(
            hasil, x="PROVINSI", y="Pertumbuhan PDRB Dalam Rupiah",
            size="Pertumbuhan PDRB Dalam Rupiah", color="REG",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"Pertumbuhan PDRB Dalam Rupiah": "Pertumbuhan"})
        fig_scatter.update_layout(
            xaxis_tickangle=-45, height=360,
            **CHART_THEME)
        fig_scatter.update_traces(marker=dict(line=dict(width=1, color="rgba(255,255,255,0.2)")))
        st.plotly_chart(fig_scatter, use_container_width=True)


# ══════════════════════════════════════════════
#  MENU 2 -- PREDIKSI PER PROVINSI
# ══════════════════════════════════════════════
elif menu == "Prediksi per Provinsi":
    section("Prediksi per Provinsi", "")

    col_sel = st.columns([1, 3])
    with col_sel[0]:
        selected_prov = st.selectbox("Pilih Provinsi", sorted(df["PROVINSI"].unique()))
    with col_sel[1]:
        st.markdown(
            f"<div style='padding-top:26px;color:rgba(255,255,255,0.45);font-size:0.8rem;'>"
            f"Analisis historis dan prediksi pertumbuhan PDRB -- "
            f"<b style='color:#00d4ff;'>{selected_prov}</b></div>",
            unsafe_allow_html=True)

    df_prov = df[df["PROVINSI"] == selected_prov].copy()
    hasil_prov = predict_all_provinces(model, encoders, df, 2025)
    pred_val = hasil_prov.loc[hasil_prov["PROVINSI"] == selected_prov, "PREDIKSI_G_PDRB_IDR"].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_prov["TAHUN"], y=df_prov["G_PDRB_IDR"],
        mode="lines+markers", name="Aktual",
        line=dict(color="#00d4ff", width=3),
        marker=dict(size=7, color="#00d4ff", symbol="circle")))
    if len(pred_val) > 0:
        fig.add_trace(go.Scatter(
            x=[2025], y=pred_val,
            mode="markers", name="Prediksi 2025",
            marker=dict(color="#f0b429", size=18, symbol="diamond",
                        line=dict(color="#fff", width=2))))
    fig.update_layout(
        title=dict(text=f"<b>Pertumbuhan PDRB -- {selected_prov}</b>",
                   font=dict(size=15, color="#f0b429")),
        xaxis_title="Tahun", yaxis_title="G_PDRB_IDR", height=440,
        **CHART_THEME)
    st.plotly_chart(fig, use_container_width=True)

    section("Data Historis", "")
    df_display = df_prov[["TAHUN", "PDRB_IDR_MLY", "PENDUDUK_RB", "G_PDRB_IDR", "KLASIFIKASI"]].copy()
    df_display.columns = ["Tahun", "PDRB (miliar Rp)", "Penduduk (ribu)", "Pertumbuhan PDRB", "Klasifikasi"]
    st.dataframe(
        df_display.style
        .format({"PDRB (miliar Rp)": "{:,.2f}", "Penduduk (ribu)": "{:,.2f}", "Pertumbuhan PDRB": "{:.4f}"})
        .map(lambda v: "color: #2ecc71" if isinstance(v, float) and v > 0 else "",
                  subset=["Pertumbuhan PDRB"]),
        use_container_width=True, height=280)


# ══════════════════════════════════════════════
#  MENU 3 -- WHAT-IF SIMULATOR
# ══════════════════════════════════════════════
elif menu == "What-If Simulator":
    api_key = get_api_key()

    st.markdown(''
        '<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">'
        '<div style="background:linear-gradient(135deg,#00d4ff,#0077b6);border-radius:12px;padding:8px 14px;">'
        '<span style="color:#fff;font-weight:800;font-size:0.9rem;">What-If</span>'
        '</div>'
        '<div style="color:rgba(255,255,255,0.4);font-size:0.8rem;font-weight:300;">'
        'Simulasikan perubahan parameter fiskal & demografi &amp; Analisis dampak pertumbuhan &amp; klasifikasi'
        '</div>'
        '</div>',
        unsafe_allow_html=True)

    col_left, col_mid, col_right = st.columns([2.2, 2.4, 1.6])

    # -- KOLOM KIRI: PETA --
    with col_left:
        sel_col1, sel_col2 = st.columns([1.3, 1])
        with sel_col1:
            selected_prov = st.selectbox("", sorted(df["PROVINSI"].unique()), key="ws_prov",
                                         label_visibility="collapsed",
                                         placeholder="Pilih provinsi...",
                                         on_change=lambda: clear_ws_results())
        with sel_col2:
            tahun = st.number_input("", 2016, 2025, 2025, key="ws_tahun",
                                    label_visibility="collapsed",
                                    on_change=lambda: clear_ws_results())

        location = prov_mapping.get(selected_prov)

        key_map = f"map_{selected_prov}_{tahun}"
        if "ws_map_fig" not in st.session_state or st.session_state.get("ws_map_key") != key_map:
            center_lat, center_lon, proj_scale = -2.5, 118, 3.2
            if location:
                geo_key = location
                for feature in geojson["features"]:
                    geo_nama = feature["properties"].get("PROVINSI") or feature["properties"].get("Propinsi", "")
                    if geo_nama == geo_key:
                        geom = feature["geometry"]
                        coords = geom["coordinates"]
                        all_lons, all_lats = [], []
                        def extract(g):
                            if g and isinstance(g, list) and len(g) > 0:
                                if isinstance(g[0], list) and len(g[0]) > 0 and isinstance(g[0][0], list):
                                    for ring in g:
                                        for c in ring:
                                            all_lons.append(c[0]); all_lats.append(c[1])
                                else:
                                    for c in g:
                                        all_lons.append(c[0]); all_lats.append(c[1])
                        if geom["type"] == "Polygon":
                            extract(coords)
                        else:
                            for poly in coords:
                                extract(poly)
                        if all_lons and all_lats:
                            center_lat = (min(all_lats) + max(all_lats)) / 2
                            center_lon = (min(all_lons) + max(all_lons)) / 2
                            lat_span = max(all_lats) - min(all_lats)
                            lon_span = max(all_lons) - min(all_lons)
                            max_span = max(lat_span, lon_span)
                            proj_scale = max(3, 7 - max_span * 0.32)
                        break

            klas_colors = {"LOW": "rgba(231,76,60,1)", "LOM": "rgba(240,180,41,1)", "UPM": "rgba(52,152,219,1)", "HIGH": "rgba(46,204,113,1)"}
            klas_map = {}
            df_klas_tahun = df[df["TAHUN"] == tahun]
            for _, r_k in df_klas_tahun.iterrows():
                klas_map[r_k["PROVINSI"]] = r_k["KLASIFIKASI"]

            fig_ws_map = go.Figure()
            for feature in geojson["features"]:
                propinsi = feature["properties"].get("PROVINSI") or feature["properties"].get("Propinsi", "")
                geom = feature["geometry"]
                coords_list = geom["coordinates"]
                if geom["type"] == "Polygon":
                    coords_list = [coords_list]
                is_selected = propinsi == location
                klas = klas_map.get(propinsi, "LOM")
                warna = klas_colors.get(klas, "#f0b429")
                if is_selected:
                    warna_fill = warna.replace(",1)", ",0.6)")
                    warna_line = warna
                    lebar_line = 2.5
                else:
                    warna_fill = warna.replace(",1)", ",0.15)")
                    warna_line = warna.replace(",1)", ",0.4)")
                    lebar_line = 0.8
                for polygon in coords_list:
                    lon_lat = [(c[0], c[1]) for c in polygon[0]]
                    lons, lats = zip(*lon_lat)
                    fig_ws_map.add_trace(go.Scattergeo(
                        lon=lons + (lons[0],), lat=lats + (lats[0],),
                        mode="lines", fill="toself",
                        fillcolor=warna_fill,
                        line=dict(color=warna_line, width=lebar_line),
                        hoverinfo="skip", showlegend=False,
                    ))
            fig_ws_map.update_geos(
                visible=False, bgcolor="rgba(0,0,0,0)",
                projection=dict(type="mercator", scale=proj_scale),
                center=dict(lat=center_lat, lon=center_lon),
                lataxis_range=[-14, 10], lonaxis_range=[92, 144],
            )
            fig_ws_map.update_layout(
                height=440, margin={"r": 0, "t": 0, "l": 0, "b": 0},
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                dragmode=False, hovermode=False,
            )
            fig_ws_map.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
            fig_ws_map.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
            st.session_state["ws_map_fig"] = fig_ws_map
            st.session_state["ws_map_key"] = key_map
        else:
            fig_ws_map = st.session_state["ws_map_fig"]

        st.plotly_chart(fig_ws_map, use_container_width=True,
                        config={"displayModeBar": False, "staticPlot": False})

    # -- KOLOM TENGAH: SLIDER --
    with col_mid:
        row = df[(df["PROVINSI"] == selected_prov) & (df["TAHUN"] == tahun)]
        if row.empty:
            tahun_used = int(df[df["PROVINSI"] == selected_prov]["TAHUN"].max())
            row = df[(df["PROVINSI"] == selected_prov) & (df["TAHUN"] == tahun_used)]
            tahun = tahun_used

        kurs_asli = float(row["KURS"].iloc[0])
        penduduk_asli = float(row["PENDUDUK_RB"].iloc[0])

        with st.form("ws_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown('<div class="param-card" style="--accent:#00d4ff"><div class="param-header"><span class="param-label">Belanja APBN</span><span class="param-icon"></span></div></div>', unsafe_allow_html=True)
                belanja_apbn_pct = st.slider("", 0.0, 50.0, DEFAULT_APBN_PCT, 0.5, format="%.1f%%", key="ws_apbn", label_visibility="collapsed",
                    help=f"Kenaikan 1% APBN -> PDRB +{KOEF_APBN:.4f}%")
            with c2:
                st.markdown('<div class="param-card" style="--accent:#2ecc71"><div class="param-header"><span class="param-label">Penyaluran KUR</span><span class="param-icon"></span></div></div>', unsafe_allow_html=True)
                kur_pct = st.slider("", 0.0, 50.0, DEFAULT_KUR_PCT, 0.5, format="%.1f%%", key="ws_kur", label_visibility="collapsed",
                    help=f"Kenaikan 1% KUR -> PDRB +{KOEF_KUR:.4f}%")
            with c3:
                st.markdown('<div class="param-card" style="--accent:#f0b429"><div class="param-header"><span class="param-label">Transfer Daerah</span><span class="param-icon"></span></div></div>', unsafe_allow_html=True)
                transfer_pct = st.slider("", 0.0, 50.0, DEFAULT_TRANSFER_PCT, 0.5, format="%.1f%%", key="ws_transfer", label_visibility="collapsed",
                    help=f"Kenaikan 1% TKD -> PDRB +{KOEF_TKD:.4f}%")

            c4, c5 = st.columns(2)
            with c4:
                st.markdown('<div class="param-card" style="--accent:#a855f7"><div class="param-header"><span class="param-label">KURS USD ke Rp</span><span class="param-icon"></span></div></div>', unsafe_allow_html=True)
                kurs_baru = st.slider("", 5000.0, 20000.0, kurs_asli, 50.0, format="%.0f", key="ws_kurs", label_visibility="collapsed")
            with c5:
                st.markdown('<div class="param-card" style="--accent:#ec4899"><div class="param-header"><span class="param-label">Jumlah Penduduk</span><span class="param-icon"></span></div></div>', unsafe_allow_html=True)
                min_penduduk = max(1.0, penduduk_asli * 0.5)
                max_penduduk = penduduk_asli * 1.5
                step_penduduk = max(1.0, penduduk_asli / 100)
                penduduk_baru = st.slider("", min_penduduk, max_penduduk, penduduk_asli, step_penduduk, format="%.0f", key="ws_penduduk", label_visibility="collapsed")

            submitted = st.form_submit_button("Jalankan Simulasi", use_container_width=True, type="primary")

        if submitted:
            changes = {
                "BELANJA_APBN_PCT": belanja_apbn_pct,
                "KUR_PCT": kur_pct,
                "TRANSFER_DAERAH_PCT": transfer_pct,
                "KURS": kurs_baru,
                "PENDUDUK_RB": penduduk_baru,
            }
            result = what_if_scenario(model, encoders, df, selected_prov, tahun, changes)
            st.session_state["ws_result"] = result
            st.session_state["ws_submitted"] = True
            st.rerun()

        # Info strip
        pdrb_str_2 = f"{float(row['PDRB_IDR_MLY'].iloc[0]):,.2f}"
        pdrbkap_str_2 = f"${float(row['PDRBKAP_USD'].iloc[0]):.2f}"
        st.markdown(''
            '<div class="info-strip">'
            '<span class="is-label">Provinsi</span>'
            '<span class="is-value">' + selected_prov + '</span>'
            '</div>'
            '<div class="info-strip">'
            '<span class="is-label">PDRB (miliar Rp)</span>'
            '<span class="is-value">' + pdrb_str_2 + '</span>'
            '</div>'
            '<div class="info-strip">'
            '<span class="is-label">PDRB / Kapita (USD)</span>'
            '<span class="is-value">' + pdrbkap_str_2 + '</span>'
            '</div>',
            unsafe_allow_html=True)

    # -- KOLOM KANAN --
    with col_right:
        pdrbkap_val = f"${float(row['PDRBKAP_USD'].iloc[0]):.2f}"
        klasifikasi_val = str(row['KLASIFIKASI'].iloc[0])
        klasifikasi_lower = klasifikasi_val.lower()
        st.markdown(''
            '<div class="result-card" style="text-align:center;">'
            '<div class="rc-label">Klasifikasi Saat Ini</div>'
            '<div style="margin:8px 0;">'
            '<span class="class-badge ' + klasifikasi_lower + '">' + klasifikasi_val + '</span>'
            '</div>'
            '<div style="color:rgba(255,255,255,0.3);font-size:0.65rem;">'
            'PDRB/kap: ' + pdrbkap_val + ''
            '</div>'
            '</div>',
            unsafe_allow_html=True)

        if api_key:
            if st.button("Analisis by System", use_container_width=True, key="ws_ai",
                         type="secondary"):
                with st.spinner("Menganalisis..."):
                    if "ws_result" not in st.session_state:
                        st.session_state["ws_result"] = what_if_scenario(
                            model, encoders, df, selected_prov, tahun,
                            {"BELANJA_APBN_PCT": 10.0, "KUR_PCT": 5.0, "TRANSFER_DAERAH_PCT": 15.0,
                             "KURS": kurs_asli, "PENDUDUK_RB": penduduk_asli})
                    r_ai = st.session_state["ws_result"]
                    data_ai = {
                        "provinsi": r_ai["provinsi"], "tahun": r_ai["tahun"],
                        "belanja_apbn_pct": 10.0, "kur_pct": 5.0, "transfer_daerah_pct": 15.0,
                        "kurs": kurs_asli, "penduduk_pct": 100.0,
                        "prediksi_original": r_ai["prediksi_original"],
                        "prediksi_skenario": r_ai["prediksi_skenario"],
                        "delta": r_ai["delta"],
                        "pdrbkap_original": r_ai["pdrbkap_original"],
                        "pdrbkap_skenario": r_ai["pdrbkap_skenario"],
                        "klasifikasi_original": r_ai["klasifikasi_original"],
                        "klasifikasi_skenario": r_ai["klasifikasi_skenario"],
                        "tahun_keluar_trap": r_ai["tahun_keluar_trap"],
                    }
                    analisis = analyze_scenario(data_ai)
                    if analisis:
                        st.session_state["ws_ai_result"] = analisis
                        st.rerun()
        else:
            st.markdown(
                "<p style='color:rgba(255,255,255,0.2);font-size:0.65rem;font-style:italic;text-align:center;'>"
                "Atur DEEPSEEK_API_KEY di .env atau Streamlit Secrets</p>", unsafe_allow_html=True)

        # -- Small box: Prediksi Keluar Economic Trap --
        if "ws_result" in st.session_state:
            r_trap = st.session_state["ws_result"]
            klas_asli = r_trap["klasifikasi_original"]
            pdrbkap_now = r_trap["pdrbkap_skenario"]
            delta_pdrbkap = r_trap["delta_pdrbkap"]

            target_map = {
                "LOW": (LOM_THRESHOLD, "Lower-Middle Income"),
                "LOM": (UPM_THRESHOLD, "Upper-Middle Income"),
                "UPM": (HIGH_THRESHOLD, "High Income"),
                "HIGH": (HIGH_THRESHOLD, "High Income"),
            }
            target, label_target = target_map.get(klas_asli, (UPM_THRESHOLD, "Upper-Middle Income"))

            # Hitung rata-rata growth PDRB/kap historis provinsi (5 tahun terakhir)
            df_prov_hist = df[df["PROVINSI"] == selected_prov]
            df_prov_hist = df_prov_hist[df_prov_hist["TAHUN"] >= df_prov_hist["TAHUN"].max() - 5]
            rata_growth_hist = df_prov_hist["G_PDRBKAP_USD"].mean() if "G_PDRBKAP_USD" in df_prov_hist.columns else 0

            # Tentukan annual delta untuk prediksi
            # Prioritas 1: delta dari simulasi (jika > 0)
            # Prioritas 2: rata-rata growth historis * PDRB/kap saat ini
            if delta_pdrbkap > 0.01:
                annual_delta = delta_pdrbkap
            elif rata_growth_hist > 0:
                annual_delta = pdrbkap_now * rata_growth_hist
            else:
                annual_delta = pdrbkap_now * 0.01  # fallback 1%

            gap = target - pdrbkap_now
            if gap <= 0:
                label = "< 1 tahun"
            else:
                tahun_prediksi = int(gap / annual_delta) + 1
                if tahun_prediksi > 200:
                    tahun_prediksi = int(200 / (1 + tahun_prediksi / 200)) + 1
                label = str(tahun_prediksi) + " tahun"

            st.markdown(
                '<div class="result-card" style="text-align:center;background:rgba(240,180,41,0.06);">'
                '<div class="rc-label">Prediksi Capai ' + label_target + '</div>'
                '<div class="rc-big" style="color:#f0b429;font-size:1.4rem;">' + label + '</div>'
                '<div class="rc-desc">Provinsi ' + selected_prov + ' diperkirakan mencapai ' + label_target + ' dalam ' + label + ' dengan skenario saat ini.</div>'
                '</div>',
                unsafe_allow_html=True)

    # -- HASIL SIMULASI --
    if "ws_submitted" in st.session_state and "ws_result" in st.session_state:
        result = st.session_state["ws_result"]
        pdrb_asli = float(row["PDRB_IDR_MLY"].iloc[0]) if not row.empty else 0
        efek_total = result["pdrb_total"] - pdrb_asli
        warna_delta = "#2ecc71" if result["delta"] >= 0 else "#e74c3c"

        klas_orig = result["klasifikasi_original"]
        klas_scen = result["klasifikasi_skenario"]
        if klas_orig != klas_scen:
            if klas_scen in ("UPM", "HIGH"):
                nar_klas = f"<b>{selected_prov}</b> <b>berhasil naik kelas</b> dari {klas_orig} ke {klas_scen}"
                gn_class = "gn-upm"
            else:
                nar_klas = f"<b>{selected_prov}</b> <b>turun kelas</b> dari {klas_orig} ke {klas_scen}"
                gn_class = "gn-lom"
        else:
            if klas_scen == "LOM":
                nar_klas = f"<b>{selected_prov}</b> tetap di LOM -- perlu akselerasi untuk keluar economic trap"
                gn_class = "gn-lom"
            elif klas_scen == "UPM":
                nar_klas = f"<b>{selected_prov}</b> tetap di UPM -- dorong pertumbuhan untuk capai HIGH"
                gn_class = "gn-stable"
            else:
                nar_klas = f"<b>{selected_prov}</b> tetap di HIGH -- kinerja ekonomi stabil"
                gn_class = "gn-upm"

        st.markdown("<div class='ws-divider'></div>", unsafe_allow_html=True)

        col_h1, col_h2, col_h3 = st.columns([1, 1, 1])

        with col_h1:
            skenario_str = f"{result['prediksi_skenario']:.4f}"
            original_str = f"{result['prediksi_original']:.4f}"
            delta_str = f"{result['delta']*100:+.2f}%"
            pdrb_str = f"{result['pdrb_total']:,.2f}"
            html_growth = (
                '<div class="result-card" style="text-align:center;">'
                '<div class="rc-label">Prediksi Pertumbuhan</div>'
                '<div class="rc-big" style="color:' + warna_delta + ';">' + skenario_str + '</div>'
                '<div class="rc-sub">Asli: ' + original_str + ' &nbsp;*&nbsp; <span style="color:' + warna_delta + ';">' + delta_str + '</span></div>'
                '<div class="rc-desc">PDRB: ' + pdrb_str + ' miliar Rp</div>'
                '</div>'
            )
            st.markdown(html_growth, unsafe_allow_html=True)

        with col_h2:
            if klas_scen in ("UPM", "HIGH"):
                trap_color = "#2ecc71"
                trap_title = "Bebas Economic Trap"
                trap_desc = f"{selected_prov} sudah di kategori {klas_scen}"
            elif result["delta_pdrbkap"] > 0 and result["tahun_keluar_trap"]:
                trap_color = "#f0b429"
                trap_title = f"Keluar: {result['tahun_keluar_trap']}"
                trap_desc = "Perkiraan tahun keluar dari Lower-Middle Income"
            else:
                trap_color = "#e74c3c"
                trap_title = "Terjebak LOM"
                trap_desc = "Percepatan pertumbuhan diperlukan"
            st.markdown(
                '<div class="result-card" style="text-align:center;">'
                '<div class="rc-label">Economic Trap</div>'
                '<div class="rc-big" style="color:' + trap_color + ';font-size:1.1rem;">' + trap_title + '</div>'
                '<div class="rc-desc">' + trap_desc + '</div>'
                '</div>',
                unsafe_allow_html=True)

            # Info target naik kelas dari indikator.xlsx
            if klas_scen != "HIGH":
                target_map = {
                    "LOW": {"naik": "LOM", "pdrb": 1146, "growth": 0.0186},
                    "LOM": {"naik": "UPM", "pdrb": 4516, "growth": 0.0169},
                    "UPM": {"naik": "HIGH", "pdrb": 14005, "growth": 0.017},
                }
                target_info = target_map.get(klas_scen)
                if target_info:
                    st.markdown(
                        '<div class="result-card" style="text-align:center;background:rgba(240,180,41,0.06);">'
                        '<div class="rc-label">Target Naik ke ' + target_info["naik"] + '</div>'
                        '<div class="rc-sub">Target PDRB/kap: $' + f"{target_info['pdrb']:,}" + '</div>'
                        '<div class="rc-sub" style="font-size:0.6rem;">Growth diperlukan: ' + f"{target_info['growth']*100:.2f}%" + '</div>'
                        '</div>',
                        unsafe_allow_html=True)

        with col_h3:
            pdrbkap_orig_res = f"${result['pdrbkap_original']:.2f}"
            pdrbkap_scen_res = f"${result['pdrbkap_skenario']:.2f}"
            st.markdown(
                '<div class="glass-narrative ' + gn_class + '">'
                '<div class="gn-title">Klasifikasi</div>'
                + nar_klas +
                '<div style="margin-top:6px;color:rgba(255,255,255,0.4);font-size:0.7rem;">'
                'PDRB/kap: ' + pdrbkap_orig_res + ' -> ' + pdrbkap_scen_res +
                '</div>'
                '</div>',
                unsafe_allow_html=True)

        if "ws_ai_result" in st.session_state:
            ai_text = st.session_state["ws_ai_result"].replace("\n", "<br>")
            st.markdown(
                '<div class="glass-narrative gn-ai">'
                '<div class="gn-title">Analisis by System</div>'
                + ai_text +
                '</div>',
                unsafe_allow_html=True)
st.markdown(
    '<div class="footer">'
    '<strong>Regional Economic Trap Early Warning System</strong> &nbsp;&bull;&nbsp;'
    'Sumber: DJPb & BPS (ddac.xlsx) &nbsp;&bull;&nbsp;'
    'Model: XGBoost &nbsp;&bull;&nbsp;'
    'Kemenkeu RP'
    '</div>',
    unsafe_allow_html=True)
