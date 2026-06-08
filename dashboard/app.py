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

from src.utils import MODELS_DIR, DATA_PROCESSED, DATA_EXTERNAL
from src.predict import load_model, predict_all_provinces, what_if_scenario

# ── Page config ──
st.set_page_config(
    page_title="Regional Economic Trap Early Warning System",
    layout="wide",
    page_icon="",
)

# ── CSS Premium ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    * { font-family: 'Inter', sans-serif !important; }

    /* ── Animated Background ── */
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

    /* ── Grid Overlay ── */
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

    /* ── Ensure content above backgrounds ── */
    .stApp > div:first-child { position: relative; z-index: 1; }
    .block-container { position: relative; z-index: 1; }

    /* ── Glass Header ── */
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

    /* ── Nav Radio ── */
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

    /* ── Glass Card ── */
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

    /* ── Metric Blocks ── */
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

    /* ── Section Headers ── */
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

    /* ── Slider ── */
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

    /* ── Dataframe ── */
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

    /* ── Info / Success ── */
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

    /* ── Button ── */
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

    /* ── Footer ── */
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

    /* ── Caption ── */
    .caption-text {
        color: rgba(255,255,255,0.3);
        font-size: 0.7rem;
        font-style: italic;
        text-align: center;
        margin-top: -8px;
    }

    /* ── Slider ── */
    .stSlider [data-baseweb="slider"] {
        padding-top: 6px;
    }
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: #f0b429 !important;
        border: none !important;
        width: 14px !important;
        height: 14px !important;
        margin-top: -6px !important;
    }
    .stSlider [data-baseweb="slider"] div[role="slider"]:focus {
        box-shadow: 0 0 0 3px rgba(240,180,41,0.2) !important;
    }
    .stSlider [data-baseweb="slider"] div {
        background: rgba(255,255,255,0.12) !important;
        height: 3px !important;
        border-radius: 2px !important;
    }
    .stSlider [data-testid="stSliderValue"] {
        color: #f0b429 !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
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
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown(f"""
<div class="header-wrap">
    <div class="header-title">
        <span>Regional Economic Trap<br>Early Warning System</span>
        <span class="header-badge">Data Analytics</span>
    </div>
    <div class="header-sub">Sistem Peringatan Dini Jebakan Ekonomi Regional berbasis Machine Learning &bull; Kemenkeu RP</div>
</div>
""", unsafe_allow_html=True)

# ── Navigasi ──
menu = st.radio("", ["Overview", "Prediksi per Provinsi", "What-If Simulator"],
                horizontal=True, label_visibility="collapsed")


# ── Cached resources ──
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

# ── Konstanta ──
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


# ══════════════════════════════════════════════
#  MENU 1 — OVERVIEW
# ══════════════════════════════════════════════
if menu == "Overview":
    tahun = st.slider("Tahun Analisis", 2016, 2025, 2025)
    hasil = predict_all_provinces(model, encoders, df, tahun)
    hasil = hasil.rename(columns={"PREDIKSI_G_PDRB_IDR": "Pertumbuhan PDRB Dalam Rupiah"})

    # ── Row 1: Peta + Line Chart ──
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

    # ── Row 2: Metric Cards ──
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(f"""
        <div class="metric-block">
            <div class="icon"></div>
            <div class="label">Rata-rata Prediksi</div>
            <div class="value">{hasil['Pertumbuhan PDRB Dalam Rupiah'].mean():.4f}</div>
            <div class="sub">Seluruh provinsi</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        prov_tertinggi = hasil.loc[hasil['Pertumbuhan PDRB Dalam Rupiah'].idxmax(), 'PROVINSI']
        val_tertinggi = hasil['Pertumbuhan PDRB Dalam Rupiah'].max()
        st.markdown(f"""
        <div class="metric-block">
            <div class="icon"></div>
            <div class="label">Provinsi Tertinggi</div>
            <div class="value">{prov_tertinggi}</div>
            <div class="sub" style="color:#2ecc71;">{val_tertinggi:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        prov_terendah = hasil.loc[hasil['Pertumbuhan PDRB Dalam Rupiah'].idxmin(), 'PROVINSI']
        val_terendah = hasil['Pertumbuhan PDRB Dalam Rupiah'].min()
        st.markdown(f"""
        <div class="metric-block">
            <div class="icon"></div>
            <div class="label">Provinsi Terendah</div>
            <div class="value">{prov_terendah}</div>
            <div class="sub" style="color:#e74c3c;">{val_terendah:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Row 3: Bar Chart ──
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

    # ── Row 4: Klasifikasi + Sebaran ──
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
#  MENU 2 — PREDIKSI PER PROVINSI
# ══════════════════════════════════════════════
elif menu == "Prediksi per Provinsi":
    section("Prediksi per Provinsi", "")

    col_sel = st.columns([1, 3])
    with col_sel[0]:
        selected_prov = st.selectbox("Pilih Provinsi", sorted(df["PROVINSI"].unique()))
    with col_sel[1]:
        st.markdown(
            f"<div style='padding-top:26px;color:rgba(255,255,255,0.45);font-size:0.8rem;'>"
            f"Analisis historis dan prediksi pertumbuhan PDRB — "
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
        title=dict(text=f"<b>Pertumbuhan PDRB — {selected_prov}</b>",
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
        .applymap(lambda v: "color: #2ecc71" if isinstance(v, float) and v > 0 else "",
                  subset=["Pertumbuhan PDRB"]),
        use_container_width=True, height=280)


# ══════════════════════════════════════════════
#  MENU 3 — WHAT-IF SIMULATOR
# ══════════════════════════════════════════════
elif menu == "What-If Simulator":
    section("What-If Simulator", "")
    st.markdown(
        "<p style='color:rgba(255,255,255,0.5);margin-top:-10px;margin-bottom:20px;font-size:0.85rem;'>"
        "Simulasikan perubahan nilai PDRB dan lihat dampaknya terhadap prediksi pertumbuhan ekonomi</p>",
        unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns([1, 1, 2])
    with col_s1:
        selected_prov = st.selectbox("Provinsi", sorted(df["PROVINSI"].unique()), key="ws_prov")
    with col_s2:
        tahun = st.number_input("Tahun", 2016, 2025, 2025, key="ws_tahun")

    row = df[(df["PROVINSI"] == selected_prov) & (df["TAHUN"] == tahun)]
    if row.empty:
        tahun_used = int(df[df["PROVINSI"] == selected_prov]["TAHUN"].max())
        row = df[(df["PROVINSI"] == selected_prov) & (df["TAHUN"] == tahun_used)]
        st.info(f"Data tahun {tahun} tidak tersedia, menggunakan data tahun {tahun_used}")
        tahun = tahun_used

    pdrb_asli = float(row["PDRB_IDR_MLY"].iloc[0])

    with col_s3:
        st.markdown(
            f"<div style='padding-top:26px;color:rgba(255,255,255,0.5);font-size:0.85rem;'>"
            f"Nilai dasar PDRB <b style='color:#00d4ff;'>{selected_prov}</b> "
            f"tahun <b>{tahun}</b>: "
            f"<b style='color:#fff;font-size:1.1rem;'>{pdrb_asli:,.2f}</b> miliar Rp</div>",
            unsafe_allow_html=True)

    pdrb_baru = st.slider(
        "Ubah nilai PDRB (miliar Rp):",
        min_value=float(pdrb_asli * 0.5), max_value=float(pdrb_asli * 1.5),
        value=float(pdrb_asli))

    if st.button("Jalankan Simulasi", use_container_width=True):
        result = what_if_scenario(model, encoders, df, selected_prov, tahun, {"PDRB_IDR_MLY": pdrb_baru})

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.markdown(f"""
            <div class="metric-block">
                <div class="label">Prediksi Asli</div>
                <div class="value">{result['prediksi_original']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_r2:
            st.markdown(f"""
            <div class="metric-block">
                <div class="label">Prediksi Skenario</div>
                <div class="value">{result['prediksi_skenario']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_r3:
            delta_val = result["delta"]
            warna_delta = "#2ecc71" if delta_val >= 0 else "#e74c3c"
            st.markdown(f"""
            <div class="metric-block" style="border-color: {warna_delta}40;">
                <div class="label">Perubahan</div>
                <div class="value" style="color: {warna_delta}">{delta_val:+.4f}</div>
                <div class="sub" style="color:{warna_delta};">{delta_val*100:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        arah = "meningkat" if result["delta"] > 0 else "menurun"
        st.success(
            f"Jika PDRB <b>{selected_prov}</b> diubah dari "
            f"<b>{pdrb_asli:,.2f}</b> menjadi <b>{pdrb_baru:,.2f}</b>, "
            f"maka pertumbuhan diprediksi <b>{arah}</b> dari "
            f"<b>{result['prediksi_original']:.4f}</b> menjadi "
            f"<b>{result['prediksi_skenario']:.4f}</b> "
            f"({result['delta']*100:+.2f}%)",
            unsafe_allow_html=True)

    section("Data Referensi", "")
    df_ref = row[["PROVINSI", "TAHUN", "PDRB_IDR_MLY", "PENDUDUK_RB", "G_PDRB_IDR", "KLASIFIKASI"]].copy()
    df_ref.columns = ["Provinsi", "Tahun", "PDRB (miliar Rp)", "Penduduk (ribu)", "Pertumbuhan PDRB", "Klasifikasi"]
    st.dataframe(
        df_ref.style.format({"PDRB (miliar Rp)": "{:,.2f}", "Penduduk (ribu)": "{:,.2f}",
                             "Pertumbuhan PDRB": "{:.4f}"}),
        use_container_width=True)


# ── Footer ──
st.markdown("""
<div class="footer">
    <strong>Regional Economic Trap Early Warning System</strong> &nbsp;&bull;&nbsp;
    Sumber: DJPb & BPS (ddac.xlsx) &nbsp;&bull;&nbsp;
    Model: XGBoost &nbsp;&bull;&nbsp;
    Kemenkeu RP
</div>
""", unsafe_allow_html=True)
