# =============================================================================
# BIO-EXPERT 360 PRO
# Version 7.0
# Analyse d'essais agronomiques
# =============================================================================

import os
import io
import zipfile
import shutil
import warnings
from datetime import date, datetime

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
import requests

from scipy import stats

warnings.filterwarnings("ignore")

# =============================================================================
# IMPORTS OPTIONNELS
# =============================================================================

HAS_PYSHP = False
HAS_STATSMODELS = False
HAS_SKLEARN = False
HAS_FPDF = False

try:
    import shapefile as pyshp
    HAS_PYSHP = True
except:
    pass

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except:
    pass

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except:
    pass

try:
    from fpdf import FPDF
    HAS_FPDF = True
except:
    pass

# =============================================================================
# CONFIGURATION STREAMLIT
# =============================================================================

st.set_page_config(
    page_title="Bio-Expert 360 PRO",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# PALETTE
# =============================================================================

COLOR_PRODUCT = "#0F766E"
COLOR_CONTROL = "#B91C1C"

PALETTE = {
    "Produit": COLOR_PRODUCT,
    "Témoin": COLOR_CONTROL
}

THEME = dict(

    plot_bgcolor="white",
    paper_bgcolor="white",

    font=dict(
        family="Inter",
        size=13,
        color="#1e293b"
    ),

    hovermode="x unified",

    margin=dict(
        l=60,
        r=40,
        t=60,
        b=60
    ),

    xaxis=dict(
        showgrid=True,
        gridcolor="#f1f5f9",
        linecolor="#cbd5e1"
    ),

    yaxis=dict(
        showgrid=True,
        gridcolor="#f1f5f9",
        linecolor="#cbd5e1"
    )
)

# =============================================================================
# STYLE CSS PREMIUM
# =============================================================================

st.markdown("""

<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html,body,[class*="css"]{

font-family:Inter;

}

.stApp{

background:#f8fafc;

}

/* HERO */

.hero{

padding:35px;

border-radius:18px;

background:linear-gradient(135deg,#0F766E,#134E4A);

box-shadow:0px 10px 35px rgba(0,0,0,.12);

margin-bottom:25px;

}

.hero h1{

color:white;

font-size:38px;

font-weight:800;

margin-bottom:5px;

}

.hero p{

color:#d1fae5;

font-size:17px;

}

/* KPI */

[data-testid="stMetric"]{

background:white;

padding:20px;

border-radius:15px;

border:1px solid #E2E8F0;

box-shadow:0px 2px 8px rgba(0,0,0,.05);

}

/* TITRES */

h2,h3{

font-weight:700;

}

/* BOITES */

.info-box{

padding:18px;

background:white;

border-radius:12px;

border-left:6px solid #0F766E;

margin-bottom:15px;

}

.success-box{

padding:18px;

background:#ECFDF5;

border-left:6px solid #16A34A;

border-radius:12px;

}

.warning-box{

padding:18px;

background:#FEF2F2;

border-left:6px solid #DC2626;

border-radius:12px;

}

</style>

""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTES
# =============================================================================

ALPHA_LEVELS = {

"10 % (Exploratoire)":0.10,
"5 % (Standard)":0.05,
"1 % (Très strict)":0.01

}

# =============================================================================
# DOSSIER TEMPORAIRE
# =============================================================================

TEMP_FOLDER="temp"

def clear_temp():

    if os.path.exists(TEMP_FOLDER):
        shutil.rmtree(TEMP_FOLDER)

    os.makedirs(TEMP_FOLDER)

# =============================================================================
# HEADER
# =============================================================================

st.markdown("""

<div class="hero">

<h1>🌱 Bio-Expert 360 PRO</h1>

<p>

Plateforme avancée d'analyse des essais agronomiques

•

Statistiques

•

ACP

•

ANOVA

•

Analyse météo

•

Rapport PDF

</p>

</div>

""", unsafe_allow_html=True)
