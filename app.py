import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import requests
import io
import zipfile
import os
import numpy as np
import shutil
from datetime import datetime, date, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# ══════════════════════════════════════════════════════════════════════════
# 1. CONFIG PAGE & STYLE
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Bio-Expert 360", layout="wide", page_icon="🌱",
                    initial_sidebar_state="expanded")

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; color:#1b4332; }
[data-testid="stMetric"] {
    background: #ffffff; border-radius: 14px; padding: 14px 16px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06); border: 1px solid #eef1ee;
}
h1, h2, h3 { color:#1b4332; }
.verdict-sig   { background:linear-gradient(135deg,#d4edda,#c3e6cb); border-left:6px solid #28a745; padding:16px 20px; border-radius:10px; color:#155724; font-size:1.05rem; }
.verdict-nosig { background:linear-gradient(135deg,#f8d7da,#f1c0c5); border-left:6px solid #dc3545; padding:16px 20px; border-radius:10px; color:#721c24; font-size:1.05rem; }
.stress-high   { background:#fdecea; border-left:6px solid #e74c3c; padding:14px 18px; border-radius:10px; color:#7b241c; }
.stress-low    { background:#eafaf1; border-left:6px solid #27ae60; padding:14px 18px; border-radius:10px; color:#145a32; }
.vulgarisation { background:#f4f6f5; border-left:4px solid #52796f; padding:12px 16px; margin-bottom:10px; border-radius:8px; font-size:0.92rem; }
.section-title { font-size:1.3rem; font-weight:700; color:#1b4332; margin-top:8px; }
hr { border-top: 1px solid #e0e4e1; }
</style>
""", unsafe_allow_html=True)


def clear_temp():
    if os.path.exists("temp"):
        shutil.rmtree("temp")
    os.makedirs("temp")


# ══════════════════════════════════════════════════════════════════════════
# 2. RÉFÉRENTIEL CULTURES (utilisé pour les seuils de stress thermique)
# ══════════════════════════════════════════════════════════════════════════
PARAM_CULTURES = {
    "Blé Tendre": {"t_echaudage": 25, "t_critique": 30, "t_gel": -2, "precip_min_jour": 0.5},
    "Maïs":       {"t_echaudage": 32, "t_critique": 36, "t_gel": 0,  "precip_min_jour": 0.5},
    "Orge":       {"t_echaudage": 25, "t_critique": 30, "t_gel": -3, "precip_min_jour": 0.5},
    "Colza":      {"t_echaudage": 27, "t_critique": 32, "t_gel": -5, "precip_min_jour": 0.5},
}
ALPHA_LEVELS = {"5 % (standard)": 0.05, "1 % (strict)": 0.01, "10 % (exploratoire)": 0.10}

# ══════════════════════════════════════════════════════════════════════════
# 3. SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🌱 Bio-Expert 360")
    st.caption("Analyse d'essais en bandes — v3.0")

    with st.expander("📥 IMPORTATION DONNÉES", expanded=True):
        uploaded_file = st.file_uploader("Fichier QGIS (.zip)", type=["zip"])
        st.caption("Colonnes attendues : `bande`, `rdt`, `potentiel` (optionnel)")

    with st.expander("🌾 CONFIGURATION ESSAI", expanded=True):
        culture = st.selectbox("Culture", list(PARAM_CULTURES.keys()))
        d_semis = st.date_input("Date de Semis", date(2024, 10, 20))
        d_appli = st.date_input("Date d'Application produit", date(2025, 3, 10))
        d_recolt = st.date_input("Date de Récolte", date(2025, 7, 15))
        alpha = st.selectbox("Seuil de significativité α", list(ALPHA_LEVELS.keys()))
        alpha_v = ALPHA_LEVELS[alpha]
        clean_iqr = st.checkbox("Nettoyage strict des outliers (IQR 1.2)", value=True)

    with st.expander("📊 OPTIONS STATISTIQUES", expanded=True):
        run_anova = st.checkbox("Activer l'ANOVA spatiale (si zones de potentiel)", value=True)

    with st.expander("💰 ÉCONOMIE", expanded=True):
        prix_vente = st.number_input("Prix de vente (€/T)", value=210)
        cout_prod = st.number_input("Coût Produit (€/ha)", value=45)

    with st.expander("🌦️ MÉTÉO", expanded=True):
        st.caption("La position est déduite automatiquement du centre de votre parcelle. "
                    "Vous pouvez la corriger manuellement si besoin.")
        manual_coords = st.checkbox("Forcer des coordonnées manuelles")
        man_lat = st.number_input("Latitude", value=48.85, format="%.4f") if manual_coords else None
        man_lon = st.number_input("Longitude", value=2.35, format="%.4f") if manual_coords else None


# ══════════════════════════════════════════════════════════════════════════
# 4. FONCTIONS STATISTIQUES (simplifiées)
# ══════════════════════════════════════════════════════════════════════════
def cohen_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.std(a, ddof=1) ** 2 + (nb - 1) * np.std(b, ddof=1) ** 2) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


def interpret_d(d):
    d = abs(d)
    if d < 0.2: return "négligeable"
    if d < 0.5: return "faible"
    if d < 0.8: return "moyen"
    if d < 1.2: return "fort"
    return "très fort"


def run_main_test(data_p, data_t, alpha_v=0.05):
    """Un seul test, auto-sélectionné selon normalité/homogénéité — résultat simple et lisible."""
    n_p, n_t = len(data_p), len(data_t)
    _, p_shap_p = stats.shapiro(data_p) if n_p >= 3 else (None, None)
    _, p_shap_t = stats.shapiro(data_t) if n_t >= 3 else (None, None)
    _, p_lev = stats.levene(data_p, data_t)

    normal = (p_shap_p or 0) > alpha_v and (p_shap_t or 0) > alpha_v
    homog = p_lev > alpha_v

    if normal and homog:
        t_stat, p_main = stats.ttest_ind(data_p, data_t)
        test_nom = "Test de Student (paramétrique)"
    elif normal:
        t_stat, p_main = stats.ttest_ind(data_p, data_t, equal_var=False)
        test_nom = "Test de Welch (variances inégales)"
    else:
        t_stat, p_main = stats.mannwhitneyu(data_p, data_t, alternative='two-sided')
        test_nom = "Test de Mann-Whitney (non-paramétrique)"

    d = cohen_d(data_p.values, data_t.values)
    return {
        'name': test_nom, 'p': p_main, 'd': d, 'label': interpret_d(d),
        'shapiro_p': p_shap_p, 'shapiro_t': p_shap_t, 'levene_p': p_lev,
    }


def run_anova_analysis(df_final, alpha_v=0.05):
    if not HAS_STATSMODELS:
        return None, "statsmodels non installé.", None, False
    if 'potentiel' not in df_final.columns:
        return None, "Colonne 'potentiel' absente du fichier.", None, False
    nb_zones = df_final['potentiel'].dropna().nunique()
    if nb_zones <= 1:
        return None, "Au moins 2 zones de potentiel différentes sont nécessaires pour l'ANOVA.", None, False

    formula = "rdt ~ C(grp) + C(potentiel) + C(grp):C(potentiel)"
    try:
        model = smf.ols(formula, data=df_final).fit()
        anova_t = sm.stats.anova_lm(model, typ=2)
        return anova_t, "📐 ANOVA à 2 facteurs : Traitement × Zone de potentiel", model, True
    except Exception as e:
        return None, f"Erreur lors du calcul statistique : {e}", None, False


# ══════════════════════════════════════════════════════════════════════════
# 5. FONCTION MÉTÉO (Open-Meteo, gratuit, sans clé API)
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def fetch_weather(lat, lon, start, end):
    """Récupère les données météo journalières (historique ou prévision) via Open-Meteo."""
    today = date.today()
    url_parts = []
    start_str, end_str = str(start), str(end)

    # Historique (archive) pour les dates passées
    if start < today:
        archive_end = min(end, today - timedelta(days=1))
        if start <= archive_end:
            url = ("https://archive-api.open-meteo.com/v1/archive"
                   f"?latitude={lat}&longitude={lon}&start_date={start}&end_date={archive_end}"
                   "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
                   "&timezone=auto")
            url_parts.append(url)

    # Prévision pour les dates futures (jusqu'à ~16j) — sinon ignoré
    if end >= today:
        forecast_start = max(start, today)
        if forecast_start <= end:
            url = ("https://api.open-meteo.com/v1/forecast"
                   f"?latitude={lat}&longitude={lon}&start_date={forecast_start}&end_date={end}"
                   "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
                   "&timezone=auto")
            url_parts.append(url)

    frames = []
    for url in url_parts:
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            d = r.json().get("daily", {})
            if d:
                frames.append(pd.DataFrame(d))
        except Exception:
            continue

    if not frames:
        return None
    df_w = pd.concat(frames, ignore_index=True).drop_duplicates(subset="time")
    df_w["time"] = pd.to_datetime(df_w["time"])
    df_w = df_w.sort_values("time").reset_index(drop=True)
    return df_w


def compute_stress(df_w, params, d_appli):
    """Calcule les jours de stress thermique et hydrique."""
    df_w = df_w.copy()
    df_w["stress_chaleur"] = df_w["temperature_2m_max"] >= params["t_echaudage"]
    df_w["stress_critique"] = df_w["temperature_2m_max"] >= params["t_critique"]
    df_w["stress_gel"] = df_w["temperature_2m_min"] <= params["t_gel"]
    df_w["jour_sec"] = df_w["precipitation_sum"] < params["precip_min_jour"]

    # Stress hydrique = séquences de jours secs consécutifs ≥ 7 jours
    df_w["run_id"] = (df_w["jour_sec"] != df_w["jour_sec"].shift()).cumsum()
    run_len = df_w.groupby("run_id")["jour_sec"].transform("size")
    df_w["stress_secheresse"] = df_w["jour_sec"] & (run_len >= 7)

    return df_w


# ══════════════════════════════════════════════════════════════════════════
# 6. LECTURE FICHIER
# ══════════════════════════════════════════════════════════════════════════
if not uploaded_file:
    st.title("🌱 Bio-Expert 360")
    st.info("👈 Importez votre fichier QGIS (.zip) dans la barre latérale pour démarrer l'analyse.")
    st.markdown("""
    **Colonnes attendues dans le fichier .shp / attributs QGIS :**

    | Colonne | Obligatoire | Description |
    |---------|-------------|-------------|
    | `bande` | ✅ | Identifiant de bande (ex: A, B, Produit, Témoin) |
    | `rdt` | ✅ | Rendement (qtx/ha ou t/ha) |
    | `potentiel` | ⚙️ recommandé | Zone de potentiel sol (ex: Faible, Moyen, Fort) |

    **Ce que fait l'application :**
    - 📊 Comparaison statistique Produit vs Témoin (test auto-sélectionné)
    - 📐 ANOVA spatiale Traitement × Zone de potentiel (si disponible)
    - 🗺️ Carte interactive de la parcelle (par bande et par rendement)
    - 🌦️ Analyse météo de la période semis → récolte avec détection de stress
    """)
    st.stop()

try:
    clear_temp()
    with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
        z.extractall("temp")

    shp_files = [f for f in os.listdir("temp") if f.endswith('.shp')]
    if not shp_files:
        st.error("❌ Aucun fichier .shp trouvé dans le zip.")
        st.stop()

    gdf_raw = gpd.read_file(os.path.join("temp", shp_files[0]))
    if gdf_raw.crs is None:
        gdf_raw.crs = "EPSG:2154"
    gdf = gdf_raw.to_crs(epsg=4326)

    df = pd.DataFrame(gdf.drop(columns='geometry'))
    df.columns = df.columns.str.lower().str.strip()

    missing = [c for c in ['bande', 'rdt'] if c not in df.columns]
    if missing:
        st.error(f"❌ Colonnes manquantes : {missing}. Colonnes disponibles : {list(df.columns)}")
        st.stop()

    df['rdt'] = pd.to_numeric(df['rdt'], errors='coerce')
    df = df.dropna(subset=['rdt'])

except Exception as e:
    st.error(f"❌ Erreur lecture fichier : {e}")
    st.stop()

# ── Centroïde pour la météo ─────────────────────────────────────────────
centroid_lat = float(gdf.geometry.centroid.y.mean())
centroid_lon = float(gdf.geometry.centroid.x.mean())
weather_lat = man_lat if manual_coords else centroid_lat
weather_lon = man_lon if manual_coords else centroid_lon

# ── Options dynamiques ───────────────────────────────────────────────────
with st.sidebar:
    bandes_dispo = sorted(df['bande'].unique().tolist())
    val_p = st.selectbox("Bande = 'Produit' ?", bandes_dispo)

df['grp'] = df['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')
df_travail = df.copy()

n_initial = len(df_travail)
if clean_iqr:
    clean_list = []
    for g in ['Produit', 'Témoin']:
        sub = df_travail[df_travail['grp'] == g]
        if not sub.empty:
            q1, q3 = sub['rdt'].quantile([0.25, 0.75])
            iqr = q3 - q1
            sub = sub[(sub['rdt'] >= q1 - 1.2 * iqr) & (sub['rdt'] <= q3 + 1.2 * iqr)]
            clean_list.append(sub)
    df_final = pd.concat(clean_list) if clean_list else df_travail.copy()
else:
    df_final = df_travail.copy()

n_removed = n_initial - len(df_final)
data_p = df_final[df_final['grp'] == 'Produit']['rdt'].dropna()
data_t = df_final[df_final['grp'] == 'Témoin']['rdt'].dropna()
n_p, n_t = len(data_p), len(data_t)

has_enough = n_p > 3 and n_t > 3
gain = data_p.mean() - data_t.mean() if has_enough else 0.0
marge = ((gain / 10) * prix_vente) - cout_prod

# ══════════════════════════════════════════════════════════════════════════
# 7. EN-TÊTE & KPIs
# ══════════════════════════════════════════════════════════════════════════
st.title("🌱 Bio-Expert 360")
st.caption(f"Culture : **{culture}** · Semis {d_semis.strftime('%d/%m/%Y')} → Récolte {d_recolt.strftime('%d/%m/%Y')}")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Obs. Produit", f"{n_p}")
c2.metric("Obs. Témoin", f"{n_t}")
c3.metric("Moy. Produit", f"{data_p.mean():.1f} qtx" if has_enough else "—")
c4.metric("Moy. Témoin", f"{data_t.mean():.1f} qtx" if has_enough else "—")
c5.metric("Gain Moyen", f"+{gain:.2f} qtx" if has_enough else "—")
c6.metric("Marge Nette", f"{marge:.0f} €/ha" if has_enough else "—")
if n_removed > 0:
    st.caption(f"⚠️ {n_removed} points supprimés par nettoyage IQR sur {n_initial} observations.")

if not has_enough:
    st.error("Données insuffisantes (< 4 obs. par groupe). Vérifiez votre fichier.")
    st.stop()

stat_res = run_main_test(data_p, data_t, alpha_v=alpha_v)
sig = stat_res['p'] < alpha_v
html = f"""<div class="{'verdict-sig' if sig else 'verdict-nosig'}">
<strong>{'✅ Impact Significatif' if sig else '❌ Impact Non Démontré'}</strong>
— {stat_res['name']} · p = {stat_res['p']:.4f} · Cohen's d = {stat_res['d']:.2f} ({stat_res['label']})
{'<br>L\'effet n\'est probablement pas dû au hasard (confiance ≥ '+str(int((1-alpha_v)*100))+'%).' if sig else '<br>La variabilité de la parcelle empêche de conclure à un effet du produit.'}
</div>"""
st.markdown(html, unsafe_allow_html=True)
st.markdown("")

# ══════════════════════════════════════════════════════════════════════════
# 8. ONGLETS
# ══════════════════════════════════════════════════════════════════════════
tab_rdt, tab_anova, tab_map, tab_meteo = st.tabs([
    "📊 Résultats & Distribution",
    "📐 ANOVA Spatiale",
    "🗺️ Carte parcelle",
    "🌦️ Météo & Stress",
])

# ─────────────────────────────────────────────────────────────────────────
# TAB 1 — Distribution
# ─────────────────────────────────────────────────────────────────────────
with tab_rdt:
    st.markdown("""
    <div class="vulgarisation">
    💡 Un seul test statistique est sélectionné automatiquement (Student, Welch, ou Mann-Whitney)
    selon la normalité et l'homogénéité de vos données, pour vous donner une réponse claire et fiable.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        fig_box = px.box(
            df_final, x="grp", y="rdt", color="grp", points="all", notched=True,
            color_discrete_map={'Produit': '#2ecc71', 'Témoin': '#e74c3c'},
            labels={"grp": "Groupe", "rdt": "Rendement (qtx/ha)"},
            title="Distribution des rendements"
        )
        st.plotly_chart(fig_box, use_container_width=True)
    with col2:
        fig_viol = px.violin(
            df_final, x="grp", y="rdt", color="grp", box=True,
            color_discrete_map={'Produit': '#2ecc71', 'Témoin': '#e74c3c'},
            labels={"grp": "Groupe", "rdt": "Rendement (qtx/ha)"},
            title="Densité de probabilité"
        )
        st.plotly_chart(fig_viol, use_container_width=True)

    desc = df_final.groupby('grp')['rdt'].describe().round(2)
    desc.columns = ['N', 'Moyenne', 'Écart-type', 'Min', 'Q25', 'Médiane', 'Q75', 'Max']
    st.dataframe(desc, use_container_width=True)

    d1, d2, d3 = st.columns(3)
    sp_p, sp_t, lev = stat_res['shapiro_p'], stat_res['shapiro_t'], stat_res['levene_p']
    d1.metric("Shapiro (Produit)", f"p = {sp_p:.4f}" if sp_p else "—", "✅ Normal" if (sp_p or 0) > alpha_v else "⚠️ Asymétrique")
    d2.metric("Shapiro (Témoin)", f"p = {sp_t:.4f}" if sp_t else "—", "✅ Normal" if (sp_t or 0) > alpha_v else "⚠️ Asymétrique")
    d3.metric("Levene (Variances)", f"p = {lev:.4f}", "✅ Homogène" if lev > alpha_v else "⚠️ Hétérogène")

# ─────────────────────────────────────────────────────────────────────────
# TAB 2 — ANOVA
# ─────────────────────────────────────────────────────────────────────────
with tab_anova:
    st.markdown("""
    <div class="vulgarisation">
    📐 L'ANOVA sépare ce qui revient à la qualité du sol de ce qui revient à l'effet réel du produit,
    et révèle si le produit fonctionne différemment selon la zone de potentiel.
    </div>
    """, unsafe_allow_html=True)

    if not run_anova:
        st.info("ANOVA désactivée dans la barre latérale.")
    elif not HAS_STATSMODELS:
        st.warning("statsmodels requis : ajoutez-le à votre requirements.txt")
    else:
        anova_table, anova_title, anova_model, has_pot = run_anova_analysis(df_final, alpha_v)
        if anova_table is None:
            st.error(f"❌ {anova_title}")
        else:
            st.subheader(anova_title)
            at = anova_table.copy()
            at.columns = [c.replace('PR(>F)', 'p-value').replace('sum_sq', 'SCE').replace('mean_sq', 'CME') for c in at.columns]
            at = at.round(4)

            def style_pval(val):
                try:
                    v = float(val)
                    if v < 0.001: return 'background-color:#d4edda; font-weight:bold; color:#155724;'
                    if v < alpha_v: return 'background-color:#fff3cd; color:#856404;'
                    return ''
                except Exception:
                    return ''

            target_cols = [c for c in at.columns if 'p-value' in c.lower()]
            styled_at = at.style.map(style_pval, subset=target_cols) if target_cols else at.style
            st.dataframe(styled_at, use_container_width=True)

            if anova_model is not None:
                col1, col2, col3 = st.columns(3)
                col1.metric("R² (variance expliquée)", f"{anova_model.rsquared:.1%}")
                col2.metric("R² ajusté", f"{anova_model.rsquared_adj:.3f}")
                col3.metric("F-Value globale", f"{anova_model.fvalue:.2f}")

            st.subheader("📊 Rendement par Zone × Traitement")
            pivot = df_final.groupby(['potentiel', 'grp'])['rdt'].agg(['mean', 'std', 'count']).round(2)
            pivot.columns = ['Rendement Moyen', 'Écart-Type', 'Nombre de points']
            st.dataframe(pivot, use_container_width=True)

            fig_inter = px.box(
                df_final, x="potentiel", y="rdt", color="grp",
                color_discrete_map={'Produit': '#2ecc71', 'Témoin': '#e74c3c'},
                title="Le produit fonctionne-t-il mieux sur certaines zones de sol ?"
            )
            st.plotly_chart(fig_inter, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────
# TAB 3 — Carte parcelle (très visuelle)
# ─────────────────────────────────────────────────────────────────────────
with tab_map:
    if 'geometry' not in gdf.columns or gdf.empty:
        st.info("Géométrie non disponible ou fichier vide.")
    else:
        try:
            gdf_plot = gdf.copy()
            gdf_plot['grp'] = gdf_plot['bande'].apply(lambda x: 'Produit' if x == val_p else 'Témoin')
            gdf_plot = gdf_plot.merge(
                df_final[['rdt']].assign(idx=df_final.index),
                left_index=True, right_on='idx', how='left'
            )
            gdf_plot['rdt_carte'] = gdf_plot['rdt']
            gdf_plot['lat'] = gdf_plot.geometry.centroid.y
            gdf_plot['lon'] = gdf_plot.geometry.centroid.x

            st.markdown("### 🗺️ Carte interactive de votre parcelle")
            vue = st.radio("Colorer la carte par :", ["Bande (Produit / Témoin)", "Rendement (qtx/ha)"], horizontal=True)

            center_lat = float(gdf_plot['lat'].median())
            center_lon = float(gdf_plot['lon'].mean())

            if vue.startswith("Bande"):
                fig_map = px.scatter_mapbox(
                    gdf_plot, lat="lat", lon="lon", color="grp", size="rdt_carte", size_max=16,
                    color_discrete_map={'Produit': '#2ecc71', 'Témoin': '#e74c3c'},
                    mapbox_style="open-street-map", zoom=16,
                    center={"lat": center_lat, "lon": center_lon}, opacity=0.85,
                    hover_data={'bande': True, 'rdt_carte': ':.1f'},
                    labels={'rdt_carte': 'Rendement (qtx/ha)', 'grp': 'Groupe'},
                    title="Carte par bande — Produit vs Témoin"
                )
            else:
                fig_map = px.scatter_mapbox(
                    gdf_plot, lat="lat", lon="lon", color="rdt_carte", size="rdt_carte", size_max=16,
                    color_continuous_scale="RdYlGn", mapbox_style="open-street-map", zoom=16,
                    center={"lat": center_lat, "lon": center_lon}, opacity=0.9,
                    hover_data={'bande': True, 'rdt_carte': ':.1f'},
                    labels={'rdt_carte': 'Rendement (qtx/ha)'},
                    title="Carte de rendement — zones chaudes/froides"
                )

            fig_map.update_layout(height=650, margin={"r": 0, "t": 40, "l": 0, "b": 0})
            st.plotly_chart(fig_map, use_container_width=True)

            if 'potentiel' in gdf_plot.columns:
                with st.expander("🌍 Vue par zone de potentiel"):
                    df_pot_carte = gdf_plot.groupby('potentiel').agg(
                        rdt_moyen=('rdt_carte', 'mean'), lat=('lat', 'mean'), lon=('lon', 'mean')
                    ).reset_index()
                    df_pot_carte['rdt_moyen'] = df_pot_carte['rdt_moyen'].round(1)
                    fig_pot = px.scatter_mapbox(
                        df_pot_carte, lat="lat", lon="lon", color="potentiel", size="rdt_moyen", size_max=24,
                        color_discrete_sequence=px.colors.qualitative.Bold, mapbox_style="open-street-map",
                        zoom=16, center={"lat": center_lat, "lon": center_lon},
                        title="Rendement moyen par zone de potentiel"
                    )
                    fig_pot.update_layout(height=500, margin={"r": 0, "t": 40, "l": 0, "b": 0})
                    st.plotly_chart(fig_pot, use_container_width=True)
        except Exception as e:
            st.warning(f"Carte indisponible : {e}")

# ─────────────────────────────────────────────────────────────────────────
# TAB 4 — Météo & Stress
# ─────────────────────────────────────────────────────────────────────────
with tab_meteo:
    st.markdown(f"""
    <div class="vulgarisation">
    🌦️ Analyse météo de la période <b>semis → récolte</b> au point GPS de votre parcelle
    (lat {weather_lat:.4f}, lon {weather_lon:.4f}). Les seuils de stress sont calibrés pour la culture
    sélectionnée : <b>{culture}</b> (chaleur ≥ {PARAM_CULTURES[culture]['t_echaudage']}°C,
    critique ≥ {PARAM_CULTURES[culture]['t_critique']}°C, gel ≤ {PARAM_CULTURES[culture]['t_gel']}°C,
    sécheresse = 7 jours consécutifs sans pluie significative).
    </div>
    """, unsafe_allow_html=True)

    if d_recolt < d_semis:
        st.error("La date de récolte doit être postérieure à la date de semis.")
    else:
        with st.spinner("Récupération des données météo…"):
            df_w = fetch_weather(weather_lat, weather_lon, d_semis, d_recolt)

        if df_w is None or df_w.empty:
            st.warning("Données météo indisponibles pour cette période/localisation. "
                       "Vérifiez vos coordonnées ou réessayez plus tard.")
        else:
            params = PARAM_CULTURES[culture]
            df_w = compute_stress(df_w, params, d_appli)

            nb_chaleur = int(df_w['stress_chaleur'].sum())
            nb_critique = int(df_w['stress_critique'].sum())
            nb_gel = int(df_w['stress_gel'].sum())
            nb_secheresse = int(df_w['stress_secheresse'].sum())
            total_jours = len(df_w)

            stress_total = nb_critique > 0 or nb_secheresse > 0
            html_s = f"""<div class="{'stress-high' if stress_total else 'stress-low'}">
            <strong>{'⚠️ Stress détecté pendant le cycle' if stress_total else '✅ Aucun stress majeur détecté'}</strong>
            — {nb_chaleur} jour(s) ≥ seuil d'échaudage, {nb_critique} jour(s) de chaleur critique,
            {nb_gel} jour(s) de gel, {nb_secheresse} jour(s) en séquence de sécheresse, sur {total_jours} jours analysés.
            </div>"""
            st.markdown(html_s, unsafe_allow_html=True)
            st.markdown("")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🔥 Jours chaleur (échaudage)", nb_chaleur)
            m2.metric("🌡️ Jours chaleur critique", nb_critique)
            m3.metric("❄️ Jours de gel", nb_gel)
            m4.metric("🏜️ Jours en séquence sèche", nb_secheresse)

            # ── Graphique combiné température / précipitations / stress ──
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_w['time'], y=df_w['precipitation_sum'], name="Précipitations (mm)",
                marker_color='#3498db', opacity=0.5, yaxis='y2'
            ))
            fig.add_trace(go.Scatter(
                x=df_w['time'], y=df_w['temperature_2m_max'], name="T° Max",
                line=dict(color='#e74c3c', width=2), mode='lines'
            ))
            fig.add_trace(go.Scatter(
                x=df_w['time'], y=df_w['temperature_2m_min'], name="T° Min",
                line=dict(color='#5dade2', width=2), mode='lines', fill='tonexty', fillcolor='rgba(93,173,226,0.08)'
            ))

            fig.add_hline(y=params['t_echaudage'], line_dash="dash", line_color="orange",
                          annotation_text=f"Seuil échaudage ({params['t_echaudage']}°C)")
            fig.add_hline(y=params['t_critique'], line_dash="dash", line_color="red",
                          annotation_text=f"Seuil critique ({params['t_critique']}°C)")
            fig.add_hline(y=params['t_gel'], line_dash="dash", line_color="#2980b9",
                          annotation_text=f"Seuil gel ({params['t_gel']}°C)")

            # Zones de stress thermique
            stress_days = df_w[df_w['stress_critique']]
            for _, row in stress_days.iterrows():
                fig.add_vrect(x0=row['time'] - pd.Timedelta(hours=12), x1=row['time'] + pd.Timedelta(hours=12),
                              fillcolor="red", opacity=0.08, line_width=0)

            # Zones de sécheresse
            dry_days = df_w[df_w['stress_secheresse']]
            for _, row in dry_days.iterrows():
                fig.add_vrect(x0=row['time'] - pd.Timedelta(hours=12), x1=row['time'] + pd.Timedelta(hours=12),
                              fillcolor="#d35400", opacity=0.06, line_width=0)

            # Marqueur date d'application produit
            appli_ts = pd.Timestamp(d_appli)
            if df_w['time'].min() <= appli_ts <= df_w['time'].max():
                fig.add_vline(x=appli_ts, line_dash="dot", line_color="green",
                              annotation_text="Application produit", annotation_position="top")

            fig.update_layout(
                title="Évolution météo et zones de stress pendant le cycle cultural",
                xaxis_title="Date", yaxis_title="Température (°C)",
                yaxis2=dict(title="Précipitations (mm)", overlaying='y', side='right', showgrid=False),
                height=520, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 Données météo journalières détaillées"):
                show_cols = ['time', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum',
                             'stress_chaleur', 'stress_critique', 'stress_gel', 'stress_secheresse']
                st.dataframe(df_w[show_cols].rename(columns={
                    'time': 'Date', 'temperature_2m_max': 'T° Max', 'temperature_2m_min': 'T° Min',
                    'precipitation_sum': 'Précip. (mm)', 'stress_chaleur': 'Stress chaleur',
                    'stress_critique': 'Stress critique', 'stress_gel': 'Gel', 'stress_secheresse': 'Sécheresse'
                }), use_container_width=True)

            with st.expander("🔍 Comment interpréter ce graphique ?"):
                st.markdown(f"""
                - La courbe **rouge** = température maximale du jour ; la courbe **bleue** = température minimale.
                - Les zones **rouges légères** marquent les jours où la chaleur a dépassé le seuil critique pour le **{culture}**
                  (risque d'échaudage / blocage de la photosynthèse).
                - Les zones **orange** marquent une **séquence de sécheresse** (≥ 7 jours sans pluie utile).
                - La ligne **verte pointillée** indique votre date d'application produit : regardez si elle tombe
                  juste avant ou pendant une période de stress, ce qui peut influencer l'efficacité du traitement.
                """)

# ══════════════════════════════════════════════════════════════════════════
# 9. EXPORT RAPPORT
# ══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("📤 Export des résultats")

report_lines = [
    f"# Rapport Bio-Expert 360 — {datetime.now().strftime('%d/%m/%Y %H:%M')}",
    f"**Culture** : {culture}  |  **Semis** : {d_semis}  |  **Application** : {d_appli}  |  **Récolte** : {d_recolt}",
    "",
    "## Résultats principaux",
    f"- N Produit : {n_p}  |  N Témoin : {n_t}",
    f"- Gain moyen : +{gain:.2f} qtx/ha",
    f"- Marge nette : {marge:.0f} €/ha",
    "",
    "## Statistiques",
    f"- Test : {stat_res['name']}  |  p = {stat_res['p']:.4f}  |  {'Significatif' if sig else 'Non significatif'} (α = {alpha_v})",
    f"- Cohen's d : {stat_res['d']:.3f} ({stat_res['label']})",
]
report_text = "\n".join(report_lines)
st.download_button(
    "⬇️ Télécharger le rapport (.md)",
    report_text.encode("utf-8"),
    file_name=f"bio_expert_360_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
    mime="text/markdown"
)

with st.expander("📋 Données filtrées (après nettoyage IQR)"):
    st.dataframe(df_final.reset_index(drop=True), use_container_width=True)
    csv = df_final.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Exporter en CSV", csv, file_name="bio_expert_donnees_filtrees.csv", mime="text/csv")
