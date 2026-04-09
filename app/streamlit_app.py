from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# =========================
# Configuration
# =========================
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "models" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "logistic_regression_best_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "deployment_metadata.json"

FEATURE_NAMES = [
    "debt_to_income",
    "credit_lines_outstanding",
    "financial_burden",
    "total_debt_outstanding",
]

SCALER_STATS = {
    "debt_to_income": {"mean": 0.124018, "std": 0.08279785979649233},
    "credit_lines_outstanding": {"mean": 1.4612, "std": 1.7437588055200834},
    "financial_burden": {"mean": 6277.013322, "std": 8397.731181946785},
    "total_debt_outstanding": {"mean": 8718.916797, "std": 6626.83339547753},
}

MODEL_NOTEBOOK_METRICS = {
    "model_name": "logistic_regression",
    "selection_reason": "meilleur ROC-AUC test, puis meilleur PR-AUC sur le notebook d'entraînement",
    "best_params": {"model__C": 10.0, "model__solver": "lbfgs"},
    "test_metrics": {
        "roc_auc": 0.9981,
        "pr_auc": 0.9924,
        "recall": 0.9811,
        "precision": 0.8919,
        "f1": 0.9344,
        "accuracy": 0.9745,
    },
    "competitors": {
        "random_forest": {"roc_auc": 0.9977, "pr_auc": 0.9908, "recall": 0.9568},
        "decision_tree": {"roc_auc": 0.9974, "pr_auc": 0.9880, "recall": 0.9757},
    },
}


# =========================
# CSS personnalisé — Design bancaire premium
# =========================
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* Reset global */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Fond général */
    .stApp {
        background: linear-gradient(160deg, #0a1628 0%, #0d2045 40%, #0a1628 100%);
        color: #e8edf5;
    }

    /* ============================
       BANNIÈRE BANCAIRE EN-TÊTE
    ============================ */
    .bank-banner {
        background: linear-gradient(135deg, #0d2045 0%, #1a3a6b 50%, #0d2045 100%);
        border-bottom: 2px solid #c9a84c;
        padding: 0;
        margin: -1rem -1rem 2rem -1rem;
        position: relative;
        overflow: hidden;
    }
    .bank-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(201,168,76,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .bank-banner-inner {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.2rem 2rem;
        position: relative;
        z-index: 1;
    }
    .bank-logo-section {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .bank-logo-icon {
        width: 52px;
        height: 52px;
        background: linear-gradient(135deg, #c9a84c, #f0d080);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.6rem;
        box-shadow: 0 4px 15px rgba(201,168,76,0.35);
        flex-shrink: 0;
    }
    .bank-name {
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 0.5px;
        line-height: 1.1;
    }
    .bank-tagline {
        font-size: 0.72rem;
        color: #c9a84c;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 500;
        margin-top: 2px;
    }
    .bank-badge {
        background: rgba(201,168,76,0.15);
        border: 1px solid rgba(201,168,76,0.4);
        border-radius: 20px;
        padding: 0.4rem 1rem;
        font-size: 0.75rem;
        color: #c9a84c;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-weight: 600;
    }

    /* ============================
       SIDEBAR
    ============================ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1e3d 0%, #0d2045 100%) !important;
        border-right: 1px solid rgba(201,168,76,0.2);
        min-width: 320px !important;
        max-width: 320px !important;
    }
    [data-testid="stSidebar"] * {
        color: #cdd8ec !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f0d080 !important;
        font-family: 'Playfair Display', serif !important;
    }

    /* ============================
       CARTES / CONTENEURS
    ============================ */
    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(201,168,76,0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    .card-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.05rem;
        color: #c9a84c;
        font-weight: 600;
        margin-bottom: 1rem;
        letter-spacing: 0.3px;
    }

    /* ============================
       FORMULAIRE & INPUTS
    ============================ */
    .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(201,168,76,0.25) !important;
        border-radius: 10px !important;
        color: #e8edf5 !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .stNumberInput > div > div > input:focus {
        border-color: #c9a84c !important;
        box-shadow: 0 0 0 2px rgba(201,168,76,0.15) !important;
    }
    label, .stNumberInput label {
        color: #9fb3d1 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.3px;
    }

    /* ============================
       BOUTON SIMULATION
    ============================ */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #c9a84c 0%, #f0d080 50%, #c9a84c 100%) !important;
        color: #0a1628 !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        padding: 0.75rem !important;
        box-shadow: 0 4px 20px rgba(201,168,76,0.35) !important;
        transition: all 0.3s ease !important;
    }
    .stFormSubmitButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(201,168,76,0.5) !important;
    }

    /* ============================
       MÉTRIQUES
    ============================ */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(201,168,76,0.2);
        border-radius: 14px;
        padding: 1rem 1.2rem;
    }
    [data-testid="stMetricLabel"] {
        color: #9fb3d1 !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    [data-testid="stMetricValue"] {
        color: #f0d080 !important;
        font-family: 'Playfair Display', serif !important;
        font-size: 1.6rem !important;
    }

    /* ============================
       JAUGE VISUELLE
    ============================ */
    .gauge-container {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(201,168,76,0.2);
        border-radius: 20px;
        padding: 2rem 1.5rem 1.5rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .gauge-title {
        font-family: 'Playfair Display', serif;
        font-size: 1rem;
        color: #c9a84c;
        margin-bottom: 1.5rem;
        letter-spacing: 0.5px;
    }
    .gauge-svg-wrap {
        display: flex;
        justify-content: center;
        margin-bottom: 0.5rem;
    }
    .gauge-value-label {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    .gauge-risk-label {
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }
    .gauge-scale {
        display: flex;
        justify-content: space-between;
        margin-top: 1rem;
        padding: 0 0.5rem;
    }
    .gauge-scale span {
        font-size: 0.68rem;
        color: #9fb3d1;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* ============================
       RÉSULTAT DÉFAUT / OK
    ============================ */
    .result-defaut {
        background: linear-gradient(135deg, rgba(220,53,69,0.15), rgba(220,53,69,0.05));
        border: 1px solid rgba(220,53,69,0.5);
        border-left: 4px solid #dc3545;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .result-ok {
        background: linear-gradient(135deg, rgba(40,167,69,0.15), rgba(40,167,69,0.05));
        border: 1px solid rgba(40,167,69,0.5);
        border-left: 4px solid #28a745;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .result-icon { font-size: 1.8rem; }
    .result-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0.3rem 0 0.2rem;
    }
    .result-subtitle { font-size: 0.82rem; color: #9fb3d1; }

    /* ============================
       COMPARAISON MODÈLES
    ============================ */
    .model-compare-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .model-compare-row:last-child { border-bottom: none; }
    .model-name { font-size: 0.82rem; color: #9fb3d1; }
    .model-name.best { color: #f0d080; font-weight: 600; }
    .model-bar-wrap {
        flex: 1;
        margin: 0 1rem;
        background: rgba(255,255,255,0.07);
        border-radius: 100px;
        height: 8px;
        overflow: hidden;
    }
    .model-bar {
        height: 100%;
        border-radius: 100px;
        background: linear-gradient(90deg, #c9a84c, #f0d080);
    }
    .model-bar.competitor {
        background: linear-gradient(90deg, #4a6fa5, #6b9bd2);
    }
    .model-score {
        font-size: 0.8rem;
        font-weight: 600;
        color: #e8edf5;
        min-width: 42px;
        text-align: right;
    }

    /* ============================
       SLIDER
    ============================ */
    .stSlider > div > div > div {
        background: rgba(201,168,76,0.3) !important;
    }
    .stSlider > div > div > div > div {
        background: #c9a84c !important;
    }

    /* Titres principaux */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        color: #e8edf5 !important;
    }
    h2 { color: #c9a84c !important; }

    /* Divider */
    hr { border-color: rgba(201,168,76,0.2) !important; }

    /* Info box */
    .stInfo {
        background: rgba(79,131,196,0.12) !important;
        border: 1px solid rgba(79,131,196,0.3) !important;
        border-radius: 12px !important;
        color: #cdd8ec !important;
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.04) !important;
        border-radius: 10px !important;
        color: #9fb3d1 !important;
    }

    /* Success / Error */
    .stSuccess {
        background: rgba(40,167,69,0.12) !important;
        border: 1px solid rgba(40,167,69,0.35) !important;
        border-radius: 12px !important;
    }
    .stError {
        background: rgba(220,53,69,0.12) !important;
        border: 1px solid rgba(220,53,69,0.35) !important;
        border-radius: 12px !important;
    }

    /* Caption */
    .stCaption { color: #6b83a6 !important; }

    /* Masquer le menu hamburger et footer Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# =========================
# Composant : Bannière
# =========================
def render_banner():
    st.markdown("""
    <div class="bank-banner">
        <div class="bank-banner-inner">
            <div class="bank-logo-section">
                <div class="bank-logo-icon">🏛️</div>
                <div>
                    <div class="bank-name">RetailBank Analytics</div>
                    <div class="bank-tagline">Credit Risk Intelligence Platform</div>
                </div>
            </div>
            <div class="bank-badge">⚡ IA · Prédiction Risque</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================
# Composant : Jauge SVG
# =========================
def render_gauge(probability: float, risk_label: str):
    pct = probability * 100

    if pct < 20:
        color = "#28a745"
    elif pct < 50:
        color = "#ffc107"
    elif pct < 75:
        color = "#fd7e14"
    else:
        color = "#dc3545"

    bar_width = int(pct)

    st.markdown(f"""
    <div class="gauge-container">
        <div class="gauge-title">📊 Probabilité de Défaut</div>
        <div style="margin: 1.5rem 0 0.5rem;">
            <div style="
                background: rgba(255,255,255,0.08);
                border-radius: 100px;
                height: 20px;
                overflow: hidden;
            ">
                <div style="
                    width: {bar_width}%;
                    height: 100%;
                    background: linear-gradient(90deg, {color}, {color}cc);
                    border-radius: 100px;
                    transition: width 0.5s ease;
                "></div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:0.4rem;">
                <span style="font-size:0.68rem;color:#6b83a6">0%</span>
                <span style="font-size:0.68rem;color:#6b83a6">50%</span>
                <span style="font-size:0.68rem;color:#6b83a6">100%</span>
            </div>
        </div>
        <div class="gauge-value-label" style="color:{color}">{pct:.1f}%</div>
        <div class="gauge-risk-label" style="color:{color}">Risque {risk_label}</div>
    </div>
    """, unsafe_allow_html=True)


# =========================
# Composant : Résultat
# =========================
def render_result_card(predicted_class: int, probability: float, threshold: float):
    if predicted_class == 1:
        st.markdown(f"""
        <div class="result-defaut">
            <div class="result-icon">🚨</div>
            <div class="result-title" style="color:#ff6b7a">Défaut probable détecté</div>
            <div class="result-subtitle">
                Probabilité de défaut : <strong style="color:#ff6b7a">{probability*100:.2f}%</strong>
                &nbsp;·&nbsp; Seuil appliqué : <strong>{threshold:.2f}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-ok">
            <div class="result-icon">✅</div>
            <div class="result-title" style="color:#51cf66">Client solvable</div>
            <div class="result-subtitle">
                Probabilité de défaut : <strong style="color:#51cf66">{probability*100:.2f}%</strong>
                &nbsp;·&nbsp; Seuil appliqué : <strong>{threshold:.2f}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =========================
# Composant : Comparaison modèles
# =========================
def render_model_comparison():
    metrics = MODEL_NOTEBOOK_METRICS["test_metrics"]
    competitors = MODEL_NOTEBOOK_METRICS["competitors"]

    best_roc = metrics["roc_auc"]
    rf_roc = competitors["random_forest"]["roc_auc"]
    dt_roc = competitors["decision_tree"]["roc_auc"]
    max_roc = max(best_roc, rf_roc, dt_roc)

    def bar_width(val):
        return f"{(val / max_roc) * 100:.1f}%"

    st.markdown(f"""
    <div style="margin-top:0.5rem">
        <div class="model-compare-row">
            <div class="model-name best">🥇 Régression Logistique</div>
            <div class="model-bar-wrap"><div class="model-bar" style="width:{bar_width(best_roc)}"></div></div>
            <div class="model-score">{best_roc:.4f}</div>
        </div>
        <div class="model-compare-row">
            <div class="model-name">Random Forest</div>
            <div class="model-bar-wrap"><div class="model-bar competitor" style="width:{bar_width(rf_roc)}"></div></div>
            <div class="model-score">{rf_roc:.4f}</div>
        </div>
        <div class="model-compare-row">
            <div class="model-name">Decision Tree</div>
            <div class="model-bar-wrap"><div class="model-bar competitor" style="width:{bar_width(dt_roc)}"></div></div>
            <div class="model-score">{dt_roc:.4f}</div>
        </div>
    </div>
    <div style="font-size:0.68rem;color:#6b83a6;text-align:right;margin-top:0.5rem">ROC-AUC (test set)</div>
    """, unsafe_allow_html=True)


# =========================
# Data / model loading
# =========================
@st.cache_data(show_spinner=False)
def load_training_arrays() -> tuple[pd.DataFrame, pd.Series]:
    x_path = DATA_DIR / "X_scaled.npy"
    y_path = DATA_DIR / "y.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "Fichiers de données introuvables. Vérifie la présence de data/X_scaled.npy et data/y.npy."
        )
    X = np.load(x_path)
    y = np.load(y_path)
    X_df = pd.DataFrame(X, columns=FEATURE_NAMES)
    y_series = pd.Series(y, name="default")
    return X_df, y_series


@st.cache_resource(show_spinner=False)
def load_or_train_model() -> Pipeline:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    X_df, y_series = load_training_arrays()
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LogisticRegression(
            C=10.0, solver="lbfgs", class_weight="balanced",
            max_iter=2000, random_state=42,
        )),
    ])
    model.fit(X_df, y_series)
    return model


# =========================
# Feature engineering
# =========================
def build_raw_feature_frame(
    credit_lines_outstanding: int,
    loan_amt_outstanding: float,
    total_debt_outstanding: float,
    income: float,
) -> pd.DataFrame:
    if income <= 0:
        raise ValueError("Le revenu doit être strictement positif.")
    if credit_lines_outstanding < 0:
        raise ValueError("Le nombre de lignes de crédit ne peut pas être négatif.")
    if loan_amt_outstanding < 0 or total_debt_outstanding < 0:
        raise ValueError("Les montants ne peuvent pas être négatifs.")

    debt_to_income = total_debt_outstanding / income
    financial_burden = credit_lines_outstanding * loan_amt_outstanding

    return pd.DataFrame([{
        "debt_to_income": debt_to_income,
        "credit_lines_outstanding": float(credit_lines_outstanding),
        "financial_burden": financial_burden,
        "total_debt_outstanding": total_debt_outstanding,
    }])


def standardize_features(raw_features: pd.DataFrame) -> pd.DataFrame:
    scaled = raw_features.copy()
    for col in FEATURE_NAMES:
        mean_ = SCALER_STATS[col]["mean"]
        std_ = SCALER_STATS[col]["std"]
        if std_ == 0:
            raise ValueError(f"Écart-type nul pour la variable {col}.")
        scaled[col] = (scaled[col] - mean_) / std_
    return scaled


def get_risk_label(probability: float) -> str:
    if probability < 0.20:
        return "Faible"
    if probability < 0.50:
        return "Modéré"
    if probability < 0.75:
        return "Élevé"
    return "Très élevé"


def format_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


# =========================
# APP PRINCIPALE
# =========================
st.set_page_config(
    page_title="RetailBank · Credit Risk",
    page_icon="https://cdn-icons-png.flaticon.com/512/2489/2489756.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()
render_banner()

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("## 🤖 Modèle déployé")
    st.success("✦ Régression Logistique sélectionnée")

    st.markdown("---")
    st.markdown("### 📊 Métriques test")
    metrics = MODEL_NOTEBOOK_METRICS["test_metrics"]
    cols = st.columns(2)
    cols[0].metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    cols[1].metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
    cols2 = st.columns(2)
    cols2[0].metric("Recall", f"{metrics['recall']:.4f}")
    cols2[1].metric("F1-score", f"{metrics['f1']:.4f}")
    cols3 = st.columns(2)
    cols3[0].metric("Precision", f"{metrics['precision']:.4f}")
    cols3[1].metric("Accuracy", f"{metrics['accuracy']:.4f}")

    st.markdown("---")
    st.markdown("### ⚖️ Comparaison des modèles")
    render_model_comparison()

    st.markdown("---")
    st.markdown("### 🎚️ Seuil de décision")
    threshold = st.slider(
        "Probabilité de coupure",
        min_value=0.10, max_value=0.90,
        value=0.50, step=0.01,
        help="Au-dessus de ce seuil → défaut probable.",
    )
    st.caption(f"Seuil actuel : **{threshold:.2f}** — Au-dessus = défaut classifié")

# ---- CONTENU PRINCIPAL ----
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.markdown("## Simulation de risque client")
    st.caption(
        "Renseignez les données brutes du client."
    )

    with st.form("simulation_form"):
        st.markdown('<div class="card-title">👤 Données financières du client</div>', unsafe_allow_html=True)

        credit_lines_outstanding = st.number_input(
            "Nombre de crédit en cours",
            min_value=0, max_value=20, value=1, step=1,
            help="Nombre de lignes de crédit ouvertes.",
        )
        loan_amt_outstanding = st.number_input(
            "Montant du prêt restant dû (€)",
            min_value=0.0, value=4052.38, step=100.0, format="%.2f",
            help="Montant du prêt personnel restant dû.",
        )
        total_debt_outstanding = st.number_input(
            "Dette totale restante (€)",
            min_value=0.0, value=6732.41, step=100.0, format="%.2f",
            help="Dette totale restante due par le client.",
        )
        income = st.number_input(
            "Revenu annuel (€)",
            min_value=1.0, value=70039.90, step=500.0, format="%.2f",
            help="Revenu annuel déclaré du client.",
        )

        submitted = st.form_submit_button("🔍 Lancer la simulation", use_container_width=True)

with col_right:
    st.markdown("## Résultat de l'analyse")

    if not submitted:
        st.markdown("""
        <div style="
            background: rgba(255,255,255,0.03);
            border: 1px dashed rgba(201,168,76,0.2);
            border-radius: 20px;
            padding: 3rem 2rem;
            text-align: center;
            color: #6b83a6;
            margin-top: 1rem;
        ">
            <div style="font-size:2.5rem;margin-bottom:1rem">📋</div>
            <div style="font-family:'Playfair Display',serif;font-size:1rem;color:#9fb3d1">
                Renseignez les données client<br>et lancez la simulation
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        try:
            raw_features = build_raw_feature_frame(
                credit_lines_outstanding=int(credit_lines_outstanding),
                loan_amt_outstanding=float(loan_amt_outstanding),
                total_debt_outstanding=float(total_debt_outstanding),
                income=float(income),
            )
            scaled_features = standardize_features(raw_features)
            model = load_or_train_model()

            default_probability = float(model.predict_proba(scaled_features)[0, 1])
            predicted_class = int(default_probability >= threshold)
            risk_label = get_risk_label(default_probability)

            # Jauge visuelle
            render_gauge(default_probability, risk_label)

            # Carte résultat
            render_result_card(predicted_class, default_probability, threshold)

            # Métriques rapides
            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Probabilité", format_pct(default_probability))
            m2.metric("Classe", "Défaut 🔴" if predicted_class == 1 else "Sain 🟢")
            m3.metric("Niveau risque", risk_label)

            # Variables calculées
            with st.expander("📐 Variables calculées & standardisées"):
                st.markdown("**Variables brutes dérivées**")
                engineered_display = raw_features.rename(columns={
                    "debt_to_income": "Ratio Dette/Revenu",
                    "credit_lines_outstanding": "Lignes de crédit",
                    "financial_burden": "Charge financière (€)",
                    "total_debt_outstanding": "Dette totale (€)",
                }).T
                engineered_display.columns = ["Valeur"]
                st.dataframe(engineered_display, use_container_width=True)

        except Exception as exc:
            st.exception(exc)
