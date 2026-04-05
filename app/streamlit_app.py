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

# Ces statistiques proviennent du notebook d'exploration déjà présent dans le repo.
# Le repo ne contient pas le scaler sauvegardé, donc on reconstruit la standardisation
# à partir des moyennes/écarts-types affichés dans les sorties du notebook.
# Les écarts-types ci-dessous sont rapprochés du std population (StandardScaler).
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

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LogisticRegression(
                    C=10.0,
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
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

    row = pd.DataFrame(
        [
            {
                "debt_to_income": debt_to_income,
                "credit_lines_outstanding": float(credit_lines_outstanding),
                "financial_burden": financial_burden,
                "total_debt_outstanding": total_debt_outstanding,
            }
        ]
    )
    return row



def standardize_features(raw_features: pd.DataFrame) -> pd.DataFrame:
    scaled = raw_features.copy()
    for col in FEATURE_NAMES:
        mean_ = SCALER_STATS[col]["mean"]
        std_ = SCALER_STATS[col]["std"]
        if std_ == 0:
            raise ValueError(f"Écart-type nul pour la variable {col}.")
        scaled[col] = (scaled[col] - mean_) / std_
    return scaled


# =========================
# UI helpers
# =========================
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
# Streamlit app
# =========================
st.set_page_config(
    page_title="Retail Bank - Default Risk Simulator",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Simulateur de risque de défaut bancaire")
st.caption(
    "Application Streamlit de déploiement du meilleur modèle identifié dans le notebook du projet MLOps."
)

with st.sidebar:
    st.header("Modèle déployé")
    st.success("Régression logistique sélectionnée")
    st.write("**Pourquoi ?**")
    st.write(
        "Le notebook `models/train_model.ipynb` classe la régression logistique en tête sur le jeu de test, "
        "devant Random Forest et Decision Tree."
    )

    st.write("**Métriques test retenues**")
    metrics = MODEL_NOTEBOOK_METRICS["test_metrics"]
    st.write(f"ROC-AUC : **{metrics['roc_auc']:.4f}**")
    st.write(f"PR-AUC : **{metrics['pr_auc']:.4f}**")
    st.write(f"Recall : **{metrics['recall']:.4f}**")
    st.write(f"Precision : **{metrics['precision']:.4f}**")
    st.write(f"F1-score : **{metrics['f1']:.4f}**")
    st.write(f"Accuracy : **{metrics['accuracy']:.4f}**")

    threshold = st.slider(
        "Seuil de décision",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.01,
        help="Au-dessus de ce seuil, la prédiction est classée comme défaut probable.",
    )

st.markdown(
    """
Cette interface attend **les variables métier brutes** utiles au modèle retenu :
- nombre de lignes de crédit en cours 
- montant du prêt 
- dette totale 
- revenu annuel

L'application recalcule ensuite les variables dérivées utilisées au modeling :
`debt_to_income` et `financial_burden`.
"""
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Entrées utilisateur")
    with st.form("simulation_form"):
        credit_lines_outstanding = st.number_input(
            "Credit lines outstanding",
            min_value=0,
            max_value=20,
            value=1,
            step=1,
            help="Nombre de lignes de crédit ouvertes.",
        )
        loan_amt_outstanding = st.number_input(
            "Loan amount outstanding",
            min_value=0.0,
            value=4052.38,
            step=100.0,
            format="%.2f",
            help="Montant du prêt personnel restant dû.",
        )
        total_debt_outstanding = st.number_input(
            "Total debt outstanding",
            min_value=0.0,
            value=6732.41,
            step=100.0,
            format="%.2f",
            help="Dette totale restante due par le client.",
        )
        income = st.number_input(
            "Annual income",
            min_value=1.0,
            value=70039.90,
            step=500.0,
            format="%.2f",
            help="Revenu annuel déclaré du client.",
        )

        submitted = st.form_submit_button("Lancer la simulation", use_container_width=True)

with col2:
    st.subheader("Aide à l'interprétation")
    st.info(
        "La sortie principale est une **probabilité de défaut**. La classe finale dépend du seuil choisi dans la barre latérale."
    )
    st.write(
        "Le modèle a été entraîné sur les 4 variables finales issues du notebook : "
        "`debt_to_income`, `credit_lines_outstanding`, `financial_burden`, `total_debt_outstanding`."
    )


if submitted:
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

        st.divider()
        st.subheader("Résultat de la simulation")

        m1, m2, m3 = st.columns(3)
        m1.metric("Probabilité de défaut", format_pct(default_probability))
        m2.metric("Classe prédite", "Défaut" if predicted_class == 1 else "Pas de défaut")
        m3.metric("Niveau de risque", risk_label)

        if predicted_class == 1:
            st.error(
                f"Le client est classé **à risque de défaut** avec un seuil de {threshold:.2f}."
            )
        else:
            st.success(
                f"Le client est classé **sans défaut probable** avec un seuil de {threshold:.2f}."
            )

        st.subheader("Variables calculées")
        engineered_display = raw_features.rename(
            columns={
                "debt_to_income": "Debt to income",
                "credit_lines_outstanding": "Credit lines outstanding",
                "financial_burden": "Financial burden",
                "total_debt_outstanding": "Total debt outstanding",
            }
        ).T
        engineered_display.columns = ["Valeur"]
        st.dataframe(engineered_display, use_container_width=True)

        with st.expander("Voir les variables standardisées envoyées au modèle"):
            st.dataframe(scaled_features, use_container_width=True, hide_index=True)

    except Exception as exc:
        st.exception(exc)


