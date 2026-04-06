import pytest
import numpy as np
import joblib
import os

def test_data_files_exist():
    assert os.path.exists("data/X_scaled.npy"), "X_scaled.npy manquant"
    assert os.path.exists("data/y.npy"), "y.npy manquant"

def test_data_format():
    X = np.load("data/X_scaled.npy")
    y = np.load("data/y.npy")
    assert X.ndim == 2, "X doit être une matrice 2D"
    assert y.ndim == 1, "y doit être un vecteur 1D"
    assert len(X) == len(y), "X et y doivent avoir le même nombre de lignes"

def test_model_exists():
    assert os.path.exists("models/artifacts/logistic_regression_best_model.joblib"), "Modèle manquant"

def test_model_prediction():
    model = joblib.load("models/artifacts/logistic_regression_best_model.joblib")
    X = np.load("data/X_scaled.npy")
    predictions = model.predict(X[:5])
    assert len(predictions) == 5, "Le modèle doit retourner 5 prédictions"
    assert set(predictions).issubset({0, 1}), "Les prédictions doivent être 0 ou 1"

def test_imports():
    import streamlit
    import sklearn
    import pandas
    import numpy
    assert all([streamlit, sklearn, pandas, numpy])