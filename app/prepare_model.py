from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "models" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

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

MODEL_METADATA = {
    "best_model": "logistic_regression",
    "best_params": {"model__C": 10.0, "model__solver": "lbfgs"},
    "metrics_from_notebook": {
        "test_roc_auc": 0.9981,
        "test_pr_auc": 0.9924,
        "test_recall": 0.9811,
        "test_precision": 0.8919,
        "test_f1": 0.9344,
        "test_accuracy": 0.9745,
    },
    "feature_names": FEATURE_NAMES,
    "scaler_stats_used_for_inference": SCALER_STATS,
}


def main() -> None:
    x_path = DATA_DIR / "X_scaled.npy"
    y_path = DATA_DIR / "y.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError("Les fichiers data/X_scaled.npy et data/y.npy sont requis.")

    X = np.load(x_path)
    y = np.load(y_path)
    X_df = pd.DataFrame(X, columns=FEATURE_NAMES)
    y_series = pd.Series(y, name="default")

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

    model_path = ARTIFACTS_DIR / "logistic_regression_best_model.joblib"
    metadata_path = ARTIFACTS_DIR / "deployment_metadata.json"

    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps(MODEL_METADATA, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
