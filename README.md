# MLOPS_RETAIL_BANK

Projet MLOps de prédiction du défaut de paiement dans le contexte de la banque de détail. Le dépôt combine une phase d'exploration des données, une phase d'entraînement et de comparaison de modèles, puis une mise à disposition du meilleur modèle dans une application Streamlit orientée démonstration métier.

L'objectif principal est de transformer un cas d'usage data science en mini-produit exploitable : données préparées, modèle sérialisé, interface utilisateur, tests automatisés et première brique de CI.

---

## 1. Vue d'ensemble du projet

### Objectifs métier
- prédire la probabilité de défaut d'un client ;
- aider à la lecture du risque de crédit ;
- industrialiser un prototype de scoring dans une logique MLOps ;
- fournir une interface simple pour tester le modèle.

### Ce que contient réellement le dépôt
Le repository contient déjà les briques suivantes :
- un notebook d'exploration des données ;
- un notebook d'entraînement et de comparaison de modèles ;
- un script Python pour reconstruire et sauvegarder le modèle retenu ;
- une application Streamlit pour l'inférence ;
- un Dockerfile ;
- une pipeline GitHub Actions de test ;
- un jeu de tests `pytest` minimal.

### Modèle retenu
Le dépôt converge vers une **régression logistique** entraînée sur 4 variables dérivées / sélectionnées :
- `debt_to_income`
- `credit_lines_outstanding`
- `financial_burden`
- `total_debt_outstanding`

Les métriques affichées dans le code applicatif sont les suivantes :
- ROC-AUC test : **0.9981**
- PR-AUC test : **0.9924**
- Recall test : **0.9811**
- Precision test : **0.8919**
- F1-score test : **0.9344**
- Accuracy test : **0.9745**

> Ces métriques proviennent du notebook / des constantes applicatives présentes dans le dépôt. Elles doivent être considérées comme les métriques de référence du projet tant qu'aucun nouveau run MLflow ne les remplace.

---

## 2. Structure du repository 

```text
MLOPS_RETAIL_BANK/
├── .github/workflows/deploy.yml      # CI actuelle : installation + pytest
├── app/
│   ├── prepare_model.py              # reconstruit et sérialise le meilleur modèle
│   └── streamlit_app.py              # interface d'inférence Streamlit
├── cicd/
│   └── Dockerfile                    # conteneur de l'application
├── data/
│   ├── X_scaled.npy                  # features d'entraînement déjà préparées
│   └── y.npy                         # cible binaire défaut / non défaut
├── models/
│   └── artifacts/
│       └── logistic_regression_best_model.joblib
├── notebooks/
│   ├── data_exploration.ipynb        # EDA, anomalies, feature engineering
│   └── train_model.ipynb             # benchmark modèles, MLflow, sélection finale
├── tests/
│   └── test_app.py                   # tests de présence, format et prédiction
├── DEVOPS.md
├── README.md
└── requirements.txt
```

### Rôle des principaux composants
- **`notebooks/data_exploration.ipynb`** : exploration, visualisations, détection d'anomalies, création de variables dérivées, première sélection de variables.
- **`notebooks/train_model.ipynb`** : split train/test, validation croisée, recherche d'hyperparamètres, tracking MLflow, comparaison des modèles et sauvegarde des artefacts.
- **`app/prepare_model.py`** : script de reconstruction du pipeline scikit-learn et de génération des artefacts de déploiement.
- **`app/streamlit_app.py`** : application d'inférence avec saisie utilisateur, standardisation des variables et affichage du score de risque.
- **`cicd/Dockerfile`** : conteneur minimal permettant d'exécuter Streamlit sur le port 8501.
- **`.github/workflows/deploy.yml`** : première base de CI déclenchée au `push` sur `main`.

---