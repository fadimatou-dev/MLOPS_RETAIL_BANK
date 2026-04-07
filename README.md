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

## 3. Pipeline fonctionnel et logique ML 

### Chaîne de traitement actuelle
1. **Exploration** des données et analyse métier dans `data_exploration.ipynb`.
2. **Feature engineering** avec création notamment de `debt_to_income` et `financial_burden`.
3. **Sélection / comparaison** de plusieurs modèles dans `train_model.ipynb`.
4. **Choix du meilleur modèle** selon `test_roc_auc`, puis `test_pr_auc`, puis `test_recall`.
5. **Sérialisation** du pipeline final en `.joblib`.
6. **Chargement dans Streamlit** pour permettre une simulation utilisateur.

### Modèles comparés dans le notebook
- régression logistique ;
- arbre de décision ;
- random forest.

### Particularités d'implémentation à connaître
- Le modèle final est encapsulé dans un `Pipeline` scikit-learn avec :
  - `SimpleImputer(strategy="median")`
  - `LogisticRegression(C=10.0, solver="lbfgs", class_weight="balanced", max_iter=2000, random_state=42)`
- L'application Streamlit **ne demande pas directement les 4 variables finales** ; elle calcule certaines features à partir d'entrées métier :
  - `debt_to_income = total_debt_outstanding / income`
  - `financial_burden = credit_lines_outstanding * loan_amt_outstanding`
- Les statistiques de standardisation sont **codées en dur** dans l'application et dans `prepare_model.py`.

### Artefacts attendus
Le fonctionnement nominal du projet repose sur :
- `data/X_scaled.npy`
- `data/y.npy`
- `models/artifacts/logistic_regression_best_model.joblib`
- éventuellement `models/artifacts/deployment_metadata.json` après exécution de `prepare_model.py`

---

## 4. Installation locale et démarrage 

### Prérequis
- Python **3.10** recommandé ;
- pip ;
- Git ;
- Docker en option.

### Installation
```bash
git clone <url-du-repo>
cd MLOPS_RETAIL_BANK
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# ou
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### Reconstruire les artefacts modèle
```bash
python app/prepare_model.py
```

Cette commande :
- recharge `X_scaled.npy` et `y.npy` ;
- réentraîne la régression logistique retenue ;
- sauvegarde le modèle dans `models/artifacts/` ;
- génère un fichier `deployment_metadata.json`.

### Lancer l'application Streamlit
```bash
streamlit run app/streamlit_app.py
```

L'application permet de saisir :
- le nombre de lignes de crédit ;
- le montant du prêt restant dû ;
- la dette totale ;
- le revenu annuel.

Puis elle :
- construit les variables dérivées ;
- les standardise ;
- calcule la probabilité de défaut ;
- affiche une classe prédite selon un seuil ajustable.

---

## 5. Tests et vérifications

### Lancer les tests
```bash
pytest tests/ -v
```

### Ce que couvrent les tests actuels
- présence des fichiers de données ;
- cohérence dimensionnelle de `X` et `y` ;
- présence du modèle sérialisé ;
- capacité du modèle à produire 5 prédictions ;
- disponibilité des imports principaux.

### Points d'attention identifiés pendant l'exploration
- Le modèle sérialisé est sensible à la **version de `scikit-learn`**. Une incompatibilité de version peut casser le chargement du pipeline joblib.
- Le fichier `requirements.txt` fixe bien `scikit-learn==1.7.2` : il est préférable de conserver cet alignement entre entraînement, test et déploiement.
- La CI actuelle reste légère : elle ne couvre ni linting, ni build Docker, ni validation du notebook, ni tests d'intégration Streamlit.

---

## 6. Exécution avec Docker 

### Construire l'image
```bash
docker build -f cicd/Dockerfile -t mlops-retail-bank .
```

### Lancer le conteneur
```bash
docker run --rm -p 8501:8501 mlops-retail-bank
```

Le `Dockerfile` actuel :
- part de `python:3.10-slim` ;
- installe les dépendances du `requirements.txt` ;
- copie le dépôt complet dans l'image ;
- expose le port **8501** ;
- lance `streamlit run app/streamlit_app.py`.

---

## 7. Limites actuelles et prochaines améliorations

Le dépôt constitue une bonne base de démonstration MLOps, mais plusieurs axes d'industrialisation restent ouverts :
- externaliser la configuration et les seuils dans des variables d'environnement ;
- versionner plus proprement les datasets et les artefacts ;
- brancher un vrai serveur MLflow distant ;
- enrichir la CI avec lint, formatage, build Docker et tests d'intégration ;
- ajouter un vrai déploiement continu ;
- sécuriser la gestion des secrets et de la configuration ;
- documenter un cycle de vie modèle plus complet : retraining, registry, rollback, monitoring.

Pour la partie industrialisation, voir aussi **`DEVOPS.md`**.