# DEVOPS.md

Ce document décrit l'état DevOps du projet **MLOPS_RETAIL_BANK**, ce qui est déjà présent dans le dépôt, et ce qu'il est pertinent de mettre en place pour faire évoluer ce prototype vers une application plus robuste et plus facilement déployable.

---

## 1. État actuel du dépôt

### Ce qui existe déjà
Le repository contient déjà plusieurs briques utiles d'un point de vue DevOps :
- un dépôt Git structuré ;
- un fichier `requirements.txt` ;
- un `Dockerfile` dans `cicd/` ;
- une pipeline GitHub Actions ;
- des tests `pytest` ;
- des notebooks pour la traçabilité de l'entraînement ;
- des artefacts de modèle dans `models/artifacts/`.

### Ce que fait actuellement la CI GitHub Actions
Le workflow `.github/workflows/deploy.yml` se déclenche sur :
- `push` sur la branche `main`

Puis exécute les étapes suivantes :
1. `actions/checkout@v3`
2. `actions/setup-python@v4` avec Python 3.10
3. `pip install -r requirements.txt`
4. `python -m pytest tests/ -v`

### Ce que cela signifie concrètement
Le dépôt dispose aujourd'hui surtout d'une **CI de validation basique**, pas encore d'une vraie chaîne CI/CD complète. En l'état :
- les dépendances sont installées ;
- les tests sont lancés ;
- aucun artefact Docker n'est publié ;
- aucun déploiement automatique n'est réalisé ;
- aucun environnement cible n'est configuré ;
- aucun secret n'est référencé dans le workflow.

---

## 2. Standards de travail recommandés

### Version de Python et dépendances
Pour conserver un comportement stable, il est recommandé de standardiser l'environnement sur :
- **Python 3.10**
- **scikit-learn 1.7.2**

Ce point est particulièrement important car les objets `joblib` de scikit-learn peuvent devenir incompatibles si le modèle est sérialisé avec une version puis relu avec une autre.

### Workflow développeur recommandé
1. créer une branche dédiée ;
2. installer les dépendances localement ;
3. exécuter `python app/prepare_model.py` si les artefacts doivent être régénérés ;
4. lancer `pytest tests/ -v` ;
5. vérifier l'application via `streamlit run app/streamlit_app.py` ;
6. ouvrir une Pull Request vers `main`.

### Convention de branches suggérée
- `main` : branche stable / démonstration ;
- `develop` : intégration continue si tu veux industrialiser davantage ;
- `feature/*` : nouvelles fonctionnalités ;
- `fix/*` : corrections ciblées ;
- `docs/*` : documentation.

### Recommandations qualité
À court terme, il serait utile d'ajouter :
- `black` pour le formatage ;
- `ruff` ou `flake8` pour le linting ;
- éventuellement `mypy` pour un contrôle de typage plus strict ;
- une couverture de tests via `pytest-cov`.

---

## 3. Conteneurisation et exécution applicative

### Dockerfile actuel
Le `Dockerfile` du projet est volontairement simple :
- image de base `python:3.10-slim` ;
- copie du `requirements.txt` ;
- installation des dépendances ;
- copie du code ;
- exposition du port `8501` ;
- lancement de Streamlit.

### Commandes utiles
```bash
docker build -f cicd/Dockerfile -t mlops-retail-bank .
docker run --rm -p 8501:8501 mlops-retail-bank
```

### Améliorations possibles sur l'image
Pour une industrialisation plus propre, on peut ajouter :
- un utilisateur non-root ;
- une meilleure gestion du cache des couches ;
- un `.dockerignore` ;
- des variables d'environnement pour la configuration ;
- un healthcheck ;
- un tag d'image versionné par commit SHA.

### Cible de déploiement envisageable
Le projet peut être déployé sur plusieurs cibles simples :
- **Streamlit Community Cloud** pour une démo ;
- **Azure Web App / Container Apps** ;
- **AWS App Runner / ECS Fargate** ;
- **Google Cloud Run**.

Pour un projet de portefeuille ou de soutenance, **Cloud Run** ou **Azure Container Apps** sont souvent un bon compromis entre simplicité et lisibilité DevOps.

---