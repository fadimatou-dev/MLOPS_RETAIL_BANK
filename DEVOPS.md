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

## 4. Gestion de la configuration, des secrets et des artefacts

### Configuration applicative
Aujourd'hui, plusieurs éléments sont codés en dur dans le dépôt :
- les statistiques de standardisation ;
- certains paramètres du modèle ;
- le seuil de décision par défaut côté interface.

Pour une version plus robuste, il est conseillé de séparer :
- la **configuration applicative** ;
- la **configuration d'environnement** ;
- les **artefacts de modèle**.

### Secrets
Même si le dépôt ne manipule pas encore de secrets sensibles, la bonne pratique est de préparer dès maintenant :
- un fichier `.env` non versionné ;
- des secrets GitHub Actions pour les futurs déploiements ;
- des variables de configuration pour l'URL MLflow, le stockage d'artefacts, ou les accès cloud.

### Gestion des artefacts ML
Le projet contient déjà un modèle sérialisé et prévoit aussi un fichier `deployment_metadata.json`. Pour aller plus loin, on recommande :
- un stockage des artefacts hors du repo si leur volume augmente ;
- un versioning clair des modèles ;
- un mapping entre version de code, version de données et version de modèle ;
- un registre de modèles si MLflow est industrialisé.

### Remarques de robustesse observées pendant l'exploration
- le workflow CI ne régénère pas le modèle avant les tests ;
- les tests supposent la présence locale d'artefacts et de données ;
- le `.gitignore` mérite une petite revue, car sa dernière ligne semble corrompue ;
- la documentation DevOps était initialement presque vide, ce fichier sert donc aussi de base de structuration pour la suite.

---

## 5. Cible CI/CD recommandée à moyen terme

### CI cible
Une CI plus complète pourrait exécuter les étapes suivantes à chaque Pull Request :
1. checkout du code ;
2. installation Python ;
3. cache pip ;
4. lint (`black --check`, `ruff`) ;
5. tests unitaires ;
6. tests d'intégration simples ;
7. build Docker ;
8. publication éventuelle d'un rapport de couverture.

### CD cible
Une CD simple mais crédible pourrait être :
1. merge dans `main` ;
2. build de l'image Docker ;
3. tag par SHA Git ;
4. push vers un registre ;
5. déploiement sur un environnement cible ;
6. smoke test post-déploiement ;
7. rollback automatique en cas d'échec.

### Exemple de séparation des environnements
- **dev** : expérimentation, tests visuels Streamlit ;
- **staging** : validation avant démonstration ;
- **prod** : environnement publié.

---

## 6. Observabilité, sécurité et exploitation

### Observabilité minimale recommandée
Pour passer du prototype à une application exploitable, il faut ajouter :
- des logs structurés ;
- des logs d'erreurs exploitables ;
- une journalisation des versions de modèle ;
- des métriques d'usage de l'application ;
- un suivi du drift de données et du drift de performance.

### Sécurité
Quelques mesures simples peuvent déjà être prévues :
- ne pas committer de secrets ;
- scanner les dépendances ;
- geler les versions critiques ;
- limiter les permissions GitHub Actions ;
- utiliser des images de base maintenues.

### MLOps / exploitation modèle
À terme, le cycle modèle idéal serait :
- entraînement reproductible ;
- tracking MLflow ;
- validation automatique ;
- promotion du modèle ;
- déploiement contrôlé ;
- monitoring ;
- retraining planifié si dérive.

---

## 7. Résumé opérationnel

En l'état, **MLOPS_RETAIL_BANK** est un bon **prototype de démonstration MLOps** :
- la chaîne de modélisation est visible ;
- l'inférence est exposée dans une interface ;
- le socle de tests et de CI existe ;
- la conteneurisation est amorcée.

Pour atteindre un niveau plus professionnel, la priorité DevOps est claire :
1. fiabiliser l'environnement ;
2. renforcer la CI ;
3. versionner proprement les artefacts ;
4. formaliser la CD ;
5. ajouter observabilité et configuration sécurisée.
