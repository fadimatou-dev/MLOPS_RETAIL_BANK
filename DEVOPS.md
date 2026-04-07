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