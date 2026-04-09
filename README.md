# MLOPS RETAIL BANK
Dans le secteur bancaire de détail, les prêts personnels constituent une source de revenus importante, mais comportent un risque inhérent : le défaut de paiement.

Ce projet vise à mettre en place un pipeline MLOps complet permettant de prédire la probabilité de défaut de chaque client et d’optimiser la gestion des risques financiers.

## Objectifs : 
- Prédire la probabilité de défaut de paiement des clients
- Améliorer les stratégies d'évaluation des risques
- Optimiser l'allocation du capital
- Déployer un modèle d'apprentissage automatique opérationnel

## MLOps Pipeline : 
- Collecte des données
- Prétraitement des données
- Ingénierie des caractéristiques
- Entraînement du modèle
- Suivi des expériences (MLflow)
- Sélection du modèle
- Déploiement (API/application)
- Pipeline CI/CD

## Tech Stack :
| Category        | Tools               |
| --------------- | ------------------- |
| Language        | Python              |
| Data            | Pandas, NumPy       |
| ML              | Scikit-learn        |
| Experimentation | MLflow              |
| App             | Streamlit / Flask   |
| DevOps          | Git, GitHub Actions |
| Cloud           | AWS / GCP / Azure   |

## Models Evaluated
- Régression logistique
- Arbre de décision
- Forêt aléatoire
Le meilleur modèle est sélectionné à l'aide de métriques de performance.

## MLflow
MLflow est utilisé pour suivre les expériences. Chaque modèle possède sa propre expérience et chaque test est enregistré comme une exécution.

## Déploiement sur AWS

L'application Streamlit est déployée sur **AWS ECS Fargate** via un pipeline CI/CD automatisé.

### Architecture de déploiement
| Service | Rôle |
|---|---|
| GitHub Actions | Déclenche le pipeline à chaque push |
| AWS CodeBuild | Build l'image Docker automatiquement |
| Amazon ECR | Stockage privé de l'image Docker |
| Amazon ECS Fargate | Exécution serverless du container |

### Accès à l'application
L'application est accessible à l'adresse :
http://51.44.167.27:8501

### Pipeline CI/CD
A chaque push sur `feature/cicd` ou `main` :
1. Les tests automatiques sont lancés (pytest)
2. GitHub Actions déclenche AWS CodeBuild
3. L'image Docker est rebuildée et pushée sur ECR
4. ECS Fargate met à jour l'application automatiquement

## Structure du projet
MLOPS_RETAIL_BANK/
├── app/                  # Application Streamlit
├── cicd/                 # Dockerfile
├── data/                 # Données
├── models/               # Modèles sauvegardés
├── notebooks/            # Notebooks d'exploration
├── tests/                # Tests automatisés
├── .github/workflows/    # Pipeline CI/CD GitHub Actions
├── buildspec.yml         # Instructions AWS CodeBuild
└── requirements.txt      # Dépendances Python


## CI/CD
Le processus est automatisé et chaque impulsion peut déclencher des tests et un déploiement.
