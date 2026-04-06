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

## Streamlit 
Le modèle est déployé sous forme d'application et les utilisateurs peuvent tester les prédictions en ligne.

## CI/CD
Le processus est automatisé et chaque impulsion peut déclencher des tests et un déploiement.
