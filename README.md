# MLOPS RETAIL BANK
Dans le secteur bancaire de détail, les prêts personnels constituent une source de revenus importante, mais comportent un risque inhérent : le défaut de paiement.

Ce projet vise à mettre en place un pipeline MLOps complet permettant de prédire la probabilité de défaut de chaque client et d’optimiser la gestion des risques financiers.

Objectifs : 
- Prédire la probabilité de défaut de paiement des clients
- Améliorer les stratégies d'évaluation des risques
- Optimiser l'allocation du capital
- Déployer un modèle d'apprentissage automatique opérationnel

MLOps Pipeline : 
graph LR
A[Data Collection] --> B[Data Preprocessing]
B --> C[Feature Engineering]
C --> D[Model Training]
D --> E[Experiment Tracking MLflow]
E --> F[Model Selection]
F --> G[Deployment API / App]
G --> H[CI/CD Pipeline]

