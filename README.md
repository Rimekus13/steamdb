# 🎮 Steam Reviews Data Pipeline

## 📌 Description
Ce projet met en place un **pipeline de données complet** permettant de collecter, transformer, stocker et visualiser les avis Steam.  
L’objectif est d’automatiser l’ingestion et l’analyse de millions d’avis joueurs afin de produire des **insights exploitables** pour les développeurs et les équipes marketing.

---

## 🏗️ Architecture
- **Collecte** : API Steam → stockage brut JSON dans **Google Cloud Storage (Bronze)**.
- **Transformation** : nettoyage, NLP (analyse de sentiment, détection de langue), enrichissement → stockage dans **Firestore (Silver/Gold)**.
- **Orchestration** : **Airflow** (VM GCP) exécute les DAGs de collecte et transformation (batch quotidien).
- **Visualisation** : **Streamlit** affiche les résultats depuis Firestore, avec message *« Données en cours de chargement »* si pipeline KO.
- **CI/CD** : **GitHub Actions** → déploiement automatique sur la VM (via `docker compose`).

---

## 📂 Arborescence du projet

```plaintext
steamdb/
├── dags/                      # DAGs Airflow (collecte, transformation, chargement Firestore)
│   ├── etl_entrypoint.py      # Entrée principale Airflow
│   └── example_dag.py         # Exemple DAG (collecte + transformation)
│
├── etl/                       # Scripts ETL (modules Python)
│   ├── __init__.py
│   ├── bronze_extract.py      # Extraction des avis depuis l’API Steam (Bronze → GCS)
│   ├── config.py              # Variables de configuration (clés, chemins, app_ids…)
│   ├── firestore_utils.py     # Fonctions d’insertion et lecture Firestore
│   ├── gcp_clients.py         # Connexions aux services GCP (GCS, Firestore)
│   ├── gcs_utils.py           # Gestion du stockage Bronze (GCS)
│   ├── gold_build.py          # Construction de la couche Gold (agrégats, indicateurs)
│   ├── http_utils.py          # Fonctions HTTP (requêtes API Steam, pagination, retry)
│   ├── mongo_utils.py         # Ancien module MongoDB (peut être déprécié si Firestore only)
│   ├── silver_clean.py        # Nettoyage et enrichissement des avis (Silver)
│   ├── state.py               # Gestion de l’état (dernier run, checkpoints)
│   └── text_utils.py          # Fonctions NLP (nettoyage texte, sentiment, langue)
│
├── dashboard/                 # Application Streamlit
│   ├── app.py                 # Dashboard principal
│   ├── analysis.py            # Fonctions d’analyse et de visualisation
│   └── requirements.txt       # Dépendances spécifiques au dashboard
│
├── logs/                      # Logs persistants (Airflow, ETL)
│
├── docker-compose.yml         # Orchestration containers (Airflow + Streamlit)
├── Dockerfile                 # Image personnalisée si besoin
├── .env                       # Variables d’environnement (APP_IDS, clés GCP…)
├── requirements.txt           # Dépendances Python globales
├── README.md                  # Documentation projet
└── .github/
    └── workflows/
        └── deploy.yml         # Workflow GitHub Actions (CI/CD déploiement VM GCP)
