
# Projet Steam — Starter (Local Mongo + Airflow + Conda)

## 1) Conda (optionnel, pour lancer les scripts en dehors d'Airflow)
```bash
conda create -n steam_etl python=3.10 -y
conda activate steam_etl
pip install -r requirements.txt
cp .env.example .env  # édite APP_IDS, MONGO_URI
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords')"
```

## 2) Docker — tout-en-un (Mongo + Airflow)
```bash
docker compose up -d
# Mongo: mongodb://localhost:27017/ (DB: steamdb)
# UI Mongo Express: http://localhost:8081 (admin/admin)
# Airflow UI: http://localhost:8080 (admin/admin)
```

> Les tâches Airflow lisent /opt/airflow/.env (monté depuis ./.env). Mets-y:
> MONGO_URI_DOCKER=mongodb://mongo:27017/ pour que les tasks parlent au conteneur Mongo.

## 3) Run manuel sans Airflow (optionnel)
```bash
python run_pipeline.py --mode full   # initial
python run_pipeline.py --mode incr   # mises à jour
```

## 4) Dashboard (extrait)
```python
import os, streamlit as st, pymongo, pandas as pd

@st.cache_resource(show_spinner=False)
def get_db():
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    dbname = os.getenv("MONGO_DB", "steamdb")
    return pymongo.MongoClient(uri)[dbname]

db = get_db()
avis = pd.DataFrame(list(db['reviews_clean'].find({}, {'_id': 0})))
co_cnt = pd.DataFrame(list(db['cooccurrences_counts'].find({}, {'_id': 0})))
co_pct = pd.DataFrame(list(db['cooccurrences_percent'].find({}, {'_id': 0})))
```

## 5) Collections générées
- RAW: `reviews_{app_id}`
- SILVER: `reviews_clean`
- GOLD: `cooccurrences_counts`, `cooccurrences_percent`
