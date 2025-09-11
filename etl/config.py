# etl/config.py
import os
from pathlib import Path

class Config:
    # --- Apps à traiter (fichier apps.txt à la racine du repo, 1 app_id par ligne)
    apps_file = os.getenv("APPS_FILE", "/opt/airflow/apps.txt")  # monté par docker-compose
    if os.path.exists(apps_file):
        with open(apps_file, "r", encoding="utf-8") as f:
            app_ids = [l.strip() for l in f if l.strip()]
    else:
        app_ids = []

    # --- Bronze: local vs GCS
    bronze_mode = os.getenv("BRONZE_MODE", "gcs")  # 'local' ou 'gcs'
    # Chemins/local
    data_root = Path(os.getenv("DATA_ROOT", "/opt/airflow/data"))
    # GCS
    gcp_project = os.getenv("GCP_PROJECT")  # peut être déduit par les creds de la VM
    gcs_bucket = os.getenv("GCS_BUCKET")    # ex: steam-raw-data-<id>
