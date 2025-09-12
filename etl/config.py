# etl/config.py
import os
from pathlib import Path

class Config:
    app_ids_file = os.getenv("APP_IDS_FILE")
    if app_ids_file and Path(app_ids_file).exists():
        app_ids = [l.strip() for l in Path(app_ids_file).read_text().splitlines() if l.strip()]
    else:
        app_ids = [s.strip() for s in os.getenv("APP_IDS", "").split(",") if s.strip()]

    bronze_mode = os.getenv("BRONZE_MODE", "gcs")
    gcs_bucket  = os.getenv("GCS_BUCKET")
    gcp_project = os.getenv("GCP_PROJECT")
    firestore_project = os.getenv("FIRESTORE_PROJECT") or gcp_project
