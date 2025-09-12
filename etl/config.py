import os
from pathlib import Path

class Config:
    # Apps
    app_ids_file = os.getenv("APP_IDS_FILE")
    app_ids = []
    if app_ids_file and Path(app_ids_file).exists():
        app_ids = [l.strip() for l in Path(app_ids_file).read_text().splitlines() if l.strip()]
    else:
        app_ids = [s.strip() for s in os.getenv("APP_IDS", "").split(",") if s.strip()]

    # Bronze → GCS
    bronze_mode = os.getenv("BRONZE_MODE", "gcs")
    gcs_bucket  = os.getenv("GCS_BUCKET")

    # Firestore
    firestore_project = os.getenv("FIRESTORE_PROJECT")
