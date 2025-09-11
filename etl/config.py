# etl/config.py
import os, pathlib

def _read_app_ids_from_file(path: str) -> list[str]:
    p = pathlib.Path(path)
    if not p.exists():
        return []
    ids = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        ids.append(ln)
    return ids

class Config:
    # On garde apps.txt (à la racine du repo)
    apps_file = os.getenv("APPS_FILE", "apps.txt")
    app_ids = _read_app_ids_from_file(apps_file)
    if not app_ids:
        # fallback env si besoin
        app_ids = [s.strip() for s in os.getenv("APP_IDS", "3527290").split(",") if s.strip()]

    # Cloud
    bronze_mode = os.getenv("BRONZE_MODE", "gcs").lower()  # "gcs" ou "files" (local)
    gcs_bucket  = os.getenv("GCS_BUCKET", "steam-raw-data")
    gcp_project = os.getenv("GCP_PROJECT")

    # Compat local (si tu gardes Mongo pour dev)
    mongo_db = os.getenv("MONGO_DB", "steamdb")
    mongo_uri = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017/steamdb?authSource=admin")
    mongo_uri_docker = os.getenv("MONGO_URI_DOCKER", mongo_uri)
