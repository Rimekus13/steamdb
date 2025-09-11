import os, pathlib

class Config:
    # APP_IDS depuis apps.txt si présent, sinon variable d'env, sinon fallback
    _apps_file = pathlib.Path("apps.txt")
    if _apps_file.exists():
        app_ids = [ln.strip() for ln in _apps_file.read_text().splitlines() if ln.strip()]
    else:
        app_ids = os.getenv("APP_IDS", "3527290").split(",")

    mongo_db = os.getenv("MONGO_DB", "steamdb")
    mongo_uri = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017/steamdb?authSource=admin")
    mongo_uri_docker = os.getenv("MONGO_URI_DOCKER", mongo_uri)

    # Tout en base → bronze en Mongo
    bronze_mode = os.getenv("BRONZE_MODE", "mongo").lower()  # "mongo" | "files"
