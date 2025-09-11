
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()  # charge .env depuis le cwd

@dataclass(frozen=True)
class Config:
    app_ids = [a.strip() for a in os.getenv("APP_IDS", "").split(",") if a.strip()]

    # Mongo host pour scripts locaux (dashboard local)
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    mongo_db = os.getenv("MONGO_DB", "steamdb")

    # Si exécuté dans Airflow (docker), utilise MONGO_URI_DOCKER si présent
    mongo_uri_docker = os.getenv("MONGO_URI_DOCKER", os.getenv("MONGO_URI", "mongodb://localhost:27017/"))

    steam_num_per_page = int(os.getenv("STEAM_NUM_PER_PAGE", 100))
    steam_filter = os.getenv("STEAM_FILTER", "updated")
    steam_review_type = os.getenv("STEAM_REVIEW_TYPE", "all")
    steam_purchase_type = os.getenv("STEAM_PURCHASE_TYPE", "all")
    steam_language = os.getenv("STEAM_LANGUAGE", "all")
    steam_day_range = int(os.getenv("STEAM_DAY_RANGE", 365))

    api_sleep_seconds = float(os.getenv("API_SLEEP_SECONDS", 1.2))
