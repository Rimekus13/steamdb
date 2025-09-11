# etl/http.py
import time, requests, random
from typing import Dict, Any
from .config import Config

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (ETL/steam-project)"})
BASE = "https://store.steampowered.com/appreviews/{app_id}"

class SteamAPIError(Exception):
    pass

def fetch_reviews_page(app_id: str, cursor: str = "*", max_retries: int = 5) -> Dict[str, Any]:
    params = {
        "json": 1,
        "filter": Config.steam_filter,       # ex: updated|recent
        "language": Config.steam_language,   # ex: all
        "day_range": Config.steam_day_range, # ex: 3650
        "review_type": Config.steam_review_type,
        "purchase_type": Config.steam_purchase_type,
        "num_per_page": Config.steam_num_per_page,  # ex: 100
        "cursor": cursor,
    }
    url = BASE.format(app_id=app_id)
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=20)
            if r.status_code != 200:
                raise SteamAPIError(f"HTTP {r.status_code}: {r.text[:200]}")
            data = r.json()
            if not data.get("success"):
                raise SteamAPIError(f"API success=0: {data}")
            time.sleep(Config.api_sleep_seconds)  # anti-throttling
            return data
        except Exception as e:
            last_err = e
            backoff = min(2 ** attempt, 8) + random.random()
            print(f"[WARN] fetch attempt {attempt}/{max_retries} failed: {e} → retry in {backoff:.1f}s")
            time.sleep(backoff)
    raise SteamAPIError(f"Failed after {max_retries} retries: {last_err}")
