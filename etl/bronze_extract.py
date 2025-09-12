# etl/bronze_extract.py
import os, io, json, datetime as dt, time
from typing import Iterable, Dict, List, Tuple
import requests

from etl.gcp_clients import get_storage_client
from etl.config import Config

STEAM_URL = "https://store.steampowered.com/appreviews/{app_id}"
HEADERS = {"User-Agent": "steamdb-etl/1.0 (+airflow)"}

def _gcs_path(app_id: str, dts: str) -> str:
    # Exemple: bronze/raw/app_id=570/dt=2025-09-12/data.ndjson
    return f"bronze/raw/app_id={app_id}/dt={dts}/data.ndjson"

def _write_ndjson_to_gcs(bucket_name: str, blob_path: str, records: Iterable[Dict]):
    client = get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    buf = io.BytesIO()
    for r in records:
        line = json.dumps(r, ensure_ascii=False).encode("utf-8") + b"\n"
        buf.write(line)
    buf.seek(0)

    blob.upload_from_file(buf, rewind=True, content_type="application/x-ndjson")

def _fetch_reviews_once(app_id: str, cursor: str = "*", num_per_page: int = 100) -> Tuple[List[Dict], str, bool]:
    """Appelle l’endpoint public des avis Steam (non authentifié)."""
    params = {
        "json": 1,
        "cursor": cursor,
        "num_per_page": num_per_page,
        "filter": "recent",
        "language": "all",
        "purchase_type": "all",
    }
    url = STEAM_URL.format(app_id=app_id)
    r = requests.get(url, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    reviews = data.get("reviews", [])
    next_cursor = data.get("cursor")
    success = bool(data.get("success", 0) == 1)
    return reviews, next_cursor, success

def _normalize_review(raw: Dict, app_id: str) -> Dict:
    # Normalise quelques champs utiles
    author = raw.get("author", {}) or {}
    return {
        "app_id": str(app_id),
        "review_id": raw.get("recommendationid"),
        "review": raw.get("review"),
        "voted_up": raw.get("voted_up"),
        "votes_up": raw.get("votes_up"),
        "votes_funny": raw.get("votes_funny"),
        "weighted_vote_score": raw.get("weighted_vote_score"),
        "comment_count": raw.get("comment_count"),
        "steam_purchase": raw.get("steam_purchase"),
        "received_for_free": raw.get("received_for_free"),
        "written_during_early_access": raw.get("written_during_early_access"),
        "timestamp_created": raw.get("timestamp_created"),
        "timestamp_updated": raw.get("timestamp_updated"),
        "author_steamid": author.get("steamid"),
        "author_num_games_owned": author.get("num_games_owned"),
        "author_num_reviews": author.get("num_reviews"),
        "author_playtime_forever": author.get("playtime_forever"),
        "author_playtime_last_two_weeks": author.get("playtime_last_two_weeks"),
        "author_playtime_at_review": author.get("playtime_at_review"),
        "author_last_played": author.get("last_played"),
        "etl_ingested_at": int(time.time()),
    }

def extract_app(app_id: str, mode: str = "incr", pages: int = 2, per_page: int = 100):
    """
    Récupère quelques pages d’avis récents pour app_id et écrit un NDJSON en bronze/GCS.
    - pages: limite pour éviter d’exploser le quota/temps lors d’un run
    """
    if Config.bronze_mode != "gcs":
        print("[BRONZE] bronze_mode != gcs → no-op")
        return

    bucket = Config.gcs_bucket
    if not bucket:
        raise RuntimeError("GCS_BUCKET non défini (Config.gcs_bucket).")

    dts = dt.datetime.utcnow().strftime("%Y-%m-%d")
    blob_path = _gcs_path(str(app_id), dts)

    cursor = "*"
    all_rows: List[Dict] = []
    for _ in range(max(1, pages)):
        try:
            reviews, cursor, success = _fetch_reviews_once(str(app_id), cursor=cursor, num_per_page=per_page)
        except requests.HTTPError as e:
            print(f"[BRONZE][WARN] HTTP {e.response.status_code} for app_id={app_id}: {e}")
            break
        except Exception as e:
            print(f"[BRONZE][ERROR] fetch error for app_id={app_id}: {e}")
            break

        if not reviews:
            break

        for rev in reviews:
            all_rows.append(_normalize_review(rev, str(app_id)))

        # L'API renvoie un cursor pour la suite ; on limite par 'pages'
        if not cursor:
            break

    if not all_rows:
        print(f"[BRONZE] aucune review pour app_id={app_id} (pages={pages}) → pas d’écriture")
        return

    _write_ndjson_to_gcs(bucket, blob_path, all_rows)
    print(f"[BRONZE] wrote gs://{bucket}/{blob_path} ({len(all_rows)} lignes)")
