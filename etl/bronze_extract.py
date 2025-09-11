# etl/bronze_extract.py
from datetime import datetime
from etl.config import Config
from google.cloud import storage
from io import BytesIO

from pathlib import Path
import gzip, json, hashlib

from .http import fetch_reviews_page
from .state import load_state, save_state

def _raw_out_dir(app_id: str, dt: str) -> Path:
    d = Path(f"data/raw/{app_id}/{dt}")
    d.mkdir(parents=True, exist_ok=True)
    return d

def _flush_chunk(out_dir: Path, chunk_idx: int, chunk_reviews: list) -> None:
    out_path = out_dir / f"reviews_chunk_{chunk_idx:04d}.json.gz"
    with gzip.open(out_path, "wt", encoding="utf-8") as gf:
        # On écrit un TABLEAU JSON (compatible avec ton parsing actuel)
        json.dump(chunk_reviews, gf, ensure_ascii=False)
    print(f"[INFO] Wrote {len(chunk_reviews)} reviews to {out_path.name}")

def _flush_chunk_cloud(app_id: str, dt: str, chunk_idx: int, chunk_reviews: list) -> None:
    client = storage.Client(project=Config.gcp_project)
    bucket = client.bucket(Config.gcs_bucket)
    blob = bucket.blob(f"raw/{app_id}/{dt}/reviews_chunk_{chunk_idx:04d}.json.gz")

    buf = BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gf:
        gf.write(json.dumps(chunk_reviews, ensure_ascii=False).encode("utf-8"))
    blob.upload_from_string(buf.getvalue(), content_type="application/gzip")
    print(f"[INFO] GCS wrote {len(chunk_reviews)} reviews to {blob.name}")


def extract_app(app_id: str, mode: str = "incr", for_airflow: bool = False,
                max_pages: int = 200, chunk_size: int = 500) -> str:
    """
    API Steam → RAW datalake avec garde-fous ET chunking:
    - on accumule plusieurs pages jusqu'à chunk_size avis puis on écrit 1 fichier .json.gz
    - garde-fous: cursor identique, page identique, 0 nouveaux IDs 2x
    """
    state = load_state(app_id)
    max_seen = state.get("max_timestamp_updated", 0)
    cursor = "*" if mode == "full" else state.get("last_cursor", "*")

    dt = datetime.utcnow().strftime("%Y-%m-%d")
    out_dir = _raw_out_dir(app_id, dt)

    pages = 0
    same_cursor_hits = 0
    last_cursor = None
    last_page_sig = None
    consecutive_no_new = 0
    seen_ids = set()
    chunk = []
    chunk_idx = 1
    total_unique = 0
    total_expected = None

    print(f"[INFO] Start extraction for {app_id}")


    while True:
        pages += 1
        if pages > max_pages:
            print(f"[INFO] Stop after max_pages={max_pages}")
            break

        data = fetch_reviews_page(app_id, cursor)
        reviews = data.get("reviews") or []
        qs = data.get("query_summary") or {}
        if total_expected is None:
            total_expected = qs.get("total_reviews")
        new_cursor = data.get("cursor") or ""

        if pages == 1:
            print(f"[DEBUG] First page: {len(reviews)} reviews, cursor={new_cursor!r}, total_expected={total_expected}")

        if not reviews:
            print("[INFO] No reviews on this page → stop.")
            break

        # Garde-fous page/cursor
        ids = [r.get("recommendationid") for r in reviews]
        page_sig = hashlib.md5(json.dumps(ids[:20], sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

        if new_cursor == last_cursor:
            same_cursor_hits += 1
        else:
            same_cursor_hits = 0

        # Sélectionne seulement les avis uniques pour le compteur (on stocke tout, mais la barre compte les uniques)
        new_ids = [i for i in ids if i and i not in seen_ids]
        for i in new_ids:
            seen_ids.add(i)
        consecutive_no_new = consecutive_no_new + 1 if len(new_ids) == 0 else 0

        if last_page_sig == page_sig or same_cursor_hits >= 2 or consecutive_no_new >= 2:
            why = []
            if last_page_sig == page_sig: why.append("page identique")
            if same_cursor_hits >= 2:     why.append("cursor inchangé")
            if consecutive_no_new >= 2:   why.append("aucun nouvel avis")
            print(f"[WARN] Arrêt ( {', '.join(why)} ).")
            break

        # Ajoute la page AU CHUNK (on stocke la page brute, pas filtrée)
        chunk.extend(reviews)
        total_unique += len(new_ids)
        print(f"[DEBUG] page={pages}, new_ids={len(new_ids)}, chunk_len={len(chunk)}")

        # Watermark
        max_page_updated = 0
        for r in reviews:
            tsu = r.get("timestamp_updated") or 0
            if tsu > max_seen: max_seen = tsu
            if tsu > max_page_updated: max_page_updated = tsu

        # Si on a atteint chunk_size → on écrit un fichier
    if len(chunk) >= chunk_size:
        if Config.bronze_mode == "gcs":
            _flush_chunk_cloud(app_id, dt, chunk_idx, chunk)
        else:
            _flush_chunk(out_dir, chunk_idx, chunk)
        chunk = []
        chunk_idx += 1
    


        # Avance le curseur
        last_cursor = new_cursor
        last_page_sig = page_sig
        cursor = new_cursor

        # Stop si on a atteint le total annoncé (si l'API le fournit)
        if total_expected and len(seen_ids) >= total_expected:
            print("[INFO] Reached total_expected from query_summary → stop.")
            break

        # Stop incrémental si watermark atteint
        if mode == "incr" and state.get("max_timestamp_updated") and max_page_updated <= state["max_timestamp_updated"]:
            print("[INFO] Incremental stop: reached previous watermark.")
            break

        if not cursor:
            print("[INFO] No cursor → stop.")
            break

    # Flush du dernier chunk partiel
    if chunk:
        _flush_chunk(out_dir, chunk_idx, chunk)

    
    print(f"[INFO] Pages fetched={pages-1}, unique reviews seen≈{len(seen_ids)}")
    save_state(app_id, {"max_timestamp_updated": max_seen, "last_cursor": cursor or "*"})
    return dt
