# etl/bronze_extract.py
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import gzip, json, hashlib, io

from .http import fetch_reviews_page
from .state import load_state, save_state
from .config import Config
from .gcs_utils import upload_bytes

def _raw_out_dir(app_id: str, dt: str) -> Path:
    d = Config.data_root / f"raw/{app_id}/{dt}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _flush_chunk(out_dir: Path, chunk_idx: int, chunk_reviews: list) -> None:
    out_path = out_dir / f"reviews_chunk_{chunk_idx:04d}.json.gz"
    with gzip.open(out_path, "wt", encoding="utf-8") as gf:
        json.dump(chunk_reviews, gf, ensure_ascii=False)
    print(f"[INFO] Wrote {len(chunk_reviews)} reviews to {out_path.name}")

def _flush_chunk_cloud(app_id: str, dt: str, chunk_idx: int, chunk_reviews: list) -> None:
    # Destination GCS: gs://<bucket>/raw/<app_id>/<YYYY-MM-DD>/reviews_chunk_XXXX.json.gz
    blob_name = f"raw/{app_id}/{dt}/reviews_chunk_{chunk_idx:04d}.json.gz"
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="w") as gz:
        gz.write(json.dumps(chunk_reviews, ensure_ascii=False).encode("utf-8"))
    upload_bytes(Config.gcs_bucket, blob_name, buf.getvalue(), content_type="application/gzip")
    print(f"[INFO] GCS uploaded: gs://{Config.gcs_bucket}/{blob_name} ({len(chunk_reviews)} reviews)")

def extract_app(app_id: str, mode: str = "incr", for_airflow: bool = False,
                max_pages: int = 200, chunk_size: int = 500) -> str:
    state = load_state(app_id)
    max_seen = state.get("max_timestamp_updated", 0)
    cursor = "*" if mode == "full" else state.get("last_cursor", "*")

    dt = datetime.utcnow().strftime("%Y-%m-%d")
    out_dir = _raw_out_dir(app_id, dt)  # utile même en mode GCS pour dev/local

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

    pbar = tqdm(desc=f"Extract {app_id}", unit="rev")

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

        if not reviews:
            print("[INFO] No reviews on this page → stop.")
            break

        ids = [r.get("recommendationid") for r in reviews]
        page_sig = hashlib.md5(json.dumps(ids[:20], sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

        if new_cursor == last_cursor:
            same_cursor_hits += 1
        else:
            same_cursor_hits = 0

        new_ids = [i for i in ids if i and i not in seen_ids]
        for i in new_ids:
            seen_ids.add(i)
        consecutive_no_new = consecutive_no_new + 1 if len(new_ids) == 0 else 0

        if last_page_sig == page_sig or same_cursor_hits >= 2 or consecutive_no_new >= 2:
            break

        # Ajoute au chunk
        chunk.extend(reviews)
        total_unique += len(new_ids)
        pbar.update(len(new_ids))
        pbar.set_postfix(pages=pages, chunk_len=len(chunk))

        # Watermark
        max_page_updated = 0
        for r in reviews:
            tsu = r.get("timestamp_updated") or 0
            if tsu > max_seen: max_seen = tsu
            if tsu > max_page_updated: max_page_updated = tsu

        # Écriture du chunk (local ou GCS)
        if len(chunk) >= chunk_size:
            if Config.bronze_mode == "gcs":
                _flush_chunk_cloud(app_id, dt, chunk_idx, chunk)
            else:
                _flush_chunk(out_dir, chunk_idx, chunk)
            chunk = []
            chunk_idx += 1

        last_cursor = new_cursor
        last_page_sig = page_sig
        cursor = new_cursor

        if total_expected and len(seen_ids) >= total_expected:
            break
        if mode == "incr" and state.get("max_timestamp_updated") and max_page_updated <= state["max_timestamp_updated"]:
            break
        if not cursor:
            break

    if chunk:
        if Config.bronze_mode == "gcs":
            _flush_chunk_cloud(app_id, dt, chunk_idx, chunk)
        else:
            _flush_chunk(out_dir, chunk_idx, chunk)

    pbar.close()
    save_state(app_id, {"max_timestamp_updated": max_seen, "last_cursor": cursor or "*"})
    return dt
