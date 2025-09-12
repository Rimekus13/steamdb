# etl/silver_clean.py
import os, json
from google.cloud import storage
from etl.gcp_clients import get_storage_client, get_firestore_client
from etl.config import Config

def _iter_gcs_ndjson(bucket_name: str, prefix: str):
    storage = get_storage_client()
    bucket = storage.bucket(bucket_name)
    for blob in bucket.list_blobs(prefix=prefix):
        data = blob.download_as_bytes().decode("utf-8")
        for line in data.splitlines():
            if line.strip():
                yield json.loads(line)

def _clean_review(r: dict) -> dict:
    # Adaptation simple : trim, normalise, etc.
    txt = (r.get("text") or "").strip()
    r["cleaned_text"] = txt
    return r

def to_silver(app_id: str, dt: str):
    bucket = Config.gcs_bucket
    if not bucket:
        raise RuntimeError("GCS_BUCKET non défini.")

    prefix = f"bronze/raw/app_id={app_id}/dt={dt}/"
    docs = (_clean_review(r) for r in _iter_gcs_ndjson(bucket, prefix))

    db = get_firestore_client()
    # collection: reviews_clean, doc app_id, sous-collection items
    col = db.collection("reviews_clean").document(str(app_id)).collection("items")

    batch = db.batch()
    n = 0
    for r in docs:
        doc_id = str(r.get("review_id") or f"{app_id}_{n}")
        ref = col.document(doc_id)
        batch.set(ref, r, merge=True)
        n += 1
        # commit par lot (sécurité quota ~500 ops par batch Firestore)
        if n % 450 == 0:
            batch.commit()
            batch = db.batch()
    if n % 450 != 0:
        batch.commit()
