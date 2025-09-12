# etl/bronze_extract.py
import os, json, datetime
from typing import Iterable, Dict
from etl.gcp_clients import get_storage_client
from etl.config import Config

def _gcs_path(app_id: str, dt: str) -> str:
    # Exemple: bronze/raw/app_id=570/dt=2025-09-12/data.ndjson
    return f"bronze/raw/app_id={app_id}/dt={dt}/data.ndjson"

def _write_ndjson_to_gcs(bucket_name: str, blob_path: str, records: Iterable[Dict]):
    storage = get_storage_client()
    bucket = storage.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    # Ecriture append (simple) : on lit puis on réécrit (ou on écrase)
    # Pour du vrai append, préférer compose API ou écrire un fichier par shard
    lines = []
    for r in records:
        lines.append(json.dumps(r, ensure_ascii=False))

    data = ("\n".join(lines) + "\n").encode("utf-8")
    blob.upload_from_string(data, content_type="application/x-ndjson")

def extract_app(app_id: str, mode: str = "incr"):
    # ... ta logique existante pour récupérer les reviews 'records'
    # records = [{'review_id': '...', 'text': '...', 'timestamp': '...'}, ...]
    # Par exemple tu as déjà un crawler dans ton code — je ne le remplace pas ici.

    # Ici, je montre juste la partie "write to GCS"
    dt = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    bucket = Config.gcs_bucket
    if not bucket:
        raise RuntimeError("GCS_BUCKET non défini (Config.gcs_bucket).")

    blob_path = _gcs_path(str(app_id), dt)

    # TODO: brancher tes 'records' récupérés réellement:
    # records = fetch_reviews_from_steam_api(app_id, mode)
    records = []  # <-- à remplacer par ton résultat réel

    if records:
        _write_ndjson_to_gcs(bucket, blob_path, records)
    else:
        # No-op si pas de nouvelles données
        pass
