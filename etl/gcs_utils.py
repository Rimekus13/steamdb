# etl/gcs_utils.py
from google.cloud import storage
from typing import Iterable, Optional

def get_gcs_client():
    # Si la VM a un Service Account attaché, pas besoin de creds explicites
    return storage.Client()

def upload_bytes(bucket_name: str, blob_name: str, data: bytes, content_type: str = "application/octet-stream"):
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)

def download_bytes(bucket_name: str, blob_name: str) -> Optional[bytes]:
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        return None
    return blob.download_as_bytes()

def list_prefix(bucket_name: str, prefix: str) -> Iterable[str]:
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    for b in client.list_blobs(bucket, prefix=prefix):
        yield b.name
