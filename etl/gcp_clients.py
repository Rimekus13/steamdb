# etl/gcp_clients.py
import os
from google.cloud import storage, firestore

def get_storage_client() -> storage.Client:
    # ADC via service account de la VM ou GOOGLE_APPLICATION_CREDENTIALS
    project = os.getenv("GCP_PROJECT")
    return storage.Client(project=project)

def get_firestore_client() -> firestore.Client:
    project = os.getenv("FIRESTORE_PROJECT") or os.getenv("GCP_PROJECT")
    return firestore.Client(project=project)
