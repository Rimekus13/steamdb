# etl/gcp_clients.py
import os
from google.cloud import storage
from google.cloud import firestore

def get_storage_client():
    # Utilise GOOGLE_APPLICATION_CREDENTIALS si défini/monté
    return storage.Client(project=os.getenv("GCP_PROJECT"))

def get_firestore_client():
    project = os.getenv("FIRESTORE_PROJECT") or os.getenv("GCP_PROJECT")
    return firestore.Client(project=project)
