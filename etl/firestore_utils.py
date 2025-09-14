# etl/firestore_utils.py
from typing import Iterable, Dict, List, Optional
import os
from google.cloud import firestore

def _detect_project() -> Optional[str]:
    # Aligne avec ton app Streamlit : on lit FIRESTORE_PROJECT puis GCP_PROJECT puis GOOGLE_CLOUD_PROJECT
    return (
        os.getenv("FIRESTORE_PROJECT")
        or os.getenv("GCP_PROJECT")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
    )

def get_fs():
    """
    Crée un client Firestore en forçant le project si dispo dans l'env.
    Cela évite que le conteneur Airflow se connecte à un projet vide par défaut.
    """
    project = _detect_project()
    if project:
        return firestore.Client(project=project)
    return firestore.Client()

# ---------- SILVER ----------
def bulk_upsert_clean(rows: List[Dict]) -> None:
    """
    Upsert des documents dans la collection reviews_clean (flat).
    Clé logique: (app_id + review_id) → doc_id = f"{app_id}__{review_id}"
    """
    if not rows:
        return
    fs = get_fs()
    col = fs.collection("reviews_clean")
    batch = fs.batch()
    i = 0
    for r in rows:
        app_id = str(r.get("app_id", "")).strip()
        review_id = str(r.get("review_id", "")).strip()
        if not app_id or not review_id:
            continue
        doc_id = f"{app_id}__{review_id}"
        batch.set(col.document(doc_id), r, merge=True)
        i += 1
        if i % 400 == 0:
            batch.commit()
            batch = fs.batch()
    batch.commit()

def col_clean_query() -> List[Dict]:
    """
    Retourne tous les documents de reviews_clean (flat).
    """
    fs = get_fs()
    return [d.to_dict() for d in fs.collection("reviews_clean").stream()]

# ---------- GOLD ----------
def replace_collection(name: str, docs: Iterable[Dict], id_keys: Optional[List[str]] = None) -> None:
    """
    Remplace complètement une collection Firestore.
    - Si id_keys est fourni, construit un doc_id déterministe (ex: app_id__period__token_a__token_b).
    - Sinon, laisse Firestore générer un ID auto.
    """
    fs = get_fs()
    col_ref = fs.collection(name)

    # Purge
    existing = list(col_ref.stream())
    for i in range(0, len(existing), 400):
        batch = fs.batch()
        for doc in existing[i:i+400]:
            batch.delete(doc.reference)
        batch.commit()

    # Insert
    docs = list(docs)
    if not docs:
        return

    def _make_id(d: Dict) -> Optional[str]:
        if not id_keys:
            return None
        try:
            parts = [str(d[k]) for k in id_keys]
            return "__".join(parts)
        except Exception:
            return None

    batch = fs.batch()
    i = 0
    for d in docs:
        doc_id = _make_id(d)
        ref = col_ref.document(doc_id) if doc_id else col_ref.document()
        batch.set(ref, d)
        i += 1
        if i % 400 == 0:
            batch.commit()
            batch = fs.batch()
    batch.commit()
