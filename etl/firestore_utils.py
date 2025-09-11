# etl/firestore_utils.py
from typing import Iterable, List, Dict, Optional, Any
from google.cloud import firestore
import os

def _client() -> firestore.Client:
    project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
    return firestore.Client(project=project)

COL_REVIEWS_CLEAN = "reviews_clean"
COL_CO_COUNTS     = "cooccurrences_counts"
COL_CO_PERCENT    = "cooccurrences_percent"

def ensure_indexes(app_id: Optional[str] = None) -> None:
    # Firestore: indexes composites à déclarer dans la console si besoin
    return

def bulk_upsert_clean(rows: List[Dict], *_args, **_kwargs) -> None:
    if not rows:
        return
    db = _client()
    batch = db.batch()
    col = db.collection(COL_REVIEWS_CLEAN)
    for r in rows:
        app_id = str(r.get("app_id"))
        review_id = str(r.get("review_id"))
        ref = col.document(f"{app_id}:{review_id}")
        batch.set(ref, r, merge=True)
    batch.commit()

def col_clean_query(_fields: Optional[Dict[str, Any]] = None) -> List[Dict]:
    db = _client()
    return [d.to_dict() or {} for d in db.collection(COL_REVIEWS_CLEAN).stream()]

def replace_collection(name: str, docs: Iterable[Dict], *_args, **_kwargs) -> None:
    db = _client()
    col = db.collection(name)

    # purge (par batch de 500)
    to_del = list(col.limit(500).stream())
    while to_del:
        batch = db.batch()
        for d in to_del:
            batch.delete(d.reference)
        batch.commit()
        to_del = list(col.limit(500).stream())

    docs = list(docs)
    if not docs:
        return

    batch = db.batch()
    for i, r in enumerate(docs):
        doc_id = f"{r.get('app_id','')}:{r.get('period','')}:{r.get('token_a','')}:{r.get('token_b','')}:{r.get('window','')}".strip(":")
        ref = col.document(doc_id) if doc_id else col.document()
        batch.set(ref, r)
        if (i + 1) % 400 == 0:
            batch.commit()
            batch = db.batch()
    batch.commit()
