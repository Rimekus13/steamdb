# etl/firestore_utils.py
from typing import Iterable, Dict, List, Optional
from google.cloud import firestore

def get_fs():
    # Si la VM a un SA attaché, le projet est résolu automatiquement
    return firestore.Client()

# ---------- SILVER ----------
def bulk_upsert_clean(rows: List[Dict]) -> None:
    """
    Upsert des documents dans la collection reviews_clean.
    Clé logique: (app_id + review_id) → doc_id = f"{app_id}__{review_id}"
    """
    if not rows:
        return
    fs = get_fs()
    batch = fs.batch()
    col = fs.collection("reviews_clean")
    i = 0
    for r in rows:
        app_id = str(r.get("app_id", ""))
        review_id = str(r.get("review_id", ""))
        if not app_id or not review_id:
            continue
        doc_id = f"{app_id}__{review_id}"
        batch.set(col.document(doc_id), r, merge=True)
        i += 1
        if i % 400 == 0:  # Firestore batch max 500 ops; 400 pour marge
            batch.commit()
            batch = fs.batch()
    batch.commit()

def col_clean_query() -> List[Dict]:
    fs = get_fs()
    docs = fs.collection("reviews_clean").stream()
    return [d.to_dict() for d in docs]

# ---------- GOLD ----------
def replace_collection(name: str, docs: Iterable[Dict]) -> None:
    """
    Remplace complètement une collection Firestore (simple: purge + réinsert).
    À utiliser pour cooccurrences_counts / cooccurrences_percent.
    """
    fs = get_fs()
    col_ref = fs.collection(name)
    # Purge (attention : Firestore n'a pas de truncate; on efface en batch)
    # Ici on fait simple : on liste et on delete (OK pour tailles raisonnables)
    to_del = list(col_ref.stream())
    for i in range(0, len(to_del), 400):
        batch = fs.batch()
        for doc in to_del[i:i+400]:
            batch.delete(doc.reference)
        batch.commit()

    # Insert en batch
    docs = list(docs)
    if not docs:
        return
    i = 0
    batch = fs.batch()
    for d in docs:
        doc_ref = col_ref.document()  # id auto
        batch.set(doc_ref, d)
        i += 1
        if i % 400 == 0:
            batch.commit()
            batch = fs.batch()
    batch.commit()
