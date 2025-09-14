# etl/firestore_utils.py
from typing import Iterable, Dict, List
from google.cloud import firestore

def get_fs() -> firestore.Client:
    """
    Retourne le client Firestore (utilise ADC de la VM).
    """
    return firestore.Client()

# ---------- SILVER ----------
def bulk_upsert_clean(rows: List[Dict]) -> None:
    """
    Upsert des documents dans la collection reviews_clean.
    Clé logique: app_id + review_id → doc_id = f"{app_id}__{review_id}".
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
        if i % 400 == 0:  # Firestore batch max 500 ops
            batch.commit()
            batch = fs.batch()
    batch.commit()


def col_clean_query() -> List[Dict]:
    """
    Retourne toutes les reviews_clean sous forme de liste de dicts.
    """
    fs = get_fs()
    docs = fs.collection("reviews_clean").stream()
    return [d.to_dict() for d in docs]

# ---------- GOLD ----------
def replace_collection(name: str, docs: Iterable[Dict], id_keys: List[str] = None) -> None:
    """
    Remplace complètement une collection Firestore (purge + réinsert).
    
    Args:
        name: nom de la collection Firestore (ex: "cooccurrences_counts").
        docs: itérable de dictionnaires.
        id_keys: si fourni, sert à générer un doc_id stable basé sur certaines colonnes.
                 ex: ["app_id", "period", "token_a", "token_b"]
    """
    fs = get_fs()
    col_ref = fs.collection(name)

    # --- Purge la collection ---
    to_del = list(col_ref.stream())
    for i in range(0, len(to_del), 400):
        batch = fs.batch()
        for doc in to_del[i:i+400]:
            batch.delete(doc.reference)
        batch.commit()

    # --- Insert en batch ---
    docs = list(docs)
    if not docs:
        return

    batch = fs.batch()
    i = 0
    for d in docs:
        if id_keys:
            # Crée un ID logique basé sur les colonnes
            doc_id = "__".join(str(d.get(k, "")).replace(" ", "_") for k in id_keys)
            ref = col_ref.document(doc_id)
        else:
            # Sinon ID auto
            ref = col_ref.document()
        batch.set(ref, d)
        i += 1
        if i % 400 == 0:
            batch.commit()
            batch = fs.batch()
    batch.commit()
