# etl/firestore_utils.py
from typing import Iterable, Dict, List, Optional
import os
from google.cloud import firestore

# ---------- Détection du projet ----------
def _detect_project() -> Optional[str]:
    return (
        os.getenv("FIRESTORE_PROJECT")
        or os.getenv("GCP_PROJECT")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
    )

def get_fs():
    """Client Firestore avec logs du projet résolu."""
    project = _detect_project()
    if project:
        print(f"[FS] Projet Firestore utilisé: {project}")
        return firestore.Client(project=project)
    print("[FS] Projet Firestore (ADC par défaut) — aucune variable projet explicite")
    return firestore.Client()

# ---------- SILVER (schéma plat) ----------
def bulk_upsert_clean(rows: List[Dict]) -> None:
    """
    Upsert dans la collection *plate* `reviews_clean`.
    Clé logique: (app_id + review_id) → doc_id = f\"{app_id}__{review_id}\".
    """
    if not rows:
        print("[SILVER] bulk_upsert_clean: 0 ligne → rien à faire")
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
        if i % 400 == 0:  # marge sous la limite 500 ops/batch
            batch.commit()
            batch = fs.batch()
    batch.commit()
    print(f"[SILVER] bulk_upsert_clean: {i} doc(s) upsert dans `reviews_clean` (plat)")

# ---------- Aide au debug : échantillons & comptages ----------
def _iter_flat(fs, limit: int = 5):
    col = fs.collection("reviews_clean")
    for d in col.limit(limit).stream():
        yield d.id, d.to_dict()

def _iter_nested(fs, limit: int = 5):
    """
    Renvoie (chemin, dict) depuis la mise en page imbriquée:
    reviews_clean/{app_id}/items/{doc}
    """
    yielded = 0
    for appdoc in fs.collection("reviews_clean").limit(1000).stream():
        sub = appdoc.reference.collection("items").limit(1000).stream()
        for d in sub:
            yield f"{appdoc.id}/items/{d.id}", d.to_dict()
            yielded += 1
            if yielded >= limit:
                return

def log_fs_state(sample: int = 5) -> None:
    """Affiche l’état des données Firestore pour faciliter le diagnostic."""
    fs = get_fs()
    project = _detect_project() or "<ADC par défaut>"
    print(f"[FS][DEBUG] Projet résolu: {project}")

    # Collection plate
    flat_docs = list(fs.collection("reviews_clean").limit(1000).stream())
    print(f"[FS][DEBUG] `reviews_clean` (plat) — premiers 1000: {len(flat_docs)} doc(s)")
    for i, (doc_id, doc) in enumerate(_iter_flat(fs, limit=sample), 1):
        keys = list(doc.keys())
        print(f"[FS][DEBUG] plat échantillon {i}: id={doc_id} clés={keys[:12]}")

    # Mise en page imbriquée (échantillons)
    nested_samples = list(_iter_nested(fs, limit=sample))
    print(f"[FS][DEBUG] imbriqué — échantillons (non exhaustif): {len(nested_samples)}")
    for i, (path, doc) in enumerate(nested_samples, 1):
        keys = list(doc.keys())
        print(f"[FS][DEBUG] imbriqué échantillon {i}: path={path} clés={keys[:12]}")

# ---------- Lecteur unifié pour Gold ----------
def col_clean_query() -> List[Dict]:
    """
    Lecteur robuste pour Gold :
    1) essaie la collection *plate* `reviews_clean`;
    2) si vide, essaie la mise en page imbriquée `reviews_clean/{app_id}/items/*`
       et *aplatit* les documents en une liste de dicts homogènes.
    """
    fs = get_fs()
    project = _detect_project() or "<ADC par défaut>"
    print(f"[GOLD][DEBUG] col_clean_query — projet = {project}")

    # 1) lecture *plate*
    flat_docs = list(fs.collection("reviews_clean").stream())
    flat_count = len(flat_docs)
    print(f"[GOLD][DEBUG] `reviews_clean` plat — nb docs: {flat_count}")
    if flat_count > 0:
        rows = [d.to_dict() for d in flat_docs]
        if rows:
            print(f"[GOLD][DEBUG] échantillon plat — clés: {list(rows[0].keys())[:12]}")
        return rows

    # 2) fallback schéma imbriqué
    print("[GOLD][DEBUG] plat vide → exploration schéma imbriqué reviews_clean/{app_id}/items/*")
    rows: List[Dict] = []
    max_appdocs = 2000
    per_items_limit = None  # None = tout lire; mettre un entier pour borner

    app_scanned = 0
    for appdoc in fs.collection("reviews_clean").limit(max_appdocs).stream():
        app_scanned += 1
        app_id = appdoc.id
        subq = appdoc.reference.collection("items")
        if per_items_limit:
            subq = subq.limit(per_items_limit)
        for d in subq.stream():
            rec = d.to_dict() or {}
            # normalisations minimales attendues par Gold/Streamlit
            rec.setdefault("app_id", str(app_id))
            if "cleaned_review" not in rec:
                txt = rec.get("review_text") or rec.get("review") or ""
                rec["cleaned_review"] = txt
            rows.append(rec)

    print(f"[GOLD][DEBUG] imbriqué — lignes collectées: {len(rows)} (sur ~{app_scanned} jeux)")
    if rows:
        print(f"[GOLD][DEBUG] échantillon imbriqué — clés: {list(rows[0].keys())[:12]}")
    return rows

# ---------- Écriture des collections Gold ----------
def replace_collection(name: str, docs: Iterable[Dict], id_keys: Optional[List[str]] = None) -> None:
    """
    Remplace complètement une collection Firestore (purge + insert).
    id_keys: liste de champs pour forcer un ID déterministe (sinon ID auto).
    """
    fs = get_fs()
    col_ref = fs.collection(name)

    # Purge (par lots)
    existing = list(col_ref.stream())
    for i in range(0, len(existing), 400):
        batch = fs.batch()
        for doc in existing[i:i+400]:
            batch.delete(doc.reference)
        batch.commit()
    print(f"[GOLD] Collection `{name}` purgée ({len(existing)} docs supprimés)")

    docs = list(docs)
    if not docs:
        print(f"[GOLD] Aucune donnée à insérer dans `{name}`.")
        return

    def _make_id(d: Dict) -> Optional[str]:
        if not id_keys:
            return None
        try:
            return "__".join(str(d[k]) for k in id_keys)
        except Exception:
            return None

    # Insert (par lots)
    i = 0
    batch = fs.batch()
    for d in docs:
        doc_id = _make_id(d)
        ref = col_ref.document(doc_id) if doc_id else col_ref.document()
        batch.set(ref, d)
        i += 1
        if i % 400 == 0:
            batch.commit()
            batch = fs.batch()
    batch.commit()
    print(f"[GOLD] {i} doc(s) insérés dans `{name}` (id_keys={id_keys})")
