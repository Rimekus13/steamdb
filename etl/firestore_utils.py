# etl/firestore_utils.py
from typing import Iterable, Dict, List, Optional
import os, time, random
from google.cloud import firestore
from google.api_core.exceptions import ServiceUnavailable, DeadlineExceeded, InternalServerError

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
    Clé logique: (app_id + review_id) → doc_id = f"{app_id}__{review_id}".
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
        if i % 400 == 0:  # marge sous 500 ops
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

def log_fs_state(sample: int = 3) -> None:
    """Affiche l’état des données Firestore pour faciliter le diagnostic."""
    fs = get_fs()
    project = _detect_project() or "<ADC par défaut>"
    print(f"[FS][DEBUG] Projet résolu: {project}")

    flat_docs = list(fs.collection("reviews_clean").limit(1000).stream())
    print(f"[FS][DEBUG] `reviews_clean` (plat) — premiers 1000: {len(flat_docs)} doc(s)")
    for i, (doc_id, doc) in enumerate(_iter_flat(fs, limit=sample), 1):
        keys = list(doc.keys())
        print(f"[FS][DEBUG] plat échantillon {i}: id={doc_id} clés={keys[:12]}")

    nested_samples = list(_iter_nested(fs, limit=sample))
    print(f"[FS][DEBUG] imbriqué — échantillons (non exhaustif): {len(nested_samples)}")
    for i, (path, doc) in enumerate(nested_samples, 1):
        keys = list(doc.keys())
        print(f"[FS][DEBUG] imbriqué échantillon {i}: path={path} clés={keys[:12]}")

# ---------- Lecture unifiée pour Gold ----------
def col_clean_query() -> List[Dict]:
    """
    Lecteur robuste pour Gold :
    1) essaie la collection *plate* `reviews_clean`;
    2) si vide, essaie la mise en page imbriquée `reviews_clean/{app_id}/items/*`
       et *aplatit* en liste de dicts homogènes.
    """
    fs = get_fs()
    project = _detect_project() or "<ADC par défaut>"
    print(f"[GOLD][DEBUG] col_clean_query — projet = {project}")

    # 1) PLAT (paginé pour éviter les timeouts)
    rows: List[Dict] = []
    col = fs.collection("reviews_clean")
    last = None
    page_size = 2000
    pages = 0
    try:
        while True:
            q = col.order_by("__name__").limit(page_size)
            if last is not None:
                q = q.start_after(last)
            docs = list(q.stream())
            if not docs:
                break
            pages += 1
            for d in docs:
                rows.append(d.to_dict())
            last = docs[-1]
        print(f"[GOLD][DEBUG] `reviews_clean` plat — pages={pages}, docs={len(rows)}")
    except Exception as e:
        print(f"[GOLD][DEBUG] lecture PLAT a échoué: {e}")

    if rows:
        print(f"[GOLD][DEBUG] échantillon plat — clés: {list(rows[0].keys())[:12]}")
        return rows

    # 2) IMBRIQUÉ
    print("[GOLD][DEBUG] plat vide → exploration schéma imbriqué reviews_clean/{app_id}/items/*")
    max_appdocs = 2000
    per_items_limit = None  # None = tout lire; sinon int
    app_scanned = 0
    rows_nested: List[Dict] = []
    for appdoc in fs.collection("reviews_clean").limit(max_appdocs).stream():
        app_scanned += 1
        app_id = appdoc.id
        sub = appdoc.reference.collection("items")
        if per_items_limit:
            sub = sub.limit(per_items_limit)
        for d in sub.stream():
            rec = d.to_dict() or {}
            rec.setdefault("app_id", str(app_id))
            if "cleaned_review" not in rec:
                txt = rec.get("review_text") or rec.get("review") or ""
                rec["cleaned_review"] = txt
            rows_nested.append(rec)
    print(f"[GOLD][DEBUG] imbriqué — lignes collectées: {len(rows_nested)} (sur ~{app_scanned} jeux)")
    if rows_nested:
        print(f"[GOLD][DEBUG] échantillon imbriqué — clés: {list(rows_nested[0].keys())[:12]}")
    return rows_nested

# ---------- Purge/Insert paginés & robustes ----------
def _paged_delete(col_ref, batch_size: int = 300, max_retries: int = 5) -> int:
    """
    Supprime TOUTE la collection en pages ordonnées par __name__ (évite les timeouts).
    Commit en batchs. Retourne le nombre de docs supprimés.
    """
    fs = get_fs()
    deleted_total = 0
    while True:
        # stream la "page"
        try:
            docs = list(col_ref.order_by("__name__").limit(batch_size).stream())
        except (ServiceUnavailable, DeadlineExceeded, InternalServerError) as e:
            if max_retries <= 0:
                print(f"[GOLD][PURGE] abandon: {e}")
                raise
            backoff = min(8, 2 ** (6 - max_retries)) + random.random()
            print(f"[GOLD][PURGE] transient error {type(e).__name__}: retry in {backoff:.1f}s")
            time.sleep(backoff)
            max_retries -= 1
            continue

        if not docs:
            break

        # supprime cette page en batchs de 400
        for i in range(0, len(docs), 400):
            batch = fs.batch()
            for d in docs[i:i+400]:
                batch.delete(d.reference)
            batch.commit()
        deleted_total += len(docs)
        # boucle: on relance une nouvelle page
    return deleted_total

def replace_collection(name: str, docs: Iterable[Dict], id_keys: Optional[List[str]] = None) -> None:
    """
    Remplace complètement une collection Firestore (purge paginée + insert paginé).
    id_keys: champs pour ID déterministe (sinon ID auto).
    """
    fs = get_fs()
    col_ref = fs.collection(name)

    # Purge paginée (robuste)
    try:
        deleted = _paged_delete(col_ref, batch_size=int(os.getenv("FS_DELETE_BATCH", "300")))
    except Exception as e:
        print(f"[GOLD] Purge `{name}` a échoué: {e}")
        raise
    print(f"[GOLD] Collection `{name}` purgée ({deleted} docs supprimés)")

    # Prépare à insérer
    data = list(docs)
    if not data:
        print(f"[GOLD] Aucune donnée à insérer dans `{name}`.")
        return

    def _make_id(d: Dict) -> Optional[str]:
        if not id_keys:
            return None
        try:
            return "__".join(str(d[k]) for k in id_keys)
        except Exception:
            return None

    # Insert paginé (batchs de 400)
    written = 0
    batch = fs.batch()
    for i, d in enumerate(data, 1):
        doc_id = _make_id(d)
        ref = col_ref.document(doc_id) if doc_id else col_ref.document()
        batch.set(ref, d)
        if i % 400 == 0:
            batch.commit()
            written += 400
            batch = fs.batch()
    batch.commit()
    written += len(data) % 400
    print(f"[GOLD] {written} doc(s) insérés dans `{name}` (id_keys={id_keys})")
