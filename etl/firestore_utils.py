# etl/firestore_utils.py
from typing import Iterable, Dict, List, Optional, Tuple
import time
import logging

from google.api_core.exceptions import ServiceUnavailable, DeadlineExceeded
from google.cloud import firestore

LOG = logging.getLogger(__name__)

def get_fs() -> firestore.Client:
    """Retourne un client Firestore ; le projet est résolu via ADC/ENV."""
    fs = firestore.Client()
    LOG.info("[FS] Projet Firestore utilisé: %s", fs.project)
    return fs

# ---------- SILVER ----------
def bulk_upsert_clean(rows: List[Dict]) -> None:
    """
    Upsert des documents dans la collection reviews_clean (schéma plat).
    Clé logique: (app_id + review_id) → doc_id = f"{app_id}__{review_id}"
    """
    if not rows:
        LOG.info("[SILVER] bulk_upsert_clean: aucune ligne à insérer.")
        return
    fs = get_fs()
    col = fs.collection("reviews_clean")

    i = 0
    batch = fs.batch()
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
    if i % 400 != 0:
        batch.commit()
    LOG.info("[SILVER] bulk_upsert_clean: %d doc(s) upsert dans `reviews_clean` (plat)", i)

def col_clean_query(limit_per_page: int = 2000, max_pages: int = 500) -> List[Dict]:
    """
    Lit les documents de `reviews_clean` (schéma plat) de façon paginée pour éviter les timeouts.
    NOTE: si ton pipeline écrit au schéma imbriqué, ce reader ne les verra pas (c'est volontaire
    car le Gold lit le plat).
    """
    fs = get_fs()
    col = fs.collection("reviews_clean")
    all_rows: List[Dict] = []

    last_doc = None
    page = 0
    while page < max_pages:
        page += 1
        try:
            q = col.limit(limit_per_page)
            if last_doc is not None:
                q = q.start_after(last_doc)

            docs = list(q.stream())
            if not docs:
                break

            all_rows.extend([d.to_dict() for d in docs])
            last_doc = docs[-1]
            LOG.info("[GOLD][DEBUG] Page %d — cumul: %d", page, len(all_rows))
            # sécurité pour ne pas aspirer un volume trop grand si tu veux limiter
            # if len(all_rows) >= SOME_LIMIT: break
        except (ServiceUnavailable, DeadlineExceeded) as e:
            LOG.warning("[GOLD][WARN] Page %d: Firestore stream timeout (%s). Retry court…", page, e)
            time.sleep(1.5)
            continue

    LOG.info("[GOLD][DEBUG] lectures paginées terminées — total: %d", len(all_rows))
    return all_rows

# ---------- GOLD ----------
def _purge_collection(col_ref, page_size: int = 300, max_loops: int = 10000) -> int:
    """
    Supprime une collection par lots (limit + batch.delete) pour éviter les scans massifs.
    Retourne le nombre total de documents supprimés.
    """
    fs = get_fs()
    deleted_total = 0
    loops = 0

    while loops < max_loops:
        loops += 1
        try:
            docs = list(col_ref.limit(page_size).stream())
        except (ServiceUnavailable, DeadlineExceeded) as e:
            LOG.warning("[GOLD][PURGE] Timeout lors du stream (loop=%d): %s. Backoff…", loops, e)
            time.sleep(2.0)
            continue

        if not docs:
            break

        batch = fs.batch()
        for d in docs:
            batch.delete(d.reference)
        batch.commit()

        deleted_total += len(docs)
        LOG.info("[GOLD][PURGE] Lot supprimé: %d doc(s) — total=%d", len(docs), deleted_total)

        # micro-pause pour éviter de saturer l'API
        time.sleep(0.1)

    return deleted_total

def replace_collection(name: str, docs: Iterable[Dict]) -> None:
    """
    Remplace complètement une collection Firestore (cooccurrences_*).
    Implémente une purge paginée robuste + insert par lots.
    """
    fs = get_fs()
    col_ref = fs.collection(name)

    # Purge paginée
    deleted = _purge_collection(col_ref, page_size=300)
    LOG.info("[GOLD] Collection `%s` purgée (%d docs supprimés)", name, deleted)

    # Insert en batch
    docs = list(docs)
    if not docs:
        LOG.info("[GOLD] Aucune donnée à insérer dans `%s`.", name)
        return

    i = 0
    batch = fs.batch()
    for d in docs:
        batch.set(col_ref.document(), d)
        i += 1
        if i % 400 == 0:
            batch.commit()
            batch = fs.batch()
    if i % 400 != 0:
        batch.commit()
    LOG.info("[GOLD] Insert terminé: %d docs dans `%s`", i, name)
