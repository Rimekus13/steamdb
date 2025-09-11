from typing import Iterable, List, Dict
from pymongo import MongoClient, ASCENDING, DESCENDING, UpdateOne
from pymongo.errors import BulkWriteError
from .config import Config

def get_client(for_airflow: bool = False) -> MongoClient:
    uri = Config.mongo_uri_docker if for_airflow else Config.mongo_uri
    return MongoClient(uri)

def get_db(for_airflow: bool = False):
    return get_client(for_airflow)[Config.mongo_db]

def col_raw(app_id: str, for_airflow: bool = False):
    return get_db(for_airflow)[f"reviews_{app_id}"]

def col_clean(for_airflow: bool = False):
    return get_db(for_airflow)["reviews_clean"]

def col_co_counts(for_airflow: bool = False):
    return get_db(for_airflow)["cooccurrences_counts"]

def col_co_percent(for_airflow: bool = False):
    return get_db(for_airflow)["cooccurrences_percent"]

def ensure_indexes(app_id: str, for_airflow: bool = False):
    """
    Crée les index utiles sur RAW / SILVER / GOLD (idempotent).
    NB: GOLD utilise désormais token_a/token_b + period + window (plus term/co_term).
    """
    # RAW (par app)
    c = col_raw(app_id, for_airflow)
    c.create_index([("recommendationid", ASCENDING)], name="uniq_reco", unique=True)
    c.create_index([("timestamp_updated", ASCENDING)])
    c.create_index([("author.steamid", ASCENDING)])

    # SILVER
    cl = col_clean(for_airflow)
    cl.create_index([("app_id", ASCENDING), ("review_id", ASCENDING)], name="uniq_clean", unique=True)
    cl.create_index([("app_id", ASCENDING), ("review_date", ASCENDING)], name="by_app_period")  # utile pour filtres temporels
    cl.create_index([("language", ASCENDING)])
    cl.create_index([("sentiment", ASCENDING)])

    # GOLD — COUNTS
    cc = col_co_counts(for_airflow)
    # Unicité d’une paire pour un jeu, une période et une fenêtre
    cc.create_index(
        [("app_id", ASCENDING), ("token_a", ASCENDING), ("token_b", ASCENDING), ("period", ASCENDING), ("window", ASCENDING)],
        name="uniq_app_tokens_period_window",
        unique=True
    )
    # Top par période
    cc.create_index([("app_id", ASCENDING), ("period", ASCENDING), ("count", DESCENDING)], name="top_by_period")

    # GOLD — PERCENTS
    cp = col_co_percent(for_airflow)
    cp.create_index(
        [("app_id", ASCENDING), ("token_a", ASCENDING), ("token_b", ASCENDING), ("period", ASCENDING), ("window", ASCENDING)],
        name="uniq_app_tokens_period_window",
        unique=True
    )
    cp.create_index([("app_id", ASCENDING), ("period", ASCENDING), ("percent", DESCENDING)], name="top_by_period")

def bulk_upsert_clean(rows: List[Dict], for_airflow: bool = False):
    """
    Upsert des documents dans reviews_clean par clé (app_id, review_id).
    Idempotent et tolérant aux collisions (BulkWriteError).
    """
    if not rows:
        return
    ops = []
    for r in rows:
        key = {"app_id": str(r["app_id"]), "review_id": str(r["review_id"])}
        ops.append(UpdateOne(key, {"$set": r}, upsert=True))
    try:
        col_clean(for_airflow).bulk_write(ops, ordered=False)
    except BulkWriteError:
        # collisions bénignes si plusieurs workers traitent les mêmes IDs
        pass


def replace_collection(name: str, docs: Iterable[Dict], for_airflow: bool = False):
    """
    Remplace complètement une collection par un nouvel ensemble de documents.
    Utilisé par le build Gold (counts/percent).
    """
    c = get_db(for_airflow)[name]
    c.drop()
    docs = list(docs)
    if docs:
        c.insert_many(docs, ordered=False)