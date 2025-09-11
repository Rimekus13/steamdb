from typing import Iterable, List, Dict
from pymongo import MongoClient, ASCENDING, DESCENDING, UpdateOne
from pymongo.errors import BulkWriteError
from .config import Config

def get_client(for_airflow: bool = False) -> MongoClient:
    uri = Config.mongo_uri_docker if for_airflow else Config.mongo_uri
    return MongoClient(uri)

def get_db(for_airflow: bool = False):
    return get_client(for_airflow)[Config.mongo_db]

# --- Collections uniques ---
def col_raw(for_airflow: bool = False):
    return get_db(for_airflow)["reviews_raw"]

def col_clean(for_airflow: bool = False):
    return get_db(for_airflow)["reviews_clean"]

def col_co_counts(for_airflow: bool = False):
    return get_db(for_airflow)["cooccurrences_counts"]

def col_co_percent(for_airflow: bool = False):
    return get_db(for_airflow)["cooccurrences_percent"]

def ensure_indexes(for_airflow: bool = False):
    # RAW
    r = col_raw(for_airflow)
    r.create_index([("app_id", ASCENDING), ("recommendationid", ASCENDING)], name="uniq_app_reco", unique=True)
    r.create_index([("app_id", ASCENDING), ("timestamp_updated", ASCENDING)], name="by_app_ts")

    # SILVER
    cl = col_clean(for_airflow)
    cl.create_index([("app_id", ASCENDING), ("review_id", ASCENDING)], name="uniq_app_review", unique=True)
    cl.create_index([("app_id", ASCENDING), ("review_date", ASCENDING)], name="by_app_date")
    cl.create_index([("language", ASCENDING)])
    cl.create_index([("sentiment", ASCENDING)])

    # GOLD — COUNTS
    cc = col_co_counts(for_airflow)
    cc.create_index(
        [("app_id", ASCENDING), ("token_a", ASCENDING), ("token_b", ASCENDING), ("period", ASCENDING), ("window", ASCENDING)],
        name="uniq_app_tokens_period_window", unique=True
    )
    cc.create_index([("app_id", ASCENDING), ("period", ASCENDING), ("count", DESCENDING)], name="top_by_period")

    # GOLD — PERCENTS
    cp = col_co_percent(for_airflow)
    cp.create_index(
        [("app_id", ASCENDING), ("token_a", ASCENDING), ("token_b", ASCENDING), ("period", ASCENDING), ("window", ASCENDING)],
        name="uniq_app_tokens_period_window", unique=True
    )
    cp.create_index([("app_id", ASCENDING), ("period", ASCENDING), ("percent", DESCENDING)], name="top_by_period")

def bulk_upsert_raw(app_id: str, reviews: List[Dict], for_airflow: bool = False):
    if not reviews: return
    ops = []
    for r in reviews:
        key = {"app_id": str(app_id), "recommendationid": str(r.get("recommendationid"))}
        doc = dict(r)
        doc["app_id"] = str(app_id)
        ops.append(UpdateOne(key, {"$set": doc}, upsert=True))
    try:
        col_raw(for_airflow).bulk_write(ops, ordered=False)
    except BulkWriteError:
        pass

def bulk_upsert_clean(rows: List[Dict], for_airflow: bool = False):
    if not rows: return
    ops = []
    for r in rows:
        key = {"app_id": str(r["app_id"]), "review_id": str(r["review_id"])}
        ops.append(UpdateOne(key, {"$set": r}, upsert=True))
    col_clean(for_airflow).bulk_write(ops, ordered=False)

def replace_collection(name: str, docs: Iterable[Dict], for_airflow: bool = False, app_id: str | None = None):
    """
    Pour GOLD: on remplace UNIQUEMENT les docs de l'app_id courant si fourni
    (sinon on drop toute la collection — à éviter en multi-app).
    """
    c = get_db(for_airflow)[name]
    docs = list(docs)
    if app_id is None:
        c.drop()
        if docs: c.insert_many(docs, ordered=False)
    else:
        c.delete_many({"app_id": str(app_id)})
        if docs: c.insert_many(docs, ordered=False)
