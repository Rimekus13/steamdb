# etl/silver_clean.py
import os, io, json
from typing import List, Dict, Any, Iterable
from datetime import datetime

import pandas as pd

from etl.gcp_clients import get_storage_client, get_firestore_client
from etl.config import Config
from etl.text_utils import clean_text, detect_lang, sentiment_scores  # <- déjà utilisé par ton gold

# ---------- I/O GCS ----------

def _iter_gcs_ndjson(bucket_name: str, prefix: str) -> Iterable[Dict[str, Any]]:
    """
    Lit tous les blobs GCS sous `prefix` et yield chaque ligne NDJSON en dict.
    """
    storage = get_storage_client()
    bucket = storage.bucket(bucket_name)
    for blob in bucket.list_blobs(prefix=prefix):
        if not blob.name.endswith(".ndjson"):
            # on tolère tout mais on lit bien les .ndjson produits en bronze
            pass
        data = blob.download_as_bytes().decode("utf-8", errors="ignore")
        for line in data.splitlines():
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception as e:
                    print(f"[WARN] NDJSON invalid ({blob.name}): {e}")

def _read_raw_all(app_id: str, dt: str) -> pd.DataFrame:
    """
    Charge le RAW Bronze depuis GCS (NDJSON) pour un app_id/date.
    Structure bronze: bronze/raw/app_id=<ID>/dt=<YYYY-MM-DD>/data.ndjson
    """
    bucket = Config.gcs_bucket
    if not bucket:
        raise RuntimeError("GCS_BUCKET non défini.")
    prefix = f"bronze/raw/app_id={app_id}/dt={dt}/"
    rows: List[Dict[str, Any]] = list(_iter_gcs_ndjson(bucket, prefix))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

# ---------- Normalisations ----------

def _standardize_ids(df: pd.DataFrame, app_id: str) -> pd.DataFrame:
    """
    - app_id toujours présent
    - review_id: utilise 'review_id' si présent, sinon 'recommendationid' (Steam), sinon fabrique.
    """
    if "app_id" not in df.columns:
        df["app_id"] = str(app_id)

    if "review_id" not in df.columns:
        if "recommendationid" in df.columns:
            df["review_id"] = df["recommendationid"].astype(str)
        else:
            # fallback: index
            df["review_id"] = [f"{app_id}_{i}" for i in range(len(df))]

    for c in ("app_id", "review_id"):
        df[c] = df[c].astype(str)

    return df

def _prep_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    - force timestamp_created/timestamp_updated en int (epoch seconds), défaut 0
    - ajoute review_date = YYYY-MM-DD (UTC) depuis timestamp_created
    """
    for c in ("timestamp_created", "timestamp_updated"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        else:
            df[c] = 0

    df["review_date"] = (
        pd.to_datetime(df["timestamp_created"], unit="s", utc=True, errors="coerce")
        .dt.date.astype(str)
        .fillna("")
    )
    return df

def _prep_text_language_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    - choisit la bonne colonne de texte: review, review_text, text
    - cleaned_review via clean_text()
    - langue via detect_lang() si vide/inconnue
    - sentiment via sentiment_scores() → compound + label {neg,neu,pos}
    """
    # 1) texte
    text_col = None
    for cand in ("review", "review_text", "text"):
        if cand in df.columns:
            text_col = cand
            break

    if text_col is None:
        df["cleaned_review"] = ""
    else:
        df[text_col] = df[text_col].fillna("").astype(str)
        df["cleaned_review"] = df[text_col].map(clean_text)

    # 2) langue
    if "language" not in df.columns:
        df["language"] = ""
    df["language"] = df["language"].fillna("").astype(str)
    need_detect = df["language"].eq("") | df["language"].eq("unknown")
    if need_detect.any():
        df.loc[need_detect, "language"] = df.loc[need_detect, "cleaned_review"].map(detect_lang)

    # 3) sentiment
    sents = df["cleaned_review"].map(sentiment_scores)
    df["compound"] = [s.get("compound", 0.0) for s in sents]
    df["sentiment"] = pd.cut(
        df["compound"], bins=[-1.0, -0.05, 0.05, 1.0],
        labels=["neg", "neu", "pos"], include_lowest=True
    ).astype(str)

    return df

# ---------- Publication Firestore ----------

def _firestore_upsert_rows(app_id: str, rows: List[Dict[str, Any]]) -> int:
    """
    Upsert en batch dans Firestore:
      collection 'reviews_clean' / document <app_id> / subcollection 'items' / doc <review_id>
    """
    db = get_firestore_client()
    col = db.collection("reviews_clean").document(str(app_id)).collection("items")

    n = 0
    batch = db.batch()
    for r in rows:
        doc_id = str(r.get("review_id", f"{app_id}_{n}"))
        ref = col.document(doc_id)
        batch.set(ref, r, merge=True)
        n += 1
        if n % 450 == 0:
            batch.commit()
            batch = db.batch()
    if n % 450 != 0:
        batch.commit()
    return n

# ---------- Entrée principale ----------

def to_silver(app_id: str, dt: str, for_airflow: bool = False) -> str:
    """
    Lit le RAW depuis GCS, clean/normalise, et upsert le CLEAN dans Firestore.
    Retourne la date traitée.
    """
    df = _read_raw_all(app_id, dt)
    if df.empty:
        print(f"[INFO] No RAW data for app_id={app_id}, dt={dt}")
        return dt

    df = _standardize_ids(df, app_id)
    df = _prep_timestamps(df)
    df = _prep_text_language_sentiment(df)

    # colonnes utiles pour downstream (gold/streamlit)
    keep = [
        "app_id", "review_id",
        "recommendationid",
        "author", "language",
        "voted_up", "votes_up", "votes_funny",
        "weighted_vote_score",
        "cleaned_review", "compound", "sentiment",
        "timestamp_created", "timestamp_updated",
        "review_date",
    ]
    # on évite les KeyError
    for col in keep:
        if col not in df.columns:
            df[col] = None

    rows = df[keep].to_dict("records")
    wrote = _firestore_upsert_rows(app_id, rows)
    print(f"[SILVER] Upserted {wrote} rows for app_id={app_id}, dt={dt}")
    return dt
