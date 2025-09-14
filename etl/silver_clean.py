# etl/silver_clean.py
from __future__ import annotations

import io
import json
from typing import List, Dict, Any, Iterable
from datetime import datetime

import pandas as pd

from etl.gcp_clients import get_storage_client, get_firestore_client
from etl.config import Config
from etl.text_utils import clean_text, detect_lang, sentiment_scores


# -----------------------------
# --------- GCS I/O -----------
# -----------------------------
def _iter_gcs_ndjson(bucket_name: str, prefix: str) -> Iterable[Dict[str, Any]]:
    """
    Parcourt tous les blobs GCS sous `prefix` et yield chaque ligne NDJSON -> dict.
    Attend des fichiers produits par bronze: bronze/raw/app_id=<ID>/dt=<YYYY-MM-DD>/data.ndjson
    """
    storage = get_storage_client()
    bucket = storage.bucket(bucket_name)
    for blob in bucket.list_blobs(prefix=prefix):
        data = blob.download_as_bytes().decode("utf-8", errors="ignore")
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"[SILVER][WARN] NDJSON invalide ({blob.name}): {e}")


def _read_raw_all(app_id: str, dt: str) -> pd.DataFrame:
    """
    Charge tout le RAW Bronze depuis GCS (NDJSON) pour un app_id/date.
    Structure: bronze/raw/app_id=<ID>/dt=<YYYY-MM-DD>/...
    """
    bucket = Config.gcs_bucket
    if not bucket:
        raise RuntimeError("GCS_BUCKET non défini (Config.gcs_bucket).")
    prefix = f"bronze/raw/app_id={app_id}/dt={dt}/"
    rows: List[Dict[str, Any]] = list(_iter_gcs_ndjson(bucket, prefix))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------
# --------- Normalisations --------
# ---------------------------------
def _standardize_ids(df: pd.DataFrame, app_id: str) -> pd.DataFrame:
    """
    - force app_id présent (string)
    - crée review_id depuis recommendationid, sinon index synthétique.
    """
    df = df.copy()
    if "app_id" not in df.columns:
        df["app_id"] = str(app_id)
    else:
        df["app_id"] = df["app_id"].astype(str)

    if "review_id" not in df.columns:
        if "recommendationid" in df.columns:
            df["review_id"] = df["recommendationid"].astype(str)
        else:
            df["review_id"] = [f"{app_id}_{i}" for i in range(len(df))]
    else:
        df["review_id"] = df["review_id"].astype(str)

    return df


def _prep_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    - cast timestamp_created / timestamp_updated -> int (epoch seconds)
    - ajoute review_date (YYYY-MM-DD, UTC) dérivée de timestamp_created
    """
    df = df.copy()
    for c in ("timestamp_created", "timestamp_updated"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        else:
            df[c] = 0

    dt_series = pd.to_datetime(df["timestamp_created"], unit="s", utc=True, errors="coerce")
    df["review_date"] = dt_series.dt.date.astype(str).fillna("")
    return df


def _prep_playtime_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée playtime_hours à partir des champs Steam (minutes -> heures):
      - author.playtime_at_review
      - author.playtime_forever
      - author.playtime_last_two_weeks (fallback si rien)
    Les colonnes peuvent être soit imbriquées dans 'author' (dict), soit à plat
    (author_playtime_*) selon le bronze.
    """
    df = df.copy()

    # 1) si bronze a aplati (author_playtime_*)
    if "author_playtime_at_review" in df.columns:
        at_rev = pd.to_numeric(df["author_playtime_at_review"], errors="coerce")
    else:
        # 2) sinon, essayer d'aller chercher dans l'objet author
        def _get_author_field(x, key):
            if isinstance(x, dict):
                return x.get(key)
            return None

        if "author" in df.columns:
            at_rev = df["author"].map(lambda x: _get_author_field(x, "playtime_at_review"))
        else:
            at_rev = None

        at_rev = pd.to_numeric(at_rev, errors="coerce")

    # fallback: forever
    if "author_playtime_forever" in df.columns:
        forever = pd.to_numeric(df["author_playtime_forever"], errors="coerce")
    else:
        if "author" in df.columns:
            forever = df["author"].map(lambda x: x.get("playtime_forever") if isinstance(x, dict) else None)
        else:
            forever = None
        forever = pd.to_numeric(forever, errors="coerce")

    # dernier recours: last two weeks
    if "author_playtime_last_two_weeks" in df.columns:
        last2 = pd.to_numeric(df["author_playtime_last_two_weeks"], errors="coerce")
    else:
        if "author" in df.columns:
            last2 = df["author"].map(lambda x: x.get("playtime_last_two_weeks") if isinstance(x, dict) else None)
        else:
            last2 = None
        last2 = pd.to_numeric(last2, errors="coerce")

    # priorité: at_review > forever > last2
    play_minutes = at_rev.fillna(forever).fillna(last2)
    hours = (play_minutes.fillna(0) / 60.0).astype(float).round(2)
    df["playtime_hours"] = hours

    return df


def _prep_text_language_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    - choisit review/review_text/text comme source
    - cleaned_review via clean_text()
    - language: garde existant sinon détecte
    - sentiment via sentiment_scores(): 'compound' + label neg/neu/pos
    """
    df = df.copy()

    # Texte source
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

    # Langue
    if "language" not in df.columns:
        df["language"] = ""
    df["language"] = df["language"].fillna("").astype(str)
    need_detect = df["language"].eq("") | df["language"].eq("unknown")
    if need_detect.any():
        df.loc[need_detect, "language"] = df.loc[need_detect, "cleaned_review"].map(detect_lang)

    # Sentiment
    sents = df["cleaned_review"].map(sentiment_scores)
    df["compound"] = [float(s.get("compound", 0.0)) for s in sents]
    df["sentiment"] = pd.cut(
        df["compound"], bins=[-1.0, -0.05, 0.05, 1.0],
        labels=["neg", "neu", "pos"], include_lowest=True
    ).astype(str)

    return df


# ----------------------------------------
# --------- Publication Firestore --------
# ----------------------------------------
def _firestore_upsert_rows(app_id: str, rows: List[Dict[str, Any]]) -> int:
    """
    Upsert en batch:
      collection 'reviews_clean' / doc <app_id> / sub 'items' / doc <review_id>
    """
    db = get_firestore_client()
    col = db.collection("reviews_clean").document(str(app_id)).collection("items")

    n = 0
    batch = db.batch()
    for r in rows:
        doc_id = str(r.get("review_id") or f"{app_id}_{n}")
        ref = col.document(doc_id)
        batch.set(ref, r, merge=True)
        n += 1
        if n % 450 == 0:
            batch.commit()
            batch = db.batch()
    if n % 450 != 0:
        batch.commit()
    return n


# ----------------------------------------
# --------- Entrée principale -----------
# ----------------------------------------
def to_silver(app_id: str, dt: str, for_airflow: bool = False) -> str:
    """
    Lit le RAW depuis GCS, clean/normalise, calcule playtime_hours & sentiment,
    puis upsert dans Firestore. Retourne dt (chaîne 'YYYY-MM-DD').
    """
    df = _read_raw_all(app_id, dt)
    if df.empty:
        print(f"[SILVER] Aucun RAW pour app_id={app_id}, dt={dt}")
        return dt

    df = _standardize_ids(df, app_id)
    df = _prep_timestamps(df)
    df = _prep_playtime_hours(df)
    df = _prep_text_language_sentiment(df)

    # colonnes utiles pour l’UI Streamlit
    keep = [
        "app_id", "review_id",
        "recommendationid",
        "author", "language",
        "voted_up", "votes_up", "votes_funny",
        "weighted_vote_score",
        "cleaned_review", "compound", "sentiment",
        "timestamp_created", "timestamp_updated",
        "review_date",
        "playtime_hours",
    ]
    # éviter KeyError si champs absents
    for col in keep:
        if col not in df.columns:
            df[col] = None

    rows = df[keep].to_dict("records")
    wrote = _firestore_upsert_rows(app_id, rows)
    print(f"[SILVER] Upserted {wrote} rows → reviews_clean/{app_id}/items/*  (dt={dt})")
    return dt
