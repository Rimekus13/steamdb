# etl/silver_clean.py
import pandas as pd
from typing import List, Dict, Any

from .text_utils import clean_text, detect_lang, sentiment_scores
from .config import Config

from .firestore_utils import bulk_upsert_clean
from google.cloud import storage
from io import BytesIO
import gzip, json


def _read_raw_all(app_id: str, dt: str) -> pd.DataFrame:
    """
    Source Bronze :
    - Si BRONZE_MODE='gcs' : gs://<bucket>/raw/{app_id}/{YYYY-MM-DD}/*.json.gz
    - Sinon : data/raw/{app_id}/{YYYY-MM-DD}/*.json.gz (local)
    """
    if Config.bronze_mode == "gcs":
        client = storage.Client(project=Config.gcp_project)
        bucket = client.bucket(Config.gcs_bucket)
        prefix = f"raw/{app_id}/{dt}/"
        records: List[Dict[str, Any]] = []
        for blob in bucket.list_blobs(prefix=prefix):
            if not blob.name.endswith(".json.gz"):
                continue
            data = blob.download_as_bytes()
            try:
                with gzip.GzipFile(fileobj=BytesIO(data), mode="rb") as gf:
                    chunk = json.loads(gf.read().decode("utf-8"))
                    if isinstance(chunk, list):
                        records.extend(chunk)
            except Exception as e:
                print(f"[WARN] Failed {blob.name}: {e}")
        return pd.DataFrame(records)

    # --- Mode fichiers (local dev) ---
    import glob
    from pathlib import Path
    base = Path(f"data/raw/{app_id}/{dt}")
    if not base.exists():
        return pd.DataFrame()
    records: List[Dict[str, Any]] = []
    for fp in sorted(glob.glob(str(base / '*.json.gz'))):
        try:
            with gzip.open(fp, "rt", encoding="utf-8") as f:
                chunk = json.load(f)
                if isinstance(chunk, list):
                    records.extend(chunk)
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")
    return pd.DataFrame(records)


def _standardize_ids(df: pd.DataFrame, app_id: str) -> pd.DataFrame:
    """
    Normalise les identifiants :
    - app_id toujours présent (str)
    - review_id = recommendationid (Steam) si absent (str)
    """
    if "app_id" not in df.columns:
        df["app_id"] = app_id
    if "review_id" not in df.columns and "recommendationid" in df.columns:
        df["review_id"] = df["recommendationid"]
    for c in ("app_id", "review_id"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def _prep_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("timestamp_created", "timestamp_updated"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        else:
            df[c] = 0
    df["review_date"] = (
        pd.to_datetime(df["timestamp_created"], unit="s", utc=True)
        .dt.date.astype(str)
    )
    return df


def _prep_text_language_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    # colonne texte
    text_col = "review" if "review" in df.columns else ("review_text" if "review_text" in df.columns else None)
    if text_col is None:
        df["cleaned_review"] = ""
    else:
        df[text_col] = df[text_col].fillna("").astype(str)
        df["cleaned_review"] = df[text_col].map(clean_text)

    # langue (détection si vide/inconnue)
    if "language" not in df.columns:
        df["language"] = ""
    df["language"] = df["language"].fillna("").astype(str)
    mask = df["language"].eq("") | df["language"].eq("unknown")
    if mask.any():
        df.loc[mask, "language"] = df.loc[mask, "cleaned_review"].map(detect_lang)

    # sentiment (vader compound → classes)
    sents = df["cleaned_review"].map(sentiment_scores)
    df["compound"] = [s.get("compound", 0.0) for s in sents]
    df["sentiment"] = pd.cut(
        df["compound"], bins=[-1.0, -0.05, 0.05, 1.0],
        labels=["neg", "neu", "pos"], include_lowest=True
    ).astype(str)
    return df


def to_silver(app_id: str, dt: str, for_airflow: bool = False) -> str:
    """
    Prépare/normalise et publie le CLEAN dans Mongo (collection reviews_clean).
    """
    df = _read_raw_all(app_id, dt)
    if df.empty:
        print(f"[INFO] No RAW data for app_id={app_id}, dt={dt}")
        return dt

    df = _standardize_ids(df, app_id)
    df = _prep_timestamps(df)
    df = _prep_text_language_sentiment(df)

    # Champs conservés (adapter selon ton dashboard)
    keep = [
        "app_id", "review_id",              # <- IMPORTANT
        "recommendationid",
        "author", "language",
        "voted_up", "votes_up", "votes_funny",
        "weighted_vote_score",
        "cleaned_review", "compound", "sentiment",
        "timestamp_created", "timestamp_updated",
        "review_date",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = None

    rows = df[keep].to_dict("records")
    bulk_upsert_clean(rows, for_airflow=for_airflow)
    print(f"[INFO] Silver upserted: {len(rows)} rows for app_id={app_id}")
    return dt
