# etl/silver_clean.py
import pandas as pd
from typing import List, Dict, Any
import json, gzip, io

from .config import Config
from .gcs_utils import list_prefix, download_bytes
from .firestore_utils import bulk_upsert_clean
from .text_utils import clean_text, detect_lang, sentiment_scores

def _read_raw_all(app_id: str, dt: str) -> pd.DataFrame:
    """
    Charge tous les fichiers RAW (json.gz) depuis GCS si bronze_mode=gcs,
    sinon depuis le disque local (dev).
    """
    records: List[Dict[str, Any]] = []

    if Config.bronze_mode == "gcs":
        prefix = f"raw/{app_id}/{dt}/"
        for blob_name in list_prefix(Config.gcs_bucket, prefix):
            if not blob_name.endswith(".json.gz"):
                continue
            data = download_bytes(Config.gcs_bucket, blob_name)
            if not data:
                continue
            with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as gz:
                chunk = json.loads(gz.read().decode("utf-8"))
                if isinstance(chunk, list):
                    records.extend(chunk)
    else:
        # mode local (dev)
        import glob
        from pathlib import Path
        base = Config.data_root / f"raw/{app_id}/{dt}"
        for fp in sorted(glob.glob(str(base / "*.json.gz"))):
            with gzip.open(fp, "rt", encoding="utf-8") as f:
                chunk = json.load(f)
                if isinstance(chunk, list):
                    records.extend(chunk)

    return pd.DataFrame(records)

def _standardize_ids(df: pd.DataFrame, app_id: str) -> pd.DataFrame:
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
    text_col = "review" if "review" in df.columns else ("review_text" if "review_text" in df.columns else None)
    if text_col is None:
        df["cleaned_review"] = ""
    else:
        df[text_col] = df[text_col].fillna("").astype(str)
        df["cleaned_review"] = df[text_col].map(clean_text)

    if "language" not in df.columns:
        df["language"] = ""
    df["language"] = df["language"].fillna("").astype(str)
    mask = df["language"].eq("") | df["language"].eq("unknown")
    if mask.any():
        df.loc[mask, "language"] = df.loc[mask, "cleaned_review"].map(detect_lang)

    sents = df["cleaned_review"].map(sentiment_scores)
    df["compound"] = [s.get("compound", 0.0) for s in sents]
    df["sentiment"] = pd.cut(
        df["compound"], bins=[-1.0, -0.05, 0.05, 1.0],
        labels=["neg", "neu", "pos"], include_lowest=True
    ).astype(str)
    return df

def to_silver(app_id: str, dt: str, for_airflow: bool = False) -> str:
    df = _read_raw_all(app_id, dt)
    if df.empty:
        print(f"[INFO] No RAW data for app_id={app_id}, dt={dt}")
        return dt

    df = _standardize_ids(df, app_id)
    df = _prep_timestamps(df)
    df = _prep_text_language_sentiment(df)

    keep = [
        "app_id", "review_id", "recommendationid",
        "author", "language", "voted_up", "votes_up", "votes_funny",
        "weighted_vote_score", "cleaned_review", "compound", "sentiment",
        "timestamp_created", "timestamp_updated", "review_date",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = None

    rows = df[keep].to_dict("records")
    bulk_upsert_clean(rows)
    print(f"[INFO] Silver upserted: {len(rows)} rows for app_id={app_id}")
    return dt
