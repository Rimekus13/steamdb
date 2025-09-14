# etl/silver_clean.py
import os, io, json
from typing import List, Dict, Any, Iterable
from datetime import datetime

import pandas as pd

from etl.gcp_clients import get_storage_client, get_firestore_client
from etl.config import Config
from etl.text_utils import clean_text, detect_lang, sentiment_scores
from etl.firestore_utils import bulk_upsert_clean, _detect_project

# ---------- I/O GCS ----------

def _iter_gcs_ndjson(bucket_name: str, prefix: str) -> Iterable[Dict[str, Any]]:
    """
    Lit tous les blobs GCS sous `prefix` et yield chaque ligne NDJSON en dict.
    """
    print(f"[SILVER][GCS] Listing blobs: bucket={bucket_name} prefix={prefix}")
    storage = get_storage_client()
    bucket = storage.bucket(bucket_name)
    found = False
    for blob in bucket.list_blobs(prefix=prefix):
        if not blob.name.endswith(".ndjson"):
            # info mais on tolère
            print(f"[SILVER][GCS] Ignored non-ndjson: {blob.name}")
        else:
            print(f"[SILVER][GCS] Reading: {blob.name}")
        data = blob.download_as_bytes().decode("utf-8", errors="ignore")
        lines = data.splitlines()
        print(f"[SILVER][GCS] {blob.name} → {len(lines)} lignes")
        found = True
        for line in lines:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception as e:
                    print(f"[SILVER][WARN] NDJSON invalide ({blob.name}): {e}")
    if not found:
        print("[SILVER][GCS] Aucun blob .ndjson sous ce préfixe.")

def _read_raw_all(app_id: str, dt: str) -> pd.DataFrame:
    """
    Charge le RAW Bronze depuis GCS (NDJSON) pour un app_id/date.
    Structure bronze: bronze/raw/app_id=<ID>/dt=<YYYY-MM-DD>/data.ndjson
    """
    bucket = Config.gcs_bucket
    if not bucket:
        raise RuntimeError("GCS_BUCKET non défini (Config.gcs_bucket).")
    prefix = f"bronze/raw/app_id={app_id}/dt={dt}/"
    rows: List[Dict[str, Any]] = list(_iter_gcs_ndjson(bucket, prefix))
    print(f"[SILVER] RAW total lu pour app_id={app_id}, dt={dt}: {len(rows)} ligne(s)")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

# ---------- Normalisations ----------

def _standardize_ids(df: pd.DataFrame, app_id: str) -> pd.DataFrame:
    """
    - app_id toujours présent
    - review_id: utilise 'review_id' si présent, sinon 'recommendationid', sinon fabrique.
    """
    if "app_id" not in df.columns:
        df["app_id"] = str(app_id)
    else:
        df["app_id"] = df["app_id"].astype(str).fillna(str(app_id))

    if "review_id" not in df.columns:
        if "recommendationid" in df.columns:
            df["review_id"] = df["recommendationid"].astype(str)
        else:
            df["review_id"] = [f"{app_id}_{i}" for i in range(len(df))]

    for c in ("app_id", "review_id"):
        df[c] = df[c].astype(str)

    return df

def _prep_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    - force timestamp_created/timestamp_updated en int (epoch seconds)
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
    - texte → cleaned_review
    - langue (detect si vide/inconnue)
    - sentiment → compound + label
    """
    # 1) texte
    text_col = None
    for cand in ("review", "review_text", "text", "cleaned_review"):
        if cand in df.columns:
            text_col = cand
            break

    if text_col is None:
        df["cleaned_review"] = ""
    else:
        df[text_col] = df[text_col].fillna("").astype(str)
        if text_col == "cleaned_review":
            df["cleaned_review"] = df[text_col]
        else:
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

# ---------- Entrée principale ----------

def to_silver(app_id: str, dt: str, for_airflow: bool = False) -> str:
    """
    Lit le RAW depuis GCS, clean/normalise, et upsert le CLEAN *plat* dans Firestore:
      collection: reviews_clean (doc_id = f"{app_id}__{review_id}")
    Retourne la date traitée.
    """
    project = _detect_project() or "<ADC par défaut>"
    print(f"[SILVER] Démarrage to_silver(app_id={app_id}, dt={dt}) — projet Firestore: {project}")
    print(f"[SILVER] Config: gcs_bucket={Config.gcs_bucket} bronze_mode={Config.bronze_mode}")

    df = _read_raw_all(app_id, dt)
    if df.empty:
        print(f"[SILVER] ❗ Aucune donnée RAW pour app_id={app_id}, dt={dt} (vérifie le chemin et le bronze).")
        return dt

    print(f"[SILVER] Colonnes RAW: {list(df.columns)[:20]}")
    print(f"[SILVER] Exemple RAW (1ère ligne): {df.iloc[0].to_dict()}")

    df = _standardize_ids(df, app_id)
    df = _prep_timestamps(df)
    df = _prep_text_language_sentiment(df)

    keep = [
        "app_id", "review_id",
        "cleaned_review", "compound", "sentiment",
        "language", "review_date",
        "timestamp_created", "timestamp_updated",
        "voted_up", "votes_funny", "votes_up", "weighted_vote_score",
    ]
    # assurer la présence de toutes les colonnes
    for col in keep:
        if col not in df.columns:
            df[col] = None

    rows = df[keep].to_dict("records")
    print(f"[SILVER] À upserter → {len(rows)} ligne(s). Exemple colonnes: {keep}")
    if rows:
        print(f"[SILVER] Exemple CLEAN (1ère ligne): {rows[0]}")

    bulk_upsert_clean(rows)
    print(f"[SILVER] ✅ Upsert terminé dans `reviews_clean` (schéma plat).")
    return dt
