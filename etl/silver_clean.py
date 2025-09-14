# etl/silver_clean.py
import io
import json
from typing import List, Dict, Any, Iterable
from datetime import datetime, timezone

import pandas as pd

from etl.gcp_clients import get_storage_client, get_firestore_client
from etl.config import Config
from etl.text_utils import clean_text, detect_lang, sentiment_scores  # déjà existants

# -------------------------------------------------------------------
# Helpers GCS
# -------------------------------------------------------------------

def _iter_gcs_ndjson(bucket_name: str, prefix: str) -> Iterable[Dict[str, Any]]:
    """
    Lit tous les blobs GCS sous `prefix` et yield chaque ligne NDJSON -> dict.
    On tolère d'autres fichiers mais on ne lit que le contenu texte.
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


def _read_raw_all(app_id: str, dt_str: str) -> pd.DataFrame:
    """
    Charge le RAW Bronze depuis GCS (NDJSON) pour un app_id/date.
    Chemin attendu: bronze/raw/app_id=<ID>/dt=<YYYY-MM-DD>/...
    """
    bucket = Config.gcs_bucket
    if not bucket:
        raise RuntimeError("GCS_BUCKET non défini (Config.gcs_bucket).")

    prefix = f"bronze/raw/app_id={app_id}/dt={dt_str}/"
    rows = list(_iter_gcs_ndjson(bucket, prefix))
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)

# -------------------------------------------------------------------
# Normalisation & enrichissements
# -------------------------------------------------------------------

# Mapping minimal codes -> libellés lisibles (complète au besoin)
LANG_LABELS = {
    "en": "English",
    "fr": "Français",
    "de": "Deutsch",
    "es": "Español",
    "it": "Italiano",
    "pt": "Português",
    "ru": "Русский",
    "zh": "中文",
    "ja": "日本語",
    "ko": "한국어",
}

def _standardize_ids(df: pd.DataFrame, app_id: str) -> pd.DataFrame:
    """
    - garantit `app_id` & `review_id`
    - `review_id` depuis `recommendationid` sinon fabrique <app_id>_<i>
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
    - force `timestamp_created` / `timestamp_updated` en int (epoch s)
    - dérive `review_date` (YYYY-MM-DD UTC) depuis `timestamp_created`
    """
    df = df.copy()
    for c in ("timestamp_created", "timestamp_updated"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        else:
            df[c] = 0

    # Review date lisible
    df["review_date"] = (
        pd.to_datetime(df["timestamp_created"], unit="s", utc=True, errors="coerce")
        .dt.date.astype(str)
        .fillna("")
    )
    return df


def _prep_text_and_language(df: pd.DataFrame) -> pd.DataFrame:
    """
    - choisit colonne de texte -> `review_text`
    - `cleaned_review` via clean_text()
    - `language` code ISO minuscule (fallback detect_lang)
    - `language_label` libellé lisible
    """
    df = df.copy()

    # 1) texte source -> review_text
    text_col = None
    for cand in ("review", "review_text", "text", "cleaned_review"):
        if cand in df.columns:
            text_col = cand
            break
    df["review_text"] = df[text_col].fillna("").astype(str) if text_col else ""

    # 2) cleaned
    df["cleaned_review"] = df["review_text"].map(clean_text)

    # 3) language code
    if "language" not in df.columns:
        df["language"] = ""

    df["language"] = (
        df["language"].fillna("").astype(str).str.strip().str.lower()
    )

    need_detect = df["language"].eq("") | df["language"].eq("unknown")
    if need_detect.any():
        df.loc[need_detect, "language"] = df.loc[need_detect, "cleaned_review"].map(detect_lang).fillna("")

    # sécurise: garde un code court (ex langdetect peut renvoyer 'en')
    df["language"] = df["language"].str[:5].str.lower().replace("", "unknown")

    # 4) label lisible
    df["language_label"] = df["language"].map(LANG_LABELS).fillna(df["language"].str.upper())

    return df


def _prep_playtime_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule `playtime_hours` depuis les champs auteur si présents :
    - author.playtime_at_review
    - author.playtime_forever
    - author.playtime_last_two_weeks
    Valeurs en minutes -> heures (float, 2 décimales).
    """
    df = df.copy()

    # déjà à plat ?
    # sinon lire dict `author`
    # On essaie d'abord les colonnes "aplaties" si elles existent.
    cols_flat = {
        "author_playtime_at_review",
        "author_playtime_forever",
        "author_playtime_last_two_weeks",
    }

    author_dict_possible = "author" in df.columns

    def _extract_minutes(row) -> float:
        # priorité: at_review > forever > last_two_weeks
        # renvoie minutes (float) ou NaN
        def _to_num(x):
            try:
                return float(x)
            except Exception:
                return float("nan")

        # a) colonnes aplat ies
        if any(c in df.columns for c in cols_flat):
            ar = _to_num(row.get("author_playtime_at_review"))
            if pd.notna(ar):
                return ar
            fr = _to_num(row.get("author_playtime_forever"))
            if pd.notna(fr):
                return fr
            tw = _to_num(row.get("author_playtime_last_two_weeks"))
            if pd.notna(tw):
                return tw

        # b) dict author
        if author_dict_possible:
            author = row.get("author")
            if isinstance(author, dict):
                ar = _to_num(author.get("playtime_at_review"))
                if pd.notna(ar):
                    return ar
                fr = _to_num(author.get("playtime_forever"))
                if pd.notna(fr):
                    return fr
                tw = _to_num(author.get("playtime_last_two_weeks"))
                if pd.notna(tw):
                    return tw

        return float("nan")

    minutes = df.apply(_extract_minutes, axis=1)
    hours = (pd.to_numeric(minutes, errors="coerce") / 60.0).round(2)
    df["playtime_hours"] = hours

    return df


def _prep_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule `compound` (float) + `sentiment` (neg/neu/pos).
    Si `compound` existe déjà, on le garde (en coerce).
    """
    df = df.copy()

    if "compound" in df.columns:
        df["compound"] = pd.to_numeric(df["compound"], errors="coerce").fillna(0.0)
    else:
        scores = df["cleaned_review"].map(sentiment_scores)
        df["compound"] = [s.get("compound", 0.0) for s in scores]

    df["sentiment"] = pd.cut(
        df["compound"],
        bins=[-1.0, -0.05, 0.05, 1.0],
        labels=["neg", "neu", "pos"],
        include_lowest=True,
    ).astype(str)

    return df


def _select_and_fill_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit le jeu de colonnes final aligné à ton schéma `reviews_clean`.
    On garde aussi `playtime_hours` & `language_label` utiles pour le front.
    """
    keep = [
        "app_id",
        "review_id",
        "cleaned_review",
        "compound",
        "sentiment",
        "language",
        "language_label",
        "review_date",
        "timestamp_created",
        "timestamp_updated",
        "voted_up",
        "votes_funny",
        "votes_up",
        "weighted_vote_score",
        "playtime_hours",
        # on laisse `review_text` pour debug/usage éventuel
        "review_text",
    ]
    out = df.reindex(columns=keep, fill_value=None).copy()

    # types de base
    for c in ("voted_up",):
        if c in out.columns:
            out[c] = out[c].astype("boolean")

    for c in ("votes_funny", "votes_up"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    if "weighted_vote_score" in out.columns:
        # ce champ est parfois une string numérique
        out["weighted_vote_score"] = out["weighted_vote_score"].astype(str)

    # review_date toujours string YYYY-MM-DD (déjà géré)
    out["review_date"] = out["review_date"].fillna("")

    return out

# -------------------------------------------------------------------
# Firestore
# -------------------------------------------------------------------

def _firestore_upsert_rows(app_id: str, rows: List[Dict[str, Any]]) -> int:
    """
    Upsert en batch dans Firestore:
    collection reviews_clean / document <app_id> / subcollection items / doc <review_id>
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

# -------------------------------------------------------------------
# Entrée principale
# -------------------------------------------------------------------

def to_silver(app_id: str, dt_str: str, for_airflow: bool = False) -> str:
    """
    Lit le RAW depuis GCS, clean/normalise, et upsert le CLEAN dans Firestore.
    Retourne la date traitée (dt_str).
    """
    print(f"[SILVER] start app_id={app_id} dt={dt_str}")

    df = _read_raw_all(app_id, dt_str)
    if df.empty:
        print(f"[SILVER] no RAW for app_id={app_id} dt={dt_str} → skip")
        return dt_str

    df = _standardize_ids(df, app_id)
    df = _prep_timestamps(df)
    df = _prep_text_and_language(df)
    df = _prep_playtime_hours(df)
    df = _prep_sentiment(df)
    df = _select_and_fill_columns(df)

    rows = df.to_dict("records")
    wrote = _firestore_upsert_rows(app_id, rows)

    print(f"[SILVER] upserted {wrote} rows to reviews_clean/{app_id}/items (dt={dt_str})")
    return dt_str
