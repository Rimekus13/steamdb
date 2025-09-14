# etl/gold_build.py
import logging
import pandas as pd
from datetime import datetime
from typing import List, Iterable, Tuple, Optional

from .text_utils import tokenize_no_stop
from .firestore_utils import col_clean_query, replace_collection

LOG = logging.getLogger(__name__)

WINDOW = 5                    # taille de fenêtre de cooccurrence (glissante)
TOP_K: Optional[int] = None   # None = pas de coupe

def _period_from_row(row) -> Optional[str]:
    rd = row.get("review_date")
    if isinstance(rd, str) and len(rd) >= 7:
        return rd[:7]  # 'YYYY-MM'
    ts = row.get("timestamp_created")
    if pd.notna(ts):
        try:
            return datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m")
        except Exception:
            pass
    return None

def _pairs_within_window(tokens: List[str], window: int = WINDOW):
    n = len(tokens)
    for i in range(n):
        a = tokens[i]
        jmax = min(n, i + window)
        for j in range(i + 1, jmax):
            b = tokens[j]
            if a != b:
                yield tuple(sorted((a, b)))

def _read_clean_all() -> pd.DataFrame:
    LOG.info("[GOLD] Lecture reviews_clean…")
    rows = col_clean_query()
    df = pd.DataFrame(rows)
    LOG.info("[GOLD] Lignes chargées depuis CLEAN: %d", len(df))
    return df

def build_gold(app_ids: Optional[List[str]] = None, for_airflow: bool = False) -> None:
    """
    Construit:
      - cooccurrences_counts(app_id, token_a, token_b, window, count, period)
      - cooccurrences_percent(app_id, token_a, token_b, window, percent, period)
    """
    df = _read_clean_all()
    if df.empty or "cleaned_review" not in df.columns:
        LOG.info("[GOLD] CLEAN vide ou colonne `cleaned_review` absente → purge des cooccurrences et sortie.")
        replace_collection("cooccurrences_counts", [])
        replace_collection("cooccurrences_percent", [])
        return

    if "app_id" not in df.columns:
        df["app_id"] = ""
    df["app_id"] = df["app_id"].astype(str)

    df["tokens"] = df["cleaned_review"].fillna("").map(tokenize_no_stop)
    df["period"] = df.apply(_period_from_row, axis=1)

    df = df[(df["tokens"].map(len) >= 2) & df["period"].notna() & df["app_id"].ne("")]
    LOG.info("[GOLD] Lignes après filtrages (tokens>=2, period, app_id): %d", len(df))
    if df.empty:
        replace_collection("cooccurrences_counts", [])
        replace_collection("cooccurrences_percent", [])
        return

    if app_ids:
        allow = {str(a) for a in app_ids}
        before = len(df)
        df = df[df["app_id"].isin(allow)]
        LOG.info("[GOLD] Filtre app_ids: %d → %d", before, len(df))
        if df.empty:
            replace_collection("cooccurrences_counts", [])
            replace_collection("cooccurrences_percent", [])
            return

    records = []
    for _, row in df.iterrows():
        app_id = row["app_id"]
        period = row["period"]
        for a, b in _pairs_within_window(row["tokens"], window=WINDOW):
            records.append((app_id, period, a, b, 1))

    if not records:
        LOG.info("[GOLD] Aucune paire générée → purge des tables et sortie.")
        replace_collection("cooccurrences_counts", [])
        replace_collection("cooccurrences_percent", [])
        return

    rec_df = pd.DataFrame(records, columns=["app_id", "period", "token_a", "token_b", "count"])
    co_counts = (rec_df
                 .groupby(["app_id", "period", "token_a", "token_b"], as_index=False)["count"]
                 .sum())
    LOG.info("[GOLD] co_counts: %d lignes agrégées", len(co_counts))

    totals = (co_counts
              .groupby(["app_id", "period"], as_index=False)["count"]
              .sum()
              .rename(columns={"count": "total_pairs"}))
    co_pct = co_counts.merge(totals, on=["app_id", "period"], how="left")
    co_pct["percent"] = (co_pct["count"] / co_pct["total_pairs"]).fillna(0.0)
    co_pct = co_pct.drop(columns=["total_pairs"])
    LOG.info("[GOLD] co_pct: %d lignes", len(co_pct))

    if TOP_K is not None and len(co_counts) > TOP_K:
        co_counts = co_counts.nlargest(TOP_K, "count")
    if TOP_K is not None and len(co_pct) > TOP_K:
        co_pct = co_pct.nlargest(TOP_K, "percent")

    co_counts["window"] = WINDOW
    co_pct["window"] = WINDOW

    co_counts = co_counts[["app_id", "token_a", "token_b", "window", "count", "period"]]
    co_pct    = co_pct   [["app_id", "token_a", "token_b", "window", "percent", "period"]]

    LOG.info("[GOLD] Remplacement Firestore: cooccurrences_counts (%d) / cooccurrences_percent (%d)", len(co_counts), len(co_pct))
    replace_collection("cooccurrences_counts", co_counts.to_dict("records"))
    replace_collection("cooccurrences_percent", co_pct.to_dict("records"))
    LOG.info("[GOLD] ✅ Terminé.")
