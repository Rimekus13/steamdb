
import pandas as pd
from collections import Counter
from itertools import combinations
from datetime import datetime
from typing import List, Iterable, Tuple

from .mongo_utils import col_clean, replace_collection
from .text_utils import tokenize_no_stop

WINDOW = 5  # taille de fenêtre de cooccurrence (glissante)
TOP_K = None  # None = pas de coupe, sinon limiter le nombre de lignes

def _read_clean_all(for_airflow: bool = False) -> pd.DataFrame:
    # On récupère aussi review_date / timestamp_created pour calculer "period"
    cur = col_clean(for_airflow).find({}, {"_id": 0, "app_id": 1, "cleaned_review": 1, "review_date": 1, "timestamp_created": 1})
    return pd.DataFrame(list(cur))

def _period_from_row(row) -> str:
    # Préférence: review_date (YYYY-MM-DD), fallback: timestamp_created (epoch sec)
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

def _pairs_within_window(tokens: List[str], window: int = WINDOW) -> Iterable[Tuple[str, str]]:
    # Fenêtre glissante: pour chaque position, on regarde les w-1 suivants
    n = len(tokens)
    for i in range(n):
        a = tokens[i]
        jmax = min(n, i + window)
        for j in range(i + 1, jmax):
            b = tokens[j]
            if a != b:
                # Normaliser l'ordre afin que (a,b) == (b,a)
                yield tuple(sorted((a, b)))

def build_gold(for_airflow: bool = False):
    """
    Produit deux collections:
      - cooccurrences_counts: {app_id, token_a, token_b, period, window, count}
      - cooccurrences_percent: {app_id, token_a, token_b, period, window, percent}
    Logique:
      - cooccurrences calculées par "app_id" ET "period" (YYYY-MM)
      - fenêtre glissante de taille WINDOW (par défaut 5)
      - percent = count / total_pairs(app_id, period)
    Remplace complètement les collections (idempotent) via replace_collection (batch rebuild).
    """
    df = _read_clean_all(for_airflow=for_airflow)
    if df.empty or "cleaned_review" not in df:
        replace_collection("cooccurrences_counts", [], for_airflow=for_airflow)
        replace_collection("cooccurrences_percent", [], for_airflow=for_airflow)
        return

    # Tokenisation + period par ligne
    df["tokens"] = df["cleaned_review"].fillna("").map(tokenize_no_stop)
    df["period"] = df.apply(_period_from_row, axis=1)
    df = df[(df["tokens"].map(len) >= 2) & df["period"].notna() & df["app_id"].notna()]

    if df.empty:
        replace_collection("cooccurrences_counts", [], for_airflow=for_airflow)
        replace_collection("cooccurrences_percent", [], for_airflow=for_airflow)
        return

    # Calcul des cooccurrences par ligne (fenêtre glissante), puis agrégation par (app_id, period, pair)
    records = []
    for _, row in df.iterrows():
        app_id = str(row["app_id"])  # normalisation str pour cohérence
        period = row["period"]
        toks = row["tokens"]
        for a, b in _pairs_within_window(toks, window=WINDOW):
            records.append((app_id, period, a, b, 1))

    if not records:
        replace_collection("cooccurrences_counts", [], for_airflow=for_airflow)
        replace_collection("cooccurrences_percent", [], for_airflow=for_airflow)
        return

    rec_df = pd.DataFrame(records, columns=["app_id", "period", "token_a", "token_b", "count"])
    co_counts = rec_df.groupby(["app_id", "period", "token_a", "token_b"], as_index=False)["count"].sum()

    # Total des paires par (app_id, period)
    totals = co_counts.groupby(["app_id", "period"], as_index=False)["count"].sum().rename(columns={"count": "total_pairs"})
    co_pct = co_counts.merge(totals, on=["app_id", "period"], how="left")
    co_pct["percent"] = (co_pct["count"] / co_pct["total_pairs"]).fillna(0.0)
    co_pct = co_pct.drop(columns=["total_pairs"])

    # Optionnel: limiter le TOP_K global (pour éviter des collections énormes)
    if TOP_K is not None and len(co_counts) > TOP_K:
        co_counts = co_counts.nlargest(TOP_K, "count")
    if TOP_K is not None and len(co_pct) > TOP_K:
        co_pct = co_pct.nlargest(TOP_K, "percent")

    # Ajouter la colonne window
    co_counts["window"] = WINDOW
    co_pct["window"] = WINDOW

    # Ordonner les colonnes pour lisibilité
    co_counts = co_counts[["app_id", "token_a", "token_b", "window", "count", "period"]]
    co_pct = co_pct[["app_id", "token_a", "token_b", "window", "percent", "period"]]

    # Remplacement complet via utilitaire existant (idempotent au niveau batch)
    replace_collection("cooccurrences_counts", co_counts.to_dict("records"), for_airflow=for_airflow)
    replace_collection("cooccurrences_percent", co_pct.to_dict("records"), for_airflow=for_airflow)
