# etl/gold_build.py
import pandas as pd
from collections import Counter
from datetime import datetime
from typing import Iterable, Tuple, List, Optional

from .mongo_utils import col_clean, col_co_counts, col_co_percent
from .text_utils import tokenize_no_stop

WINDOW = 5         # taille de fenêtre de cooccurrence (glissante)
TOP_K: Optional[int] = None  # None = pas de coupe ; sinon limite le nb de lignes insérées


def _period_from_row(row) -> str:
    """Prend review_date si dispo (YYYY-MM-DD) sinon fallback sur timestamp_created → YYYY-MM."""
    rd = row.get("review_date")
    if isinstance(rd, str) and len(rd) >= 7:
        return rd[:7]
    ts = row.get("timestamp_created")
    if pd.notna(ts):
        try:
            return datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m")
        except Exception:
            pass
    return None


def _pairs_within_window(tokens: List[str], window: int = WINDOW) -> Iterable[Tuple[str, str]]:
    """Fenêtre glissante : pour chaque i, on prend les tokens j dans (i, i+window)."""
    n = len(tokens)
    for i in range(n):
        a = tokens[i]
        jmax = min(n, i + window)
        for j in range(i + 1, jmax):
            b = tokens[j]
            if a != b:
                # ordre normalisé pour éviter (a,b) / (b,a) en double
                yield tuple(sorted((a, b)))


def _build_for_one_app(app_id: str, for_airflow: bool = False) -> None:
    """Construit counts et percent pour un app_id donné, toutes périodes confondues (YYYY-MM)."""
    cur = col_clean(for_airflow).find(
        {"app_id": str(app_id)},
        {"_id": 0, "cleaned_review": 1, "review_date": 1, "timestamp_created": 1, "app_id": 1}
    )
    df = pd.DataFrame(list(cur))
    if df.empty or "cleaned_review" not in df:
        # purge sélective (aucune data pour cet app)
        col_co_counts(for_airflow).delete_many({"app_id": str(app_id)})
        col_co_percent(for_airflow).delete_many({"app_id": str(app_id)})
        return

    # Tokenisation + période
    df["tokens"] = df["cleaned_review"].fillna("").map(tokenize_no_stop)
    df["period"] = df.apply(_period_from_row, axis=1)
    df = df[(df["tokens"].map(len) >= 2) & df["period"].notna()]

    if df.empty:
        col_co_counts(for_airflow).delete_many({"app_id": str(app_id)})
        col_co_percent(for_airflow).delete_many({"app_id": str(app_id)})
        return

    # Génération des paires (par ligne), agrégation par (period, token_a, token_b)
    records = []
    for _, row in df.iterrows():
        period = row["period"]
        toks = row["tokens"]
        for a, b in _pairs_within_window(toks, window=WINDOW):
            records.append((period, a, b, 1))

    if not records:
        col_co_counts(for_airflow).delete_many({"app_id": str(app_id)})
        col_co_percent(for_airflow).delete_many({"app_id": str(app_id)})
        return

    rec_df = pd.DataFrame(records, columns=["period", "token_a", "token_b", "count"])
    co_counts = rec_df.groupby(["period", "token_a", "token_b"], as_index=False)["count"].sum()
    co_counts.insert(0, "app_id", str(app_id))

    # Totaux par période (pour calcul de percent)
    totals = co_counts.groupby(["period"], as_index=False)["count"].sum().rename(columns={"count": "total_pairs"})
    co_pct = co_counts.merge(totals, on=["period"], how="left")
    co_pct["percent"] = (co_pct["count"] / co_pct["total_pairs"]).fillna(0.0)
    co_pct = co_pct.drop(columns=["total_pairs"])

    # Coupe optionnelle
    if TOP_K is not None and len(co_counts) > TOP_K:
        co_counts = co_counts.nlargest(TOP_K, "count")
    if TOP_K is not None and len(co_pct) > TOP_K:
        co_pct = co_pct.nlargest(TOP_K, "percent")

    # Ajoute window et réordonne
    co_counts["window"] = WINDOW
    co_pct["window"] = WINDOW

    co_counts = co_counts[["app_id", "token_a", "token_b", "window", "count", "period"]]
    co_pct    = co_pct   [["app_id", "token_a", "token_b", "window", "percent", "period"]]

    # Remplacement SÉLECTIF (uniquement l'app courant)
    col_co_counts(for_airflow).delete_many({"app_id": str(app_id)})
    col_co_percent(for_airflow).delete_many({"app_id": str(app_id)})

    if not co_counts.empty:
        col_co_counts(for_airflow).insert_many(co_counts.to_dict("records"), ordered=False)
    if not co_pct.empty:
        col_co_percent(for_airflow).insert_many(co_pct.to_dict("records"), ordered=False)


def build_gold(app_ids: Optional[List[str]] = None, for_airflow: bool = False):
    """
    Calcul Gold. Si app_ids est None → on récupère la liste des app_id présents en SILVER.
    Sinon on traite seulement la liste fournie (recommandé en CI pour du multi-jeux).
    """
    if app_ids is None:
        # détecte les app_ids existants en SILVER
        cur = col_clean(for_airflow).distinct("app_id")
        app_ids = [str(a) for a in cur if a is not None]

    for app in app_ids:
        _build_for_one_app(str(app), for_airflow=for_airflow)
