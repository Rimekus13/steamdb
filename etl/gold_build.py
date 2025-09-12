# etl/gold_build.py
import pandas as pd
from datetime import datetime
from typing import List, Iterable, Tuple, Optional

from .text_utils import tokenize_no_stop
# 🔧 On s'appuie sur tes helpers Firestore (comme pour silver)
#  - col_clean_query() doit retourner une liste de dicts "clean" (reviews_clean)
#  - replace_collection(name, records) remplace complètement la collection cible
from .firestore_utils import col_clean_query, replace_collection

WINDOW = 5                    # taille de fenêtre de cooccurrence (glissante)
TOP_K: Optional[int] = None   # None = pas de coupe ; sinon limite le nb de lignes insérées


def _period_from_row(row) -> Optional[str]:
    """
    Période mensuelle :
      - priorité à review_date (YYYY-MM-DD) → YYYY-MM
      - sinon fallback timestamp_created (epoch sec) → YYYY-MM
    """
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
    """
    Fenêtre glissante : pour chaque i, on prend les tokens j dans (i, i+window).
    (a,b) est normalisé (ordre trié) pour éviter les doublons (a,b)/(b,a).
    """
    n = len(tokens)
    for i in range(n):
        a = tokens[i]
        jmax = min(n, i + window)
        for j in range(i + 1, jmax):
            b = tokens[j]
            if a != b:
                yield tuple(sorted((a, b)))


def _read_clean_all() -> pd.DataFrame:
    """
    Lit l'intégralité de la source CLEAN (Firestore → reviews_clean/items) via utilitaire.
    On s’attend à trouver au minimum : app_id, cleaned_review, review_date|timestamp_created.
    """
    rows = col_clean_query()
    return pd.DataFrame(rows)


def build_gold(app_ids: Optional[List[str]] = None, for_airflow: bool = False) -> None:
    """
    Calcule les tables Gold et remplace complètement les collections Firestore :
      - cooccurrences_counts: {app_id, token_a, token_b, window, count, period}
      - cooccurrences_percent: {app_id, token_a, token_b, window, percent, period}

    Si app_ids est fourni, on filtre le calcul à ces jeux ; sinon on prend tous les app_id présents en CLEAN.
    """
    df = _read_clean_all()
    if df.empty or "cleaned_review" not in df.columns:
        replace_collection("cooccurrences_counts", [])
        replace_collection("cooccurrences_percent", [])
        return

    # Normalisations de base
    if "app_id" not in df.columns:
        # si ta couche CLEAN ne force pas app_id, on met une valeur vide pour filtrer ensuite
        df["app_id"] = ""
    df["app_id"] = df["app_id"].astype(str)

    # Tokenisation + période mensuelle
    df["tokens"] = df["cleaned_review"].fillna("").map(tokenize_no_stop)
    df["period"] = df.apply(_period_from_row, axis=1)

    # Filtrages
    df = df[(df["tokens"].map(len) >= 2) & df["period"].notna() & df["app_id"].ne("")]
    if df.empty:
        replace_collection("cooccurrences_counts", [])
        replace_collection("cooccurrences_percent", [])
        return

    # Filtre optionnel par app_ids
    if app_ids:
        allow = {str(a) for a in app_ids}
        df = df[df["app_id"].isin(allow)]
        if df.empty:
            replace_collection("cooccurrences_counts", [])
            replace_collection("cooccurrences_percent", [])
            return

    # Génération des paires (ligne → paires dans fenêtre), puis agrégation globale
    records = []
    for _, row in df.iterrows():
        app_id = row["app_id"]
        period = row["period"]
        toks = row["tokens"]
        for a, b in _pairs_within_window(toks, window=WINDOW):
            records.append((app_id, period, a, b, 1))

    if not records:
        replace_collection("cooccurrences_counts", [])
        replace_collection("cooccurrences_percent", [])
        return

    rec_df = pd.DataFrame(records, columns=["app_id", "period", "token_a", "token_b", "count"])
    co_counts = (rec_df
                 .groupby(["app_id", "period", "token_a", "token_b"], as_index=False)["count"]
                 .sum())

    # Totaux par (app_id, period) pour le pourcentage
    totals = (co_counts
              .groupby(["app_id", "period"], as_index=False)["count"]
              .sum()
              .rename(columns={"count": "total_pairs"}))

    co_pct = co_counts.merge(totals, on=["app_id", "period"], how="left")
    co_pct["percent"] = (co_pct["count"] / co_pct["total_pairs"]).fillna(0.0)
    co_pct = co_pct.drop(columns=["total_pairs"])

    # Coupe optionnelle
    if TOP_K is not None and len(co_counts) > TOP_K:
        co_counts = co_counts.nlargest(TOP_K, "count")
    if TOP_K is not None and len(co_pct) > TOP_K:
        co_pct = co_pct.nlargest(TOP_K, "percent")

    # Ajoute window + ordre colonnes
    co_counts["window"] = WINDOW
    co_pct["window"] = WINDOW

    co_counts = co_counts[["app_id", "token_a", "token_b", "window", "count", "period"]]
    co_pct    = co_pct   [["app_id", "token_a", "token_b", "window", "percent", "period"]]

    # Remplacement complet des collections Firestore (idempotent)
    replace_collection("cooccurrences_counts", co_counts.to_dict("records"))
    replace_collection("cooccurrences_percent", co_pct.to_dict("records"))
