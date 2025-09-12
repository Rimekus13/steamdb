# etl/gold_build.py
from __future__ import annotations

import pandas as pd
from datetime import datetime
from typing import List, Iterable, Tuple, Optional, Dict
from collections import defaultdict

from .text_utils import tokenize_no_stop
from .firestore_utils import col_clean_query, replace_collection

# ===== Réglages =====
WINDOW = 5                   # fenêtre de cooccurrence (glissante)
TOP_K: Optional[int] = None  # None = pas de coupe ; sinon limite par groupe (app_id, period)
FIELD_CLEAN = ("cleaned_review", "cleaned_text", "review_clean")  # champs possibles du SILVER


def _period_from_row(row: Dict) -> Optional[str]:
    """Prend review_date (YYYY-MM-DD) si dispo, sinon timestamp_created → YYYY-MM."""
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
                yield a if a < b else b, b if a < b else a


def _read_clean_all() -> pd.DataFrame:
    """
    Lit la collection Firestore 'reviews_clean' (via util fourni) et renvoie un DataFrame.
    col_clean_query() doit renvoyer un iterable de dicts :
      { app_id: "...", review_date: "...", timestamp_created: ..., cleaned_review|cleaned_text: "..." , ...}
    """
    rows = list(col_clean_query())  # si gros volume, on peut streamer par batch et agréger (voir NOTE plus bas)
    return pd.DataFrame(rows)


def _choose_clean_text(row: Dict) -> str:
    for k in FIELD_CLEAN:
        v = row.get(k)
        if isinstance(v, str):
            return v
    return ""


def build_gold(app_ids: Optional[List[str]] = None, for_airflow: bool = False) -> None:
    """
    Calcule les tables Gold et remplace les collections Firestore :
      - cooccurrences_counts:  {app_id, token_a, token_b, window, count,  period}
      - cooccurrences_percent: {app_id, token_a, token_b, window, percent, period}

    Si app_ids est fourni, filtre sur ces jeux. Sinon, prend tout le SILVER.
    """
    df = _read_clean_all()
    if df.empty:
        replace_collection("cooccurrences_counts", [])
        replace_collection("cooccurrences_percent", [])
        return

    # Normalisations de base
    if "app_id" not in df.columns:
        df["app_id"] = ""

    # sélection du champ de texte nettoyé (supporte plusieurs noms)
    if not any(c in df.columns for c in FIELD_CLEAN):
        # rien à traiter
        replace_collection("cooccurrences_counts", [])
        replace_collection("cooccurrences_percent", [])
        return

    # Construire tokens + period de façon robuste
    # (on évite apply python sur toute la DF si très grande : on passe par itertuples)
    app_ids_allow = set(str(a) for a in app_ids) if app_ids else None

    # Compteurs groupés : (app_id, period, token_a, token_b) -> count
    counts = defaultdict(int)
    totals_per_group = defaultdict(int)  # (app_id, period) -> total pairs

    for row in df.to_dict("records"):
        aid = str(row.get("app_id", "") or "")
        if app_ids_allow is not None and aid not in app_ids_allow:
            continue

        period = _period_from_row(row)
        if not period or not aid:
            continue

        txt = _choose_clean_text(row)
        toks = tokenize_no_stop((txt or "").strip())
        if len(toks) < 2:
            continue

        # pairs
        n_this = 0
        for a, b in _pairs_within_window(toks, window=WINDOW):
            counts[(aid, period, a, b)] += 1
            n_this += 1
        totals_per_group[(aid, period)] += n_this

    if not counts:
        replace_collection("cooccurrences_counts", [])
        replace_collection("cooccurrences_percent", [])
        return

    # Conversion en DataFrame
    rec_df = pd.DataFrame(
        [(k[0], k[1], k[2], k[3], v) for k, v in counts.items()],
        columns=["app_id", "period", "token_a", "token_b", "count"]
    )

    # TOP-K par (app_id, period) si demandé
    if TOP_K is not None:
        rec_df = (rec_df.sort_values(["app_id", "period", "count"], ascending=[True, True, False])
                        .groupby(["app_id", "period"], as_index=False, group_keys=False)
                        .head(TOP_K))

    # Percent
    rec_df["window"] = WINDOW
    # Map totals
    rec_df["total_pairs"] = rec_df.apply(
        lambda r: totals_per_group.get((r["app_id"], r["period"]), 0),
        axis=1
    )
    rec_df["percent"] = (rec_df["count"] / rec_df["total_pairs"]).fillna(0.0)

    # Ordonner colonnes pour export
    co_counts = rec_df[["app_id", "token_a", "token_b", "window", "count", "period"]].copy()
    co_pct    = rec_df[["app_id", "token_a", "token_b", "window", "percent", "period"]].copy()

    # Ecriture Firestore (remplacement total)
    # NOTE: replace_collection doit gérer l'effacement + insertion batch (limites ~500 ops/batch).
    replace_collection("cooccurrences_counts", co_counts.to_dict("records"))
    replace_collection("cooccurrences_percent", co_pct.to_dict("records"))
