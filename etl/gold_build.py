# etl/gold_build.py
from __future__ import annotations

import os
from datetime import datetime
from typing import List, Iterable, Tuple, Optional

import pandas as pd

from .text_utils import tokenize_no_stop
from .firestore_utils import col_clean_query, replace_collection

# =========================
# Paramètres (surchargés par env)
# =========================
# Fenêtre par défaut (nombre de tokens "en avant" pris en compte)
DEFAULT_WINDOW = 5
# Période d’agrégation: "month" (YYYY-MM) ou "day" (YYYY-MM-DD)
DEFAULT_PERIOD = "month"
# Coupe optionnelle pour limiter le volume écrit
DEFAULT_TOP_K: Optional[int] = None  # None = pas de coupe

def _env_window() -> int:
    val = os.getenv("COOC_WINDOW", "").strip()
    if not val:
        # compat: si on t’a mis COOC_WINDOWS="2,5" on prend le premier
        many = os.getenv("COOC_WINDOWS", "").strip()
        if many:
            try:
                return int(many.split(",")[0])
            except Exception:
                return DEFAULT_WINDOW
        return DEFAULT_WINDOW
    try:
        return int(val)
    except Exception:
        return DEFAULT_WINDOW

def _env_period() -> str:
    p = (os.getenv("COOC_PERIOD", DEFAULT_PERIOD) or DEFAULT_PERIOD).strip().lower()
    return p if p in ("month", "day") else DEFAULT_PERIOD

def _env_topk() -> Optional[int]:
    val = os.getenv("COOC_TOPK", "").strip()
    if not val:
        return DEFAULT_TOP_K
    try:
        x = int(val)
        return x if x > 0 else None
    except Exception:
        return DEFAULT_TOP_K


# =========================
# Helpers période & cooccurrences
# =========================
def _period_from_row(row, period: str) -> Optional[str]:
    """
    Période:
      - "month" → "YYYY-MM"
      - "day"   → "YYYY-MM-DD"
    Priorité à review_date (str), sinon fallback sur timestamp_created (epoch sec).
    """
    if period == "day":
        # review_date est déjà "YYYY-MM-DD" dans le silver
        rd = row.get("review_date")
        if isinstance(rd, str) and len(rd) >= 10:
            return rd[:10]
    else:
        # month
        rd = row.get("review_date")
        if isinstance(rd, str) and len(rd) >= 7:
            return rd[:7]

    ts = row.get("timestamp_created")
    if pd.notna(ts):
        try:
            dt = datetime.utcfromtimestamp(int(ts))
            return dt.strftime("%Y-%m-%d") if period == "day" else dt.strftime("%Y-%m")
        except Exception:
            pass
    return None


def _pairs_within_window(tokens: List[str], window: int) -> Iterable[Tuple[str, str]]:
    """
    Fenêtre glissante **en avant**:
      pour i, on considère j ∈ [i+1, i+window] (inclus si dispo).
      On renvoie toujours (a, b) trié (ordre canonique) et a != b.
    """
    n = len(tokens)
    for i in range(n):
        a = tokens[i]
        if not a:
            continue
        jmax = min(n, i + window + 1)  # +1 pour inclure i+window
        for j in range(i + 1, jmax):
            b = tokens[j]
            if not b or a == b:
                continue
            yield (a, b) if a <= b else (b, a)


# =========================
# Lecture CLEAN
# =========================
def _read_clean_all() -> pd.DataFrame:
    """
    Lit l’intégralité de reviews_clean via le helper Firestore.
    Colonnes attendues (au minimum) :
      - app_id (str), cleaned_review (str), review_date (str) ou timestamp_created (int)
    """
    rows = col_clean_query()
    if not rows:
        return pd.DataFrame(columns=[
            "app_id", "cleaned_review", "review_date", "timestamp_created"
        ])
    df = pd.DataFrame(rows)
    # normalisations minimales
    if "app_id" not in df.columns:
        df["app_id"] = ""
    df["app_id"] = df["app_id"].astype(str)
    if "cleaned_review" not in df.columns:
        df["cleaned_review"] = ""
    return df


# =========================
# Build GOLD
# =========================
def build_gold(app_ids: Optional[List[str]] = None, for_airflow: bool = False) -> None:
    """
    Construit les tables GOLD et **remplace** complètement les collections Firestore :
      - cooccurrences_counts(app_id, token_a, token_b, window, count,  period)
      - cooccurrences_percent(app_id, token_a, token_b, window, percent, period)  <-- % (0..100)

    Si `app_ids` est fourni, filtre le calcul à ces jeux.
    Paramètres pilotables par ENV:
      - COOC_WINDOW      (ex: 5)
      - COOC_WINDOWS     (ex: "2,5" → on prendra le premier si COOC_WINDOW non défini)
      - COOC_PERIOD      ("month" | "day")
      - COOC_TOPK        (ex: 50000)
    """
    df = _read_clean_all()
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

    # Paramètres
    window = _env_window()
    period = _env_period()
    topk   = _env_topk()

    # Tokenisation & période
    df["tokens"] = df["cleaned_review"].fillna("").map(tokenize_no_stop)
    df["period"] = df.apply(lambda r: _period_from_row(r, period=period), axis=1)

    # Filtrages minimaux
    df = df[(df["app_id"].ne("")) & df["period"].notna() & (df["tokens"].map(len) >= 2)]
    if df.empty:
        replace_collection("cooccurrences_counts", [])
        replace_collection("cooccurrences_percent", [])
        return

    # Génération des paires par ligne
    recs = []
    for _, row in df.iterrows():
        app_id = row["app_id"]
        pkey   = row["period"]
        for a, b in _pairs_within_window(row["tokens"], window=window):
            recs.append((app_id, pkey, a, b, 1))

    if not recs:
        replace_collection("cooccurrences_counts", [])
        replace_collection("cooccurrences_percent", [])
        return

    # Agrégation des counts
    counts = pd.DataFrame(recs, columns=["app_id", "period", "token_a", "token_b", "count"])
    counts = (counts
              .groupby(["app_id", "period", "token_a", "token_b"], as_index=False)["count"]
              .sum())
    counts["window"] = int(window)
    counts = counts[["app_id", "token_a", "token_b", "window", "count", "period"]]

    # Percent par (app_id, window, period)
    totals = (counts.groupby(["app_id", "window", "period"], as_index=False)["count"]
                    .sum().rename(columns={"count": "total"}))
    pct = counts.merge(totals, on=["app_id", "window", "period"], how="left")
    pct["percent"] = (pct["count"] / pct["total"].replace(0, pd.NA)) * 100.0
    pct["percent"] = pct["percent"].fillna(0.0).round(4)
    pct = pct[["app_id", "token_a", "token_b", "window", "percent", "period"]]

    # Coupe optionnelle
    if topk is not None:
        if len(counts) > topk:
            counts = counts.nlargest(topk, "count")
        if len(pct) > topk:
            pct = pct.nlargest(topk, "percent")

    # Écritures Firestore (remplacement complet des collections)
    replace_collection(
        "cooccurrences_counts",
        co_counts.to_dict("records"),
        id_keys=["app_id", "period", "token_a", "token_b"]
    )
    
    replace_collection(
        "cooccurrences_percent",
        co_pct.to_dict("records"),
        id_keys=["app_id", "period", "token_a", "token_b"]
    )

    print(f"[GOLD] Done. window={window} period={period} "
          f"rows_counts={len(counts)} rows_percent={len(pct)}")
#