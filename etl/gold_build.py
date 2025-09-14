# etl/gold_build.py
import pandas as pd
from datetime import datetime
from typing import List, Iterable, Tuple, Optional

from .text_utils import tokenize_no_stop
from .firestore_utils import col_clean_query, replace_collection, log_fs_state

WINDOW = 5                    # taille de fenêtre de cooccurrence
TOP_K: Optional[int] = None   # None = pas de coupe ; sinon limite le nombre de lignes publiées

def _periode_depuis_ligne(row) -> Optional[str]:
    """
    Période mensuelle (YYYY-MM) :
      - priorité à review_date (YYYY-MM-DD) → YYYY-MM
      - sinon fallback sur timestamp_created (epoch sec) → YYYY-MM
    """
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

def _paires_fenetre(tokens: List[str], window: int = WINDOW):
    """
    Fenêtre glissante : pour chaque i, prend les tokens j dans (i, i+window).
    On normalise l’ordre (a,b) trié pour éviter les doublons (a,b)/(b,a).
    """
    n = len(tokens)
    for i in range(n):
        a = tokens[i]
        jmax = min(n, i + window)
        for j in range(i + 1, jmax):
            b = tokens[j]
            if a != b:
                yield tuple(sorted((a, b)))

def build_gold(app_ids: Optional[List[str]] = None, for_airflow: bool = False) -> None:
    """
    Construit et publie les tables Gold dans Firestore :
      - cooccurrences_counts  {app_id, token_a, token_b, window, count,  period}
      - cooccurrences_percent {app_id, token_a, token_b, window, percent, period}
    """
    print("[GOLD] Inspection de l’état Firestore…")
    log_fs_state(sample=3)

    print("[GOLD] Lecture de reviews_clean…")
    df = pd.DataFrame(col_clean_query())
    print(f"[GOLD] Lignes chargées depuis CLEAN: {len(df)}")
    if df.empty or "cleaned_review" not in df.columns:
        print("[GOLD] CLEAN vide ou colonne `cleaned_review` absente → purge des cooccurrences et sortie.")
        replace_collection("cooccurrences_counts", [])
        replace_collection("cooccurrences_percent", [])
        return

    # Normalisations
    if "app_id" not in df.columns:
        df["app_id"] = ""
    df["app_id"] = df["app_id"].astype(str)

    # Option : filtrer sur certaines apps
    if app_ids:
        allow = {str(a) for a in app_ids}
        df = df[df["app_id"].isin(allow)]
        print(f"[GOLD] Filtre app_ids → {len(df)} lignes")

    # Tokenisation + période mensuelle
    df["tokens"] = df["cleaned_review"].fillna("").map(tokenize_no_stop)
    df["period"] = df.apply(_periode_depuis_ligne, axis=1)

    # Filtre qualité minimum
    df = df[(df["tokens"].map(len) >= 2) & df["period"].notna() & df["app_id"].ne("")]
    print(f"[GOLD] Après filtrage (tokens>=2 & période & app_id) → {len(df)} lignes")

    if df.empty:
        replace_collection("cooccurrences_counts", [])
        replace_collection("cooccurrences_percent", [])
        return

    # Génération des paires dans la fenêtre
    recs = []
    for _, row in df.iterrows():
        app_id = row["app_id"]
        period = row["period"]
        for a, b in _paires_fenetre(row["tokens"], window=WINDOW):
            recs.append((app_id, period, a, b, 1))

    if not recs:
        print("[GOLD] Aucune paire générée → purge & sortie.")
        replace_collection("cooccurrences_counts", [])
        replace_collection("cooccurrences_percent", [])
        return

    rec_df = pd.DataFrame(recs, columns=["app_id", "period", "token_a", "token_b", "count"])
    co_counts = (rec_df
                 .groupby(["app_id", "period", "token_a", "token_b"], as_index=False)["count"]
                 .sum())
    print(f"[GOLD] cooccurrences_counts brut: {len(co_counts)} lignes")

    # Totaux par (app_id, period) pour convertir en pourcentage
    totals = (co_counts
              .groupby(["app_id", "period"], as_index=False)["count"]
              .sum()
              .rename(columns={"count": "total_pairs"}))

    co_pct = co_counts.merge(totals, on=["app_id", "period"], how="left")
    co_pct["percent"] = (co_pct["count"] / co_pct["total_pairs"]).fillna(0.0)
    co_pct = co_pct.drop(columns=["total_pairs"])
    print(f"[GOLD] cooccurrences_percent brut: {len(co_pct)} lignes")

    # Coupe optionnelle
    if TOP_K is not None and len(co_counts) > TOP_K:
        co_counts = co_counts.nlargest(TOP_K, "count")
    if TOP_K is not None and len(co_pct) > TOP_K:
        co_pct = co_pct.nlargest(TOP_K, "percent")

    # Ajout de la fenêtre + ordre des colonnes (schéma demandé)
    co_counts["window"] = WINDOW
    co_pct["window"] = WINDOW

    co_counts = co_counts[["app_id", "token_a", "token_b", "window", "count", "period"]]
    co_pct    = co_pct   [["app_id", "token_a", "token_b", "window", "percent", "period"]]

    # Publication Firestore (IDs déterministes)
    print("[GOLD] Publication Firestore…")
    replace_collection(
        "cooccurrences_counts",
        co_counts.to_dict("records"),
        id_keys=["app_id", "period", "token_a", "token_b", "window"],
    )
    replace_collection(
        "cooccurrences_percent",
        co_pct.to_dict("records"),
        id_keys=["app_id", "period", "token_a", "token_b", "window"],
    )
    print("[GOLD] ✅ Terminé.")
