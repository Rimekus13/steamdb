# app.py — Clean UI v4 (sidebar with clear sections & dual playtime filter)
import os
import re
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd

# -------------------------------------------------
# Mode test (évite d'exécuter la partie UI/DB sous pytest)
# -------------------------------------------------
IS_TEST = bool(os.getenv("PYTEST_CURRENT_TEST"))

# -------------------------------------------------
# Imports "tolérants" (ne doivent pas casser les tests)
# -------------------------------------------------
try:
    from config import APP_TITLE, LAYOUT, BASE_CSS
except Exception:
    APP_TITLE, LAYOUT, BASE_CSS = "Steam Reviews", "wide", ""

# Ces modules peuvent exister dans ton projet ; ils ne déclenchent pas l'UI en mode test
try:
    from utils import clamp
except Exception:
    # fallback minimal pour les tests au cas où utils manque
    def clamp(x, lo, hi):
        return max(lo, min(hi, x))

try:
    from data_loader import get_db, get_game_name, load_df
except Exception:
    # stubs neutres si importé pendant les tests (mais non utilisés)
    def get_db(): raise RuntimeError("get_db not available in test mode")
    def get_game_name(app_id): return str(app_id)
    def load_df(name, db): return pd.DataFrame()

# Ici, on garde la signature telle que dans ton projet réel (UI),
# mais pendant les tests, on ne passera JAMAIS par ce chemin.
try:
    from analysis import get_vader, compute_sentiment
except Exception:
    def get_vader(): return None
    def compute_sentiment(*args, **kwargs): return 0.0

# -------------------------------------------------
# ⚙️ Fonction utilitaire utilisée par les tests (indépendante de Streamlit)
# -------------------------------------------------
def apply_filters(
    df: pd.DataFrame,
    languages=None,
    date_range=None,          # tuple (start, end) ; accepte str/pd.Timestamp/date
    sentiment_range=None,     # tuple (lo, hi)
    search_terms=None         # list[str]
) -> pd.DataFrame:
    """
    Version minimaliste, indépendante de Streamlit/config, utilisée par les tests.
    - Normalise la date vers une colonne 'timestamp' si besoin (à partir de 'review_date').
    - Applique les filtres de langue, date, sentiment et recherche texte.
    """
    out = df.copy()

    # Normaliser la date -> 'timestamp' si absent
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    elif "review_date" in out.columns:
        out["timestamp"] = pd.to_datetime(out["review_date"], errors="coerce")

    # Langues
    if languages:
        out = out[out["language"].isin(languages)]

    # Dates
    if date_range and "timestamp" in out.columns:
        start, end = date_range
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        out = out[(out["timestamp"] >= start) & (out["timestamp"] <= end)]

    # Sentiment
    if sentiment_range and "sentiment" in out.columns:
        lo, hi = sentiment_range
        out = out[(out["sentiment"] >= float(lo)) & (out["sentiment"] <= float(hi))]

    # Recherche texte (sur review_text si dispo, sinon review)
    if search_terms:
        base_text = "review_text" if "review_text" in out.columns else ("review" if "review" in out.columns else None)
        if base_text:
            mask = pd.Series(False, index=out.index)
            for term in search_terms:
                if isinstance(term, str) and term:
                    mask |= out[base_text].fillna("").str.contains(term, case=False, regex=True)
            out = out[mask]

    return out

# -------------------------------------------------
# ❌ Tout ce qui suit (UI Streamlit) NE DOIT PAS s'exécuter sous pytest
# -------------------------------------------------
if not IS_TEST:
    import streamlit as st
    # Tabs UI (tes modules existants)
    from tabs import (
        synthese, sentiment, themes, langues, playtime,
        longueur, cooccurrences, anomalies, qualite,
        explorateur, updates
    )
    from dotenv import load_dotenv
    load_dotenv()

    # ---------------------------------------------
    # Page config & CSS
    # ---------------------------------------------
    st.set_page_config(page_title=APP_TITLE, layout=LAYOUT)
    if BASE_CSS:
        st.markdown(BASE_CSS, unsafe_allow_html=True)

    # ---------------------------------------------
    # DB & discovery
    # ---------------------------------------------
    db = get_db()
    all_cols = db.list_collection_names()

    GLOBAL_CLEAN_NAMES = {"reviews_clean", "reviews_clean_airflow", "silver_clean", "clean_reviews"}
    per_app_cols = [c for c in all_cols if c.startswith("reviews_") and c not in GLOBAL_CLEAN_NAMES]
    per_app_ids = [c.split("reviews_", 1)[1] for c in per_app_cols]
    per_app_ids_numeric = sorted([x for x in per_app_ids if x.isdigit()])

    clean_like = [c for c in ["reviews_clean", "reviews_clean_airflow", "silver_clean", "clean_reviews"] if c in all_cols]

    if per_app_ids_numeric:
        mode = "per_collection"
        app_ids = per_app_ids_numeric
    elif clean_like:
        mode = "single_clean"
        base_clean = clean_like[0]
        try:
            ids = db[base_clean].distinct("app_id") or db[base_clean].distinct("appid") or db[base_clean].distinct("appId")
            def norm(x):
                if isinstance(x, int): return str(x) if x > 0 else None
                if isinstance(x, str) and x.isdigit(): return x
                return None
            app_ids = sorted(set(filter(None, (norm(x) for x in ids))))[:3000]
        except Exception:
            app_ids = []
    else:
        mode = "none"
        app_ids = []

    with st.expander("🛠️ Debug Mongo", expanded=False):
        st.write({
            "db": getattr(db, "name", "?"),
            "mode": mode,
            "collections": all_cols[:30],
            "per_app_ids_numeric": per_app_ids_numeric[:25],
            "clean_like": clean_like,
            "num_apps": len(app_ids),
        })

    if not app_ids:
        st.error("Aucun jeu détecté.")
        st.stop()

    names = {app_id: get_game_name(app_id) for app_id in app_ids}
    selected_app = st.selectbox("🎮 Jeu", options=app_ids, format_func=lambda a: names.get(a, a))
    st.markdown(f"### {names.get(selected_app, selected_app)}")
    st.image(f"https://cdn.akamai.steamstatic.com/steam/apps/{selected_app}/header.jpg", use_container_width=True)

    # ---------------------------------------------
    # Loading
    # ---------------------------------------------
    def fetch_clean_reviews(_db, app_id_str: str) -> pd.DataFrame:
        targets = [c for c in ["reviews_clean", "reviews_clean_airflow", "silver_clean", "clean_reviews"] if c in _db.list_collection_names()]
        if not targets:
            return pd.DataFrame()
        col = targets[0]
        ors = [{"app_id": app_id_str}]
        if app_id_str.isdigit():
            ors.append({"app_id": int(app_id_str)})
        q = {"$or": ors}
        docs = list(_db[col].find(q, {"_id": 0}))
        return pd.DataFrame(docs)

    def normalize_from_clean(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=[
                "review_date","language","voted_up","cleaned_review","review_text",
                "compound","playtime_hours"
            ])
        df = df.copy()

        # text
        if "cleaned_review" in df.columns:
            df["review_text"] = df["cleaned_review"].fillna("").astype(str)
        else:
            df["review_text"] = df.get("review_text", "").fillna("").astype(str)

        # language
        if "language" not in df.columns:
            df["language"] = "unknown"

        # voted_up
        if "voted_up" not in df.columns:
            df["voted_up"] = pd.NA

        # date
        if "review_date" in df.columns:
            df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
        elif "timestamp_created" in df.columns:
            df["review_date"] = pd.to_datetime(df["timestamp_created"], unit="s", errors="coerce")
        else:
            df["review_date"] = pd.NaT

        # sentiment (ETL)
        if "compound" in df.columns:
            df["compound"] = pd.to_numeric(df["compound"], errors="coerce").fillna(0.0)
        else:
            df["compound"] = 0.0

        # playtime hours
        if "author" in df.columns:
            try:
                pt = df["author"].apply(lambda x: x.get("playtime_forever") if isinstance(x, dict) else None)
                df["playtime_hours"] = (pd.to_numeric(pt, errors="coerce").fillna(0) / 60).round(2)
            except Exception:
                df["playtime_hours"] = pd.NA
        if "playtime_hours" not in df.columns or df["playtime_hours"].isna().all():
            if "playtime_forever" in df.columns:
                df["playtime_hours"] = (pd.to_numeric(df["playtime_forever"], errors="coerce").fillna(0) / 60).round(2)
        if "playtime_hours" not in df.columns or df["playtime_hours"].isna().all():
            if "playtime_at_review" in df.columns:
                df["playtime_hours"] = (pd.to_numeric(df["playtime_at_review"], errors="coerce").fillna(0) / 60).round(2)
        if "playtime_hours" not in df.columns:
            df["playtime_hours"] = pd.NA

        return df

    if mode == "per_collection":
        df = load_df(f"reviews_{selected_app}", db)
    else:
        df = fetch_clean_reviews(db, selected_app)
        df = normalize_from_clean(df)

    if df.empty:
        st.warning("Aucune donnée disponible pour ce jeu."); st.stop()

    # ---------------------------------------------
    # Sentiment numeric + label
    # ---------------------------------------------
    if "compound" in df.columns:
        df["sentiment"] = pd.to_numeric(df["compound"], errors="coerce").fillna(0.0)
    else:
        sia = get_vader()
        # NOTE: ta signature UI semble être compute_sentiment(sia, text)
        df["sentiment"] = df["review_text"].apply(lambda t: compute_sentiment(sia, t))

    def senti_label(x: float) -> str:
        try:
            if x > 0.05: return "positif"
            if x < -0.05: return "négatif"
            return "neutre"
        except Exception:
            return "neutre"
    df["sentiment_label"] = df["sentiment"].apply(senti_label)

    SENTI_COLORS = {"positif":"#22c55e", "neutre":"#9ca3af", "négatif":"#ef4444"}

    # ---------------------------------------------
    # Play profiles
    # ---------------------------------------------
    def play_profile(h):
        try:
            if pd.isna(h): return "Inconnu"
            h = float(h)
            if h <= 1: return "Découverte (≤1h)"
            if h <= 5: return "Casual (1–5h)"
            if h <= 20: return "Régulier (5–20h)"
            if h <= 100: return "Core (20–100h)"
            return "Hardcore (>100h)"
        except Exception:
            return "Inconnu"
    df["play_profile"] = df.get("playtime_hours", pd.Series([np.nan]*len(df))).apply(play_profile)

    # ---------------------------------------------
    # Default dates
    # ---------------------------------------------
    if df["review_date"].notna().any():
        global_min = df["review_date"].min().date()
        global_max = df["review_date"].max().date()
    else:
        global_min = date(2024, 1, 1)
        global_max = datetime.now().date()

    if st.session_state.get("current_app") != selected_app:
        st.session_state.current_app = selected_app
        st.session_state.date_min = global_min
        st.session_state.date_max = global_max

    st.session_state.date_min = clamp(st.session_state.get("date_min", global_min), global_min, global_max)
    st.session_state.date_max = clamp(st.session_state.get("date_max", global_min), global_min, global_max)
    if st.session_state.date_min > st.session_state.date_max:
        st.session_state.date_min, st.session_state.date_max = st.session_state.date_max, st.session_state.date_min

    # ---------------------------------------------
    # Sidebar filters (clearly separated sections)
    # ---------------------------------------------
    def render_filters_sidebar(df, global_min, global_max):
        sb = st.sidebar
        sb.header("🧭 Filtres")

        # ========== 🎯 Essentiels ==========
        sb.subheader("🎯 Essentiels")
        # Dates
        dmin = sb.date_input("📅 Depuis", value=st.session_state.date_min,
                            min_value=global_min, max_value=global_max, key="f_date_min")
        dmax = sb.date_input("📅 Jusqu’à", value=st.session_state.date_max,
                            min_value=global_min, max_value=global_max, key="f_date_max")
        col_q1, col_q2, col_q3 = sb.columns(3)
        if col_q1.button("7 j", key="f_q7"): 
            st.session_state.date_min = max(global_min, global_max - timedelta(days=6)); st.session_state.date_max = global_max; st.rerun()
        if col_q2.button("30 j", key="f_q30"):
            st.session_state.date_min = max(global_min, global_max - timedelta(days=29)); st.session_state.date_max = global_max; st.rerun()
        if col_q3.button("90 j", key="f_q90"):
            st.session_state.date_min = max(global_min, global_max - timedelta(days=89)); st.session_state.date_max = global_max; st.rerun()

        # Langues
        langs = sorted(df["language"].dropna().unique().tolist())
        if not langs: langs = ["unknown"]
        chosen_langs = sb.multiselect("🌐 Langues", options=langs, default=langs, key="f_langs")

        sb.divider()

        # ========== ⏱️ Heures de jeu ==========
        sb.subheader("⏱️ Heures de jeu")
        play_mode = sb.radio("Mode", ["Profils prédéfinis", "Plage d'heures"], key="f_play_mode", horizontal=True)

        ALL_PROFILES = ["Découverte (≤1h)","Casual (1–5h)","Régulier (5–20h)","Core (20–100h)","Hardcore (>100h)","Inconnu"]
        present_profiles = [p for p in ALL_PROFILES if p in df.get("play_profile", pd.Series(dtype=str)).unique().tolist()]
        if not present_profiles: present_profiles = ["Inconnu"]

        chosen_profiles = None
        hours_min = None
        hours_max = None
        include_unknown = False

        if play_mode == "Profils prédéfinis":
            chosen_profiles = sb.multiselect("👤 Profils", options=ALL_PROFILES, default=present_profiles, key="f_profiles")
        else:
            h_series = pd.to_numeric(df.get("playtime_hours", pd.Series(dtype=float)), errors="coerce")
            hmin = int(np.nanmin(h_series)) if h_series.notna().any() else 0
            hmax_raw = int(np.nanmax(h_series)) if h_series.notna().any() else 0
            hmax = max(hmin + 1, int(np.ceil(hmax_raw / 10) * 10) if hmax_raw > 0 else 10)
            default_low = hmin
            default_high = hmax_raw if hmax_raw > hmin else hmax
            hours_min, hours_max = sb.slider("Plage (heures)", min_value=hmin, max_value=hmax,
                                            value=(int(default_low), int(default_high)), step=1, key="f_play_range")
            include_unknown = sb.checkbox("Inclure 'Inconnu'", value=False, key="f_play_inc_unknown")

        sb.divider()

        # ========== 🙂 Sentiment ==========
        sb.subheader("🙂 Sentiment")
        senti_choice = sb.radio("", options=["Tous", "Positifs", "Neutres", "Négatifs"],
                                index=0, key="f_senti", horizontal=True)

        sb.divider()

        # ========== 🔎 Mots-clés ==========
        sb.subheader("🔎 Mots-clés")
        keywords_raw = sb.text_input("Ex: bug, crash", key="f_keywords")
        match_all = sb.checkbox("ET logique (tous les mots)", value=False, key="f_keywords_all")

        sb.divider()

        # ========== ⚙️ Actions ==========
        sb.subheader("⚙️ Actions")
        hard_refresh = sb.checkbox("🧹 Purger le cache avant calcul", value=False, key="f_hard_refresh")
        col_a, col_b = sb.columns(2)
        apply_clicked = col_a.button("🔄 Appliquer", use_container_width=True)
        reset_clicked = col_b.button("♻️ Réinitialiser", use_container_width=True)

        if reset_clicked:
            st.session_state.date_min = global_min
            st.session_state.date_max = global_max
            st.session_state.f_senti = "Tous"
            st.session_state.f_play_mode = "Profils prédéfinis"
            st.session_state.f_profiles = present_profiles
            st.session_state.f_langs = langs
            st.session_state.f_keywords = ""
            st.session_state.f_keywords_all = False
            st.session_state.f_hard_refresh = False
            st.rerun()

        if apply_clicked:
            st.session_state.date_min = clamp(dmin, global_min, global_max)
            st.session_state.date_max = clamp(dmax, global_min, global_max)
            if st.session_state.date_min > st.session_state.date_max:
                st.session_state.date_min, st.session_state.date_max = st.session_state.date_max, st.session_state.date_min
            if hard_refresh:
                st.cache_data.clear()
            st.rerun()

        return {
            "chosen_langs": chosen_langs,
            "senti_choice": senti_choice,
            "play_mode": play_mode,
            "chosen_profiles": chosen_profiles,
            "hours_min": hours_min,
            "hours_max": hours_max,
            "include_unknown": include_unknown,
            "keywords_raw": keywords_raw,
            "match_all": match_all,
        }

    flt = render_filters_sidebar(df, global_min, global_max)

    # Apply filters (UI)
    mask = pd.Series(True, index=df.index)

    if flt["chosen_langs"]:
        mask &= df["language"].isin(flt["chosen_langs"])

    if df["review_date"].notna().any():
        mask &= (df["review_date"].dt.date >= st.session_state.date_min) & (df["review_date"].dt.date <= st.session_state.date_max)

    if flt["senti_choice"] == "Positifs":
        mask &= df["sentiment"] > 0.05
    elif flt["senti_choice"] == "Neutres":
        mask &= (df["sentiment"] >= -0.05) & (df["sentiment"] <= 0.05)
    elif flt["senti_choice"] == "Négatifs":
        mask &= df["sentiment"] < -0.05

    # Playtime filter
    if flt["play_mode"] == "Profils prédéfinis":
        if flt["chosen_profiles"]:
            mask &= df["play_profile"].isin(flt["chosen_profiles"])
    else:
        ph = pd.to_numeric(df["playtime_hours"], errors="coerce") if "playtime_hours" in df.columns else pd.Series(index=df.index, dtype=float)
        in_range = ph.between(flt["hours_min"], flt["hours_max"], inclusive="both")
        if flt["include_unknown"]:
            in_range = in_range | ph.isna()
        mask &= in_range

    df_f = df[mask].copy()

    kw = flt["keywords_raw"]
    if kw and kw.strip() and not df_f.empty:
        kws = [k.strip().lower() for k in kw.split(",") if k.strip()]
        if kws:
            if flt["match_all"]:
                lookaheads = "".join([rf"(?=.*\b{re.escape(k)}\b)" for k in kws])
                pattern = lookaheads + r".*"
            else:
                pattern = r"\b(" + "|".join([re.escape(k) for k in kws]) + r")\b"
            df_f = df_f[df_f["review_text"].astype(str).str.contains(pattern, regex=True, na=False)]

    # Caption
    if flt["play_mode"] == "Profils prédéfinis":
        play_txt = f"Profils: {', '.join(flt['chosen_profiles']) if flt['chosen_profiles'] else '—'}"
    else:
        rng = f"{flt['hours_min']}–{flt['hours_max']}h"
        if flt["include_unknown"]:
            rng += " (+ inconnus)"
        play_txt = f"Heures: {rng}"

    st.caption(
        f"🗓️ **{st.session_state.date_min} → {st.session_state.date_max}** • "
        f"{play_txt} • "
        f"Avis filtrés : **{len(df_f):,}**"
    )

    # ---------------------------------------------
    # KPIs + context
    # ---------------------------------------------
    pos = (df_f["sentiment"] > 0.05).mean() * 100 if len(df_f) else 0.0
    neu = ((df_f["sentiment"] >= -0.05) & (df_f["sentiment"] <= 0.05)).mean() * 100 if len(df_f) else 0.0
    neg = (df_f["sentiment"] < -0.05).mean() * 100 if len(df_f) else 0.0
    avg_len = df_f["review_text"].astype(str).str.split().apply(len).replace(0, np.nan).mean() if len(df_f) else 0.0

    theme_dict = {
        "performances":["lag","fps","performance","stuttering","freeze","latence"],
        "gameplay":["gameplay","controls","mécaniques","mechanics","control"],
        "graphismes":["graphics","graphismes","art","textures"],
        "multijoueur":["multiplayer","coop","serveur","server"],
        "bugs":["bug","crash","error","issue","glitch"],
        "contenu":["content","dlc","missions","maps","map"]
    }
    rows = []
    for th, kws in theme_dict.items():
        freq = df_f["review_text"].apply(lambda x: any(k.lower() in str(x).lower() for k in kws)).mean()*100 if len(df_f) else 0.0
        rows.append((th, freq))
    freq_df = pd.DataFrame(rows, columns=["Thème","Fréquence (%)"]).sort_values("Fréquence (%)", ascending=False)

    # Tabs
    tabs = st.tabs([
        "📌 Synthèse","🙂 Sentiment","🧩 Thèmes","🌍 Langues","⏱️ Heures de jeu",
        "✍️ Longueur & lisibilité","🔗 Cooccurrences","⚠️ Anomalies","✅ Qualité données","🔎 Explorateur","🛠️ Mises à jour"
    ])

    ctx = {
        "df": df, "df_f": df_f,
        "pos": pos, "neu": neu, "neg": neg, "avg_len": avg_len,
        "theme_dict": theme_dict, "freq_df": freq_df,
        "sentiment_colors": {"positif":"#22c55e","neutre":"#9ca3af","négatif":"#ef4444"}
    }

    with tabs[0]: synthese.render(st, ctx)
    with tabs[1]: sentiment.render(st, ctx)
    with tabs[2]: themes.render(st, ctx)
    with tabs[3]: langues.render(st, ctx)
    with tabs[4]: playtime.render(st, ctx)
    with tabs[5]: longueur.render(st, ctx)
    with tabs[6]: cooccurrences.render(st, ctx)
    with tabs[7]: anomalies.render(st, ctx)
    with tabs[8]: qualite.render(st, ctx)
    with tabs[9]: explorateur.render(st, ctx)
    with tabs[10]: updates.render(st, ctx)
