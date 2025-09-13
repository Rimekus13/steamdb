# app.py — Firestore + Auth robuste (streamlit-authenticator) + UI Steam Reviews
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

try:
    from utils import clamp
except Exception:
    def clamp(x, lo, hi):
        return max(lo, min(hi, x))

try:
    from analysis import get_vader, compute_sentiment
except Exception:
    def get_vader(): return None
    def compute_sentiment(*args, **kwargs): return 0.0

# -------------------------------------------------
# 🔗 Firestore client + helpers
# -------------------------------------------------
PROJECT = os.getenv("FIRESTORE_PROJECT") or os.getenv("GCP_PROJECT")

def fs_get_db():
    """Client Firestore via ADC (VM GCE / Workload Identity)."""
    if not PROJECT:
        raise RuntimeError(
            "Projet Firestore inconnu. Définis FIRESTORE_PROJECT ou GCP_PROJECT "
            "dans l'environnement du conteneur."
        )
    from google.cloud import firestore
    return firestore.Client(project=PROJECT)

def fs_list_app_ids(db, limit_scan: int = 5000):
    """
    Détecte les app_ids via les documents top-level de la collection 'reviews_clean'.
    Schéma Silver: reviews_clean/{app_id}/items/{doc}
    """
    cols = db.collection("reviews_clean").list_documents(page_size=limit_scan)
    app_ids = sorted([doc_ref.id for doc_ref in cols if doc_ref.id and doc_ref.id.isdigit()])
    return app_ids

def fs_fetch_clean_df(db, app_id: str, limit: int = 50000) -> pd.DataFrame:
    """
    Lit la sous-collection 'items' sous reviews_clean/{app_id}.
    Normalise les colonnes vers les attentes de l’UI existante.
    """
    items_col = db.collection("reviews_clean").document(str(app_id)).collection("items")
    docs = items_col.limit(limit).stream()
    rows = [d.to_dict() for d in docs]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()

    if df.empty:
        return pd.DataFrame(columns=[
            "review_date","language","voted_up","cleaned_review","review_text",
            "compound","playtime_hours","timestamp_created","timestamp_updated","app_id"
        ])

    df = df.copy()

    # app_id
    if "app_id" not in df.columns:
        df["app_id"] = str(app_id)
    else:
        df["app_id"] = df["app_id"].astype(str)

    # texte nettoyé -> review_text
    if "cleaned_review" in df.columns:
        df["review_text"] = df["cleaned_review"].fillna("").astype(str)
    else:
        base_txt = "review_text" if "review_text" in df.columns else ("text" if "text" in df.columns else "review")
        df["review_text"] = df.get(base_txt, "").fillna("").astype(str)

    # langue
    if "language" not in df.columns:
        df["language"] = "unknown"
    else:
        df["language"] = df["language"].fillna("unknown").astype(str)

    # voted_up
    if "voted_up" not in df.columns:
        df["voted_up"] = pd.NA

    # timestamps -> review_date
    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    elif "timestamp_created" in df.columns:
        df["review_date"] = pd.to_datetime(df["timestamp_created"], unit="s", errors="coerce")
    else:
        df["review_date"] = pd.NaT

    # sentiment
    if "compound" in df.columns:
        df["compound"] = pd.to_numeric(df["compound"], errors="coerce").fillna(0.0)
    else:
        df["compound"] = 0.0

    # playtime_hours
    if "playtime_hours" in df.columns:
        df["playtime_hours"] = pd.to_numeric(df["playtime_hours"], errors="coerce")
    else:
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

def fs_get_game_name(app_id: str) -> str:
    return str(app_id)

# -------------------------------------------------
# ⚙️ Fonction utilitaire utilisée par les tests (indépendante de Streamlit)
# -------------------------------------------------
def apply_filters(
    df: pd.DataFrame,
    languages=None,
    date_range=None,
    sentiment_range=None,
    search_terms=None
) -> pd.DataFrame:
    """
    Version minimaliste, indépendante de Streamlit/config, utilisée par les tests.
    """
    out = df.copy()

    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    elif "review_date" in out.columns:
        out["timestamp"] = pd.to_datetime(out["review_date"], errors="coerce")

    if languages:
        out = out[out["language"].isin(languages)]

    if date_range and "timestamp" in out.columns:
        start, end = date_range
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        out = out[(out["timestamp"] >= start) & (out["timestamp"] <= end)]

    if sentiment_range and "sentiment" in out.columns:
        lo, hi = sentiment_range
        out = out[(out["sentiment"] >= float(lo)) & (out["sentiment"] <= float(hi))]

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
    import json
    import streamlit as st
    import streamlit_authenticator as stauth
    from dotenv import load_dotenv
    load_dotenv()

    # ---------------------------------------------
    # AUTH robuste multi-versions (sans Hasher)
    # ---------------------------------------------
    AUTH_ON = (os.getenv("STREAMLIT_AUTH", "ON").upper() in ("1", "TRUE", "ON", "YES"))

    if AUTH_ON:
        users_env = os.getenv("AUTH_USERS_JSON", "").strip()
        credentials = None
        if users_env:
            try:
                users = json.loads(users_env)
                credentials = {
                    "usernames": {
                        u["username"]: {"name": u["name"], "password": u["password"]}
                        for u in users
                        if all(k in u for k in ("username", "name", "password"))
                    }
                }
            except Exception:
                credentials = None

        # Fallback: admin/admin (bcrypt déjà hashé)
        if not credentials:
            ADMIN_BCRYPT = "$2b$12$KIXQ4Q.ZX7o9qpiapqYPOuNsq4CPGK/c/pXHiY/VKwxLBzME2Y8a2"
            credentials = {"usernames": {"admin": {"name": "Admin", "password": ADMIN_BCRYPT}}}

        cookie_name = os.getenv("AUTH_COOKIE_NAME", "steamdb_auth")
        cookie_key  = os.getenv("AUTH_COOKIE_KEY",  "change-me")
        cookie_days = float(os.getenv("AUTH_COOKIE_EXPIRY_DAYS", "7"))

        authenticator = stauth.Authenticate(
            credentials,
            cookie_name,
            cookie_key,
            cookie_expiry_days=cookie_days,
        )

        # Wrapper de login multi-signatures (selon la version installée)
        def do_login():
            attempts = [
                lambda: authenticator.login("main", "Connexion"),           # (location, form_name)  ← versions récentes
                lambda: authenticator.login("Connexion", "main"),           # (form_name, location)
                lambda: authenticator.login(location="main"),               # (kw-only)
                lambda: authenticator.login("Connexion"),                   # (form_name)
                lambda: authenticator.login("main"),                        # (location)
                lambda: authenticator.login("Connexion", location="main"),  # mix
            ]
            for call in attempts:
                try:
                    return call()
                except Exception:
                    continue
            # Dernier essai (la plus probable) pour exposer l'erreur si tout échoue
            return authenticator.login("main", "Connexion")

        login_out = do_login()
        if isinstance(login_out, tuple) and len(login_out) == 3:
            name, auth_status, username = login_out
        else:
            name = getattr(login_out, "name", None) if login_out else None
            auth_status = getattr(login_out, "authentication_status", None) if login_out else None
            username = getattr(login_out, "username", None) if login_out else None

        if auth_status is False:
            st.error("Identifiants invalides.")
            st.stop()
        elif auth_status is None:
            st.info("Veuillez vous authentifier.")
            st.stop()

        st.sidebar.write(f"🔒 Connecté en tant que **{name or 'utilisateur'}**")
        # Déconnexion (compat multi-versions)
        try:
            authenticator.logout("Déconnexion", "sidebar")
        except Exception:
            try:
                authenticator.logout(location="sidebar")
            except Exception:
                pass

    # ---------------------------------------------
    # Page config & CSS
    # ---------------------------------------------
    st.set_page_config(page_title=APP_TITLE, layout=LAYOUT)
    if BASE_CSS:
        st.markdown(BASE_CSS, unsafe_allow_html=True)

    # ---------------------------------------------
    # DB & discovery (Firestore)
    # ---------------------------------------------
    try:
        db = fs_get_db()
    except Exception as e:
        st.error(f"Impossible d'initialiser Firestore: {e}")
        st.stop()

    try:
        app_ids = fs_list_app_ids(db)
    except Exception as e:
        st.error(f"Erreur lors du scan des jeux (reviews_clean): {e}")
        st.stop()

    if not app_ids:
        st.error("Aucun jeu détecté dans Firestore (collection 'reviews_clean').")
        st.stop()

    names = {app_id: fs_get_game_name(app_id) for app_id in app_ids}
    selected_app = st.selectbox("🎮 Jeu", options=app_ids, index=0, format_func=lambda a: names.get(a, a))
    st.markdown(f"### {names.get(selected_app, selected_app)}")
    st.image(f"https://cdn.akamai.steamstatic.com/steam/apps/{selected_app}/header.jpg", use_container_width=True)

    # ---------------------------------------------
    # Loading (Silver depuis Firestore)
    # ---------------------------------------------
    try:
        df = fs_fetch_clean_df(db, selected_app)
    except Exception as e:
        st.error(f"Erreur de lecture des avis Silver pour {selected_app}: {e}")
        st.stop()

    if df.empty:
        st.warning("Aucune donnée disponible pour ce jeu.")
        st.stop()

    # ---------------------------------------------
    # Sentiment numeric + label
    # ---------------------------------------------
    if "compound" in df.columns:
        df["sentiment"] = pd.to_numeric(df["compound"], errors="coerce").fillna(0.0)
    else:
        sia = get_vader()
        df["sentiment"] = df["review_text"].apply(lambda t: compute_sentiment(sia, t))

    def senti_label(x: float) -> str:
        try:
            if x > 0.05: return "positif"
            if x < -0.05: return "négatif"
            return "neutre"
        except Exception:
            return "neutre"
    df["sentiment_label"] = df["sentiment"].apply(senti_label)

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
    df["playtime_hours"] = pd.to_numeric(df.get("playtime_hours", np.nan), errors="coerce")
    df["play_profile"] = df["playtime_hours"].apply(play_profile)

    # ---------------------------------------------
    # Default dates
    # ---------------------------------------------
    if df["review_date"].notna().any():
        global_min = pd.to_datetime(df["review_date"], errors="coerce").dropna().min().date()
        global_max = pd.to_datetime(df["review_date"], errors="coerce").dropna().max().date()
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
    # Sidebar filters
    # ---------------------------------------------
    def render_filters_sidebar(df, global_min, global_max):
        sb = st.sidebar
        sb.header("🧭 Filtres")

        # 🎯 Essentiels
        sb.subheader("🎯 Essentiels")
        dmin = sb.date_input(
            "📅 Depuis",
            value=st.session_state.date_min,
            min_value=global_min,
            max_value=global_max,
            key="f_date_min"
        )
        dmax = sb.date_input(
            "📅 Jusqu’à",
            value=st.session_state.date_max,
            min_value=global_min,
            max_value=global_max,
            key="f_date_max"
        )
        col_q1, col_q2, col_q3 = sb.columns(3)
        if col_q1.button("7 j", key="f_q7"):
            st.session_state.date_min = max(global_min, global_max - timedelta(days=6)); st.session_state.date_max = global_max; st.rerun()
        if col_q2.button("30 j", key="f_q30"):
            st.session_state.date_min = max(global_min, global_max - timedelta(days=29)); st.session_state.date_max = global_max; st.rerun()
        if col_q3.button("90 j", key="f_q90"):
            st.session_state.date_min = max(global_min, global_max - timedelta(days=89)); st.session_state.date_max = global_max; st.rerun()

        # 🌐 Langues (simple — tu pourras brancher un mapping complet plus tard)
        langs = sorted(df["language"].dropna().unique().tolist()) or ["unknown"]
        # "Toutes les langues" par défaut : on coche tout
        default_langs = langs
        chosen_langs = sb.multiselect("🌐 Langues", options=langs, default=default_langs, key="f_langs")
        # Bouton pratique pour tout sélectionner
        if sb.button("Tout sélectionner", key="f_langs_all"):
            st.session_state.f_langs = langs
            st.rerun()

        sb.divider()

        # ⏱️ Heures de jeu
        sb.subheader("⏱️ Heures de jeu")
        play_mode = sb.radio("Mode", ["Profils prédéfinis", "Plage d'heures"], key="f_play_mode", horizontal=True)

        ALL_PROFILES = ["Découverte (≤1h)","Casual (1–5h)","Régulier (5–20h)","Core (20–100h)","Hardcore (>100h)","Inconnu"]
        present_profiles = [p for p in ALL_PROFILES if p in df.get("play_profile", pd.Series(dtype=str)).unique().tolist()] or ["Inconnu"]

        chosen_profiles = None
        hours_min = None
        hours_max = None
        include_unknown = False

        if play_mode == "Profils prédéfinis":
            chosen_profiles = sb.multiselect("👤 Profils", options=ALL_PROFILES, default=present_profiles, key="f_profiles")
        else:
            h_series = pd.to_numeric(df.get("playtime_hours", pd.Series(dtype=float)), errors="coerce")
            if h_series.notna().any():
                hmin = int(np.nanmin(h_series))
                hmax_raw = int(np.nanmax(h_series))
                if hmax_raw <= hmin:
                    hmax_raw = hmin + 1
            else:
                hmin, hmax_raw = 0, 1  # évite slider min==max
            hmax = max(hmin + 1, int(np.ceil(hmax_raw / 10) * 10) if hmax_raw > 0 else 10)
            default_low = hmin
            default_high = hmax_raw if hmax_raw > hmin else hmax
            hours_min, hours_max = sb.slider(
                "Plage (heures)",
                min_value=int(hmin), max_value=int(hmax),
                value=(int(default_low), int(default_high)),
                step=1, key="f_play_range"
            )
            include_unknown = sb.checkbox("Inclure 'Inconnu'", value=False, key="f_play_inc_unknown")

        sb.divider()

        # 🙂 Sentiment
        sb.subheader("🙂 Sentiment")
        senti_choice = sb.radio(
            "", options=["Tous", "Positifs", "Neutres", "Négatifs"],
            index=0, key="f_senti", horizontal=True
        )

        sb.divider()

        # 🔎 Mots-clés
        sb.subheader("🔎 Mots-clés")
        keywords_raw = sb.text_input("Ex: bug, crash", key="f_keywords")
        match_all = sb.checkbox("ET logique (tous les mots)", value=False, key="f_keywords_all")

        sb.divider()

        # ⚙️ Actions
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
        ph = pd.to_numeric(df["playtime_hours"], errors="coerce")
        in_range = ph.between(flt["hours_min"], flt["hours_max"], inclusive="both")
        if flt["include_unknown"]:
            in_range = in_range | ph.isna()
        mask &= in_range

    df_f = df[mask].copy()

    # Filtre mots-clés
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
        f"Avis filtrés : **{len(df_f):,}** • Projet Firestore: **{PROJECT}**"
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
    from tabs import (
        synthese, sentiment as t_sentiment, themes, langues, playtime,
        longueur, cooccurrences, anomalies, qualite,
        explorateur, updates
    )

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
    with tabs[1]: t_sentiment.render(st, ctx)
    with tabs[2]: themes.render(st, ctx)
    with tabs[3]: langues.render(st, ctx)
    with tabs[4]: playtime.render(st, ctx)
    with tabs[5]: longueur.render(st, ctx)
    with tabs[6]: cooccurrences.render(st, ctx)
    with tabs[7]: anomalies.render(st, ctx)
    with tabs[8]: qualite.render(st, ctx)
    with tabs[9]: explorateur.render(st, ctx)
    with tabs[10]: updates.render(st, ctx)
#