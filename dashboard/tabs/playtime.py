# tabs/playtime.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import title

def segment_playtime(h):
    try:
        h = float(h)
    except Exception:
        return "Inconnu"
    if h < 10:
        return "Occasionnel (<10h)"
    if h < 100:
        return "Régulier (10–100h)"
    return "Intensif (≥100h)"

def render(st, ctx):
    df_f = ctx["df_f"]
    st.markdown("<div class='card'><h3>Profils par heures de jeu</h3>", unsafe_allow_html=True)

    # Série heures → numérique
    h = pd.to_numeric(df_f.get("playtime_hours"), errors="coerce") if len(df_f) else pd.Series(dtype=float)
    valid = h.dropna()

    if valid.empty:
        st.info("Heures de jeu non disponibles.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Bornes brutes
    min_h = float(np.nanmin(valid))
    max_h = float(np.nanmax(valid))
    if not np.isfinite(min_h):
        min_h = 0.0
    if not np.isfinite(max_h):
        max_h = min_h

    # Arrondis
    min_h = max(0.0, float(np.floor(min_h)))
    max_h = float(np.ceil(max_h))

    # Cas dégénéré: une seule valeur (ex: 0h partout) → slider désactivé
    if max_h <= min_h:
        st.warning("Toutes les heures de jeu valent la même valeur (ex: 0 h). Plage verrouillée.")
        # on affiche un slider désactivé juste pour l’UI
        st.slider(
            "Plage d'heures de jeu",
            min_value=float(min_h),
            max_value=float(min_h + 1.0),
            value=(float(min_h), float(min_h + 1.0)),
            step=1.0,
            key="playtime_range_slider_disabled",
            disabled=True,
        )
        pmin = pmax = float(min_h)
        in_range = valid.eq(pmin).reindex(h.index, fill_value=False)
    else:
        # Slider normal
        default_low = float(min_h)
        default_high = float(max_h)
        pmin, pmax = st.slider(
            "Plage d'heures de jeu",
            min_value=float(min_h),
            max_value=float(max_h),
            value=(default_low, default_high),
            step=1.0,
            key="playtime_range_slider",
        )
        in_range = h.between(pmin, pmax, inclusive="both")

    df_p = df_f[in_range].copy()
    if df_p.empty:
        st.warning("Aucun avis dans la plage d'heures sélectionnée.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Segmentation + garde-fous
    df_p["playtime_hours"] = pd.to_numeric(df_p.get("playtime_hours"), errors="coerce")
    df_p["player_segment"] = df_p["playtime_hours"].apply(segment_playtime)
    if "review_text" not in df_p.columns:
        df_p["review_text"] = ""
    if "sentiment" not in df_p.columns:
        df_p["sentiment"] = 0.0

    seg = (
        df_p.groupby("player_segment")
            .agg(avis=("review_text", "count"), sentiment_moy=("sentiment", "mean"))
            .reset_index()
            .sort_values("avis", ascending=False)
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(seg, use_container_width=True, height=220)
    with c2:
        if not seg.empty:
            fig, ax = plt.subplots(figsize=(5, 2.2))
            ax.bar(seg["player_segment"], seg["sentiment_moy"])
            ax.tick_params(axis="x", rotation=15)
            title(ax, "Sentiment par segment")
            st.pyplot(fig, use_container_width=True)
            st.markdown(
                "<div class='small'>Cibler des actions par profils (onboarding vs end-game).</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Aucun segment à afficher.")

    # Nuage de points (sentiment vs heures)
    x = pd.to_numeric(df_p.get("playtime_hours"), errors="coerce")
    y = pd.to_numeric(df_p.get("sentiment"), errors="coerce")
    mask = x.notna() & y.notna()
    if mask.any():
        fig, ax = plt.subplots(figsize=(10, 2.2))
        ax.scatter(x[mask], y[mask], s=8, alpha=0.35)
        title(ax, "Sentiment vs Heures")
        ax.set_xlabel("Heures")
        ax.set_ylabel("Score")
        st.pyplot(fig, use_container_width=True)
        st.markdown("<div class='small'>Relation expérience ↔ ressenti.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
