# tabs/playtime.py
import numpy as np
import matplotlib.pyplot as plt
from utils import title

def segment_playtime(h):
    try: h=float(h)
    except: return "Inconnu"
    if h<10: return "Occasionnel (<10h)"
    if h<100: return "Régulier (10–100h)"
    return "Intensif (≥100h)"

def render(st, ctx):
    df_f = ctx["df_f"]
    st.markdown("<div class='card'><h3>Profils par heures de jeu</h3>", unsafe_allow_html=True)
    if "playtime_hours" in df_f.columns and len(df_f):
        min_h=float(np.nanmin(df_f["playtime_hours"])) if len(df_f) else 0.0
        max_h=float(np.nanmax(df_f["playtime_hours"])) if len(df_f) else 0.0
        pmin,pmax=st.slider("Plage d'heures de jeu", float(min_h), float(max_h), (float(min_h), float(max_h)), 1.0, key="playtime_range_slider")
        df_p=df_f[(df_f["playtime_hours"]>=pmin)&(df_f["playtime_hours"]<=pmax)].copy()
        df_p["player_segment"]=df_p["playtime_hours"].apply(segment_playtime)

        seg=df_p.groupby("player_segment").agg(avis=("review_text","count"), sentiment_moy=("sentiment","mean")).reset_index().sort_values("avis", ascending=False)
        c1,c2=st.columns([1,1])
        with c1: st.dataframe(seg, use_container_width=True, height=220)
        with c2:
            fig, ax = plt.subplots(figsize=(5,2.2))
            ax.bar(seg["player_segment"], seg["sentiment_moy"]); ax.tick_params(axis='x', rotation=15)
            title(ax,"Sentiment par segment"); st.pyplot(fig, use_container_width=True)
            st.markdown("<div class='small'>Cibler des actions par profils (onboarding vs end‑game).</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10,2.2))
        ax.scatter(df_p["playtime_hours"], df_p["sentiment"], s=8, alpha=0.35)
        title(ax,"Sentiment vs Heures"); ax.set_xlabel("Heures"); ax.set_ylabel("Score")
        st.pyplot(fig, use_container_width=True)
        st.markdown("<div class='small'>Relation expérience ↔ ressenti.</div>", unsafe_allow_html=True)
    else: st.info("Heures de jeu non disponibles.")
    st.markdown("</div>", unsafe_allow_html=True)
