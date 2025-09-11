# tabs/synthese.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import DEFAULT_FIGSIZE_WIDE, title, compact_time_axis
from config import PRIMARY, GOOD, BAD, GREY

def render(st, ctx):
    df_f = ctx["df_f"]
    freq_df = ctx["freq_df"]
    avg_len = ctx["avg_len"]
    pos, neu, neg = ctx["pos"], ctx["neu"], ctx["neg"]

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-xl kpi-pos">
          <div class="hl">💬 Satisfaction moyenne</div>
          <div class="val">{(df_f['sentiment'].mean() if len(df_f) else 0):.2f}</div>
          <div class="tag">Score de -1 (négatif) à +1 (positif)</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-xl kpi-neu">
          <div class="hl">📈 % Pos / Neu / Neg</div>
          <div class="val">{pos:.0f}% / {neu:.0f}% / {neg:.0f}%</div>
          <div class="tag">Répartition des avis</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-xl kpi-len">
          <div class="hl">📝 Mots / avis (moy.)</div>
          <div class="val">{(avg_len or 0):.1f}</div>
          <div class="tag">Indicateur d'engagement</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-xl kpi-lang">
          <div class="hl">🌍 Langues distinctes</div>
          <div class="val">{df_f['language'].nunique() if len(df_f) else 0}</div>
          <div class="tag">Diversité des retours</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='card'><h3>Vue d’ensemble</h3>", unsafe_allow_html=True)

    l1,l2 = st.columns([1,1])
    with l1: syn_min_sent = st.slider("Seuil min. sentiment", -1.0, 1.0, -1.0, 0.05, key="syn_min_sent_slider")
    with l2: syn_max_sent = st.slider("Seuil max. sentiment", -1.0, 1.0, 1.0, 0.05, key="syn_max_sent_slider")
    df_syn = df_f[(df_f["sentiment"]>=syn_min_sent) & (df_f["sentiment"]<=syn_max_sent)]

    A,B,C = st.columns([1.5,1,1])
    with A:
        if df_syn["review_date"].notna().any() and len(df_syn):
            ts = df_syn.dropna(subset=["review_date"]).copy()
            ts["date"] = ts["review_date"].dt.to_period("W").dt.start_time
            series = ts.groupby("date")["sentiment"].mean()
            fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_WIDE)
            ax.plot(series.index, series.values)
            compact_time_axis(ax,3,5); title(ax,"Sentiment hebdo")
            ax.set_xlabel(""); ax.set_ylabel("Score")
            st.pyplot(fig, use_container_width=True)
            st.markdown("<div class='small'>Évolution lissée par semaine sur la période filtrée.</div>", unsafe_allow_html=True)
        else: st.info("Pas de dates exploitables.")
    with B:
        fig, ax = plt.subplots(figsize=(4,2.2))
        sns.barplot(y=freq_df["Thème"].head(5), x=freq_df["Fréquence (%)"].head(5), ax=ax)
        title(ax,"Thèmes dominants"); st.pyplot(fig, use_container_width=True)
        st.markdown("<div class='small'>Part d’avis citant explicitement chaque thème.</div>", unsafe_allow_html=True)
    with C:
        cats=["Positif","Neutre","Négatif"]; vals=[
            (df_syn["sentiment"]>0.05).mean()*100 if len(df_syn) else 0,
            ((df_syn["sentiment"]>=-0.05)&(df_syn["sentiment"]<=0.05)).mean()*100 if len(df_syn) else 0,
            (df_syn["sentiment"]<-0.05).mean()*100 if len(df_syn) else 0
        ]
        fig, ax = plt.subplots(figsize=(4,2.2))
        ax.bar(cats, vals); ax.set_ylim(0,100); title(ax,"Répartition des avis")
        st.pyplot(fig, use_container_width=True)
        st.markdown("<div class='small'>Équilibre global des ressentis sur la période.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
