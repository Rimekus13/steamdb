# tabs/themes.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import DEFAULT_FIGSIZE_WIDE, title, compact_time_axis
from analysis import contains_any, top_unigrams_bigrams, pick_examples

def render(st, ctx):
    df_f = ctx["df_f"]
    theme_dict = ctx["theme_dict"]

    st.markdown("<div class='card'><h3>Analyse par thème</h3>", unsafe_allow_html=True)

    tc1, tc2, tc3 = st.columns([1,1,1])
    with tc1: theme_choice = st.selectbox("Thème ciblé", options=list(theme_dict.keys()), key="topics_theme_choice")
    with tc2: tmin = st.slider("Sentiment min.", -1.0, 1.0, -1.0, 0.05, key="topic_min_slider")
    with tc3: tmax = st.slider("Sentiment max.", -1.0, 1.0, 1.0, 0.05, key="topic_max_slider")

    df_theme = df_f[df_f["cleaned_review"].apply(lambda t: contains_any(t, theme_dict[theme_choice]))].copy()
    df_theme = df_theme[(df_theme["sentiment"]>=tmin) & (df_theme["sentiment"]<=tmax)]

    if df_theme.empty:
        st.info("Aucun avis pour ce thème avec les filtres actuels."); st.markdown("</div>", unsafe_allow_html=True)
        return

    p=(df_theme["sentiment"]>0.05).mean()*100; n=((df_theme["sentiment"]>=-0.05)&(df_theme["sentiment"]<=0.05)).mean()*100; g=(df_theme["sentiment"]<-0.05).mean()*100
    k1,k2,k3,k4 = st.columns(4)
    with k1: st.markdown(f"<div class='kpi-xl kpi-neu'><div class='hl'>🧩 Avis (thème)</div><div class='val'>{len(df_theme):,}</div><div class='tag'>Nombre d'avis concernés</div></div>", unsafe_allow_html=True)
    with k2: st.markdown(f"<div class='kpi-xl kpi-pos'><div class='hl'>💬 Sentiment moyen</div><div class='val'>{df_theme['sentiment'].mean():.2f}</div><div class='tag'>De -1 à +1</div></div>", unsafe_allow_html=True)
    with k3: st.markdown(f"<div class='kpi-xl kpi-neu'><div class='hl'>📊 % Pos / Neu / Neg</div><div class='val'>{p:.0f}% / {n:.0f}% / {g:.0f}%</div><div class='tag'>Répartition</div></div>", unsafe_allow_html=True)
    with k4: st.markdown(f"<div class='kpi-xl kpi-len'><div class='hl'>📝 Mots / avis (moy.)</div><div class='val'>{df_theme['cleaned_review'].str.split().apply(len).replace(0,np.nan).mean():.1f}</div><div class='tag'>Engagement</div></div>", unsafe_allow_html=True)

    r1,r2 = st.columns([1.4, 1])
    with r1:
        if df_theme["review_date"].notna().any():
            ts = df_theme.dropna(subset=["review_date"]).copy()
            ts["date"] = ts["review_date"].dt.to_period("W").dt.start_time
            series = ts.groupby("date")["sentiment"].mean()
            fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_WIDE)
            ax.plot(series.index, series.values)
            compact_time_axis(ax,3,6); title(ax, f"Tendance hebdo – {theme_choice}")
            ax.set_xlabel(""); ax.set_ylabel("Score")
            st.pyplot(fig, use_container_width=True)
            st.markdown("<div class='small'>Évolution du ressenti ; utile pour mesurer l’effet des patchs.</div>", unsafe_allow_html=True)
        else: st.info("Pas de dates exploitables pour la tendance de ce thème.")
    with r2:
        fig, ax = plt.subplots(figsize=(4.2,2.2))
        sns.histplot(df_theme["sentiment"], bins=24, kde=True, ax=ax)
        title(ax,"Distribution des scores"); st.pyplot(fig, use_container_width=True)
        st.markdown("<div class='small'>Retour plutôt mixte ou polarisé ?</div>", unsafe_allow_html=True)

    uni, big = top_unigrams_bigrams(df_theme["cleaned_review"].dropna().tolist(), n_top=15)
    c1,c2 = st.columns([1,1])
    with c1:
        st.markdown("**Top mots – thème**")
        st.dataframe(pd.DataFrame(uni, columns=["Mot","Fréquence"]), use_container_width=True, height=220)
        st.markdown("<div class='small'>Vocabulaire le plus associé à ce thème.</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("**Top expressions (bigrams)**")
        st.dataframe(pd.DataFrame(big, columns=["Bigramme","Fréquence"]), use_container_width=True, height=220)
        st.markdown("<div class='small'>Expressions fréquentes (ex. “crash serveur”, “drop fps”).</div>", unsafe_allow_html=True)

    pos_ex, neg_ex = pick_examples(df_theme, n=3)
    e1,e2 = st.columns([1,1])
    with e1:
        st.markdown("**Exemples positifs (résumé)**")
        for t in pos_ex: st.write(f"• {t}")
        st.markdown("<div class='small'>Ce que les joueurs apprécient sur ce thème.</div>", unsafe_allow_html=True)
    with e2:
        st.markdown("**Exemples négatifs (résumé)**")
        for t in neg_ex: st.write(f"• {t}")
        st.markdown("<div class='small'>Douleurs récurrentes à prioriser.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
