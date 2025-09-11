# tabs/longueur.py
import numpy as np
import matplotlib.pyplot as plt
from utils import title

def render(st, ctx):
    df_f = ctx["df_f"]
    st.markdown("<div class='card'><h3>Longueur des avis & ressenti</h3>", unsafe_allow_html=True)
    if len(df_f):
        df_f = df_f.copy()
        df_f["word_count"]=df_f["cleaned_review"].str.split().apply(len)
        wc_max=int(np.nanpercentile(df_f["word_count"],99)) if len(df_f) else 500
        wc_cut=st.slider("Max mots/avis (trim)", 20, max(40, wc_max), wc_max, key="wc_cut_slider")
        df_l=df_f[df_f["word_count"]<=wc_cut]
        c1,c2=st.columns([1,1])
        with c1:
            fig, ax = plt.subplots(figsize=(5,2.2))
            ax.hist(df_l["word_count"], bins=40)
            title(ax,"Distribution longueur d’avis"); st.pyplot(fig, use_container_width=True)
            st.markdown("<div class='small'>Répartition des tailles d’avis (trim 99e pct).</div>", unsafe_allow_html=True)
        with c2:
            fig, ax = plt.subplots(figsize=(5,2.2))
            ax.scatter(df_l["word_count"], df_l["sentiment"], s=8, alpha=0.35)
            title(ax,"Sentiment vs longueur"); ax.set_xlabel("Mots/avis"); ax.set_ylabel("Score")
            st.pyplot(fig, use_container_width=True)
            st.markdown("<div class='small'>Les avis longs sont-ils plus critiques ?</div>", unsafe_allow_html=True)
    else: st.info("Aucune donnée filtrée.")
    st.markdown("</div>", unsafe_allow_html=True)
