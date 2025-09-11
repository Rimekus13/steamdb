# tabs/langues.py
import matplotlib.pyplot as plt
from utils import title

def render(st, ctx):
    df_f = ctx["df_f"]
    st.markdown("<div class='card'><h3>Langues</h3>", unsafe_allow_html=True)
    if len(df_f):
        lang_agg = df_f.groupby("language").agg(avis=("review_text","count"), sentiment_moy=("sentiment","mean")).reset_index().sort_values("avis", ascending=False)
        top = lang_agg.head(10)
        c1,c2 = st.columns([1,1])
        with c1:
            fig, ax = plt.subplots(figsize=(5,2.2))
            ax.barh(top["language"], top["avis"]); ax.invert_yaxis()
            title(ax,"Volume (Top 10)"); st.pyplot(fig, use_container_width=True)
            st.markdown("<div class='small'>Langues les plus actives.</div>", unsafe_allow_html=True)
        with c2:
            fig, ax = plt.subplots(figsize=(5,2.2))
            ax.barh(top["language"], top["sentiment_moy"]); ax.invert_yaxis()
            title(ax,"Sentiment moyen (Top 10)"); st.pyplot(fig, use_container_width=True)
            st.markdown("<div class='small'>Comparer satisfaction par langue.</div>", unsafe_allow_html=True)
    else: st.info("Aucune donnée filtrée.")
    st.markdown("</div>", unsafe_allow_html=True)
