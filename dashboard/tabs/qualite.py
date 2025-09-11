# tabs/qualite.py
def render(st, ctx):
    df_f = ctx["df_f"]
    st.markdown("<div class='card'><h3>Qualité des données</h3>", unsafe_allow_html=True)
    if len(df_f):
        cols=["review_text","language","review_date"]
        comp={c:f"{df_f[c].notna().mean()*100:.1f}%" if c in df_f.columns else "absent" for c in cols}
        st.write("**Complétude (non-nulles)** :", comp)
        dup_rate=(df_f.duplicated(subset=["review_text","review_date"]).mean()*100) if {"review_text","review_date"}.issubset(df_f.columns) else 0.0
        st.write(f"**Duplicats estimés** : {dup_rate:.1f}%")
        st.markdown("<div class='small'>Une bonne complétude et peu de doublons = indicateurs fiables.</div>", unsafe_allow_html=True)
    else: st.info("Aucune donnée filtrée.")
    st.markdown("</div>", unsafe_allow_html=True)
