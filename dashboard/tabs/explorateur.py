# tabs/explorateur.py
import re

def render(st, ctx):
    df_f = ctx["df_f"]
    st.markdown("<div class='card'><h3>Explorateur d’avis</h3>", unsafe_allow_html=True)
    view_cols=[c for c in ["review_date","language","sentiment","playtime_hours","review_text"] if c in df_f.columns]
    e1,e2=st.columns([2,1])
    with e1: local_kw=st.text_input("Filtrer par mots (explorateur)", value="", key="explore_kw")
    with e2: smin,smax=st.slider("Plage de sentiment", -1.0, 1.0, (-1.0,1.0), 0.05, key="explore_sent_range")
    df_view_base=df_f[(df_f["sentiment"]>=smin)&(df_f["sentiment"]<=smax)].copy()
    if local_kw.strip():
        pattern=r"\\b(" + "|".join([re.escape(k.strip()) for k in local_kw.split(",") if k.strip()]) + r")\\b"
        df_view_base=df_view_base[df_view_base["cleaned_review"].str.contains(pattern, regex=True, na=False)]
    def highlight_keywords(text,kws):
        if not kws or not isinstance(text,str): return text
        pattern=r"(" + "|".join([re.escape(k) for k in kws]) + r")"; return re.sub(pattern,r"<mark>\\1</mark>", text, flags=re.IGNORECASE)
    if view_cols and len(df_view_base):
        df_view=df_view_base[view_cols].copy()
        if local_kw.strip():
            kws=[k.strip() for k in local_kw.split(",") if k.strip()]
            df_view["review_text"]=df_view["review_text"].astype(str).apply(lambda t: highlight_keywords(t,kws))
            st.write("Les mots saisis sont **surlignés** dans les avis.")
        st.write(df_view.to_html(escape=False, index=False), unsafe_allow_html=True)
        csv=df_view_base[view_cols].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export CSV (avis filtrés)", data=csv, file_name="avis_filtres.csv", mime="text/csv", key="explore_download")
    else: st.info("Aucune donnée à afficher.")
    st.markdown("</div>", unsafe_allow_html=True)
