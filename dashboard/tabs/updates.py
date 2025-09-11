# tabs/updates.py
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from utils import title
def render(st, ctx):
    df_f = ctx["df_f"]
    st.markdown("<div class='card'><h3>Impact d’une mise à jour</h3>", unsafe_allow_html=True)
    if df_f["review_date"].notna().any() and len(df_f):
        df_f_min = df_f["review_date"].min().date()
        df_f_max = df_f["review_date"].max().date()
        pivot=st.date_input("Date pivot", value=df_f_min, min_value=df_f_min, max_value=df_f_max, key="pivot_input")
        before=df_f[df_f["review_date"].dt.date < pivot]["sentiment"].mean()
        after =df_f[df_f["review_date"].dt.date >= pivot]["sentiment"].mean()
        var=(after-before) if (pd.notna(before) and pd.notna(after)) else np.nan
        c1,c2,c3=st.columns(3)
        c1.markdown(f"<div class='kpi-xl kpi-neu'><div class='hl'>🕒 Avant</div><div class='val'>{before if pd.notna(before) else 0:.2f}</div><div class='tag'>Moyenne</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='kpi-xl kpi-neu'><div class='hl'>⚡ Après</div><div class='val'>{after if pd.notna(after) else 0:.2f}</div><div class='tag'>Moyenne</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='kpi-xl kpi-pos'><div class='hl'>Δ Variation</div><div class='val'>{var if pd.notna(var) else 0:+.2f}</div><div class='tag'>Impact ressenti</div></div>", unsafe_allow_html=True)
        tdf=df_f.dropna(subset=["review_date"]).copy(); tdf["period"]=np.where(tdf["review_date"].dt.date < pivot, "Avant", "Après")
        fig, ax = plt.subplots(figsize=(5,2.2))
        sns.boxplot(data=tdf, x="period", y="sentiment", hue="period", legend=False, ax=ax)
        title(ax,"Avant / Après (sentiment)"); ax.set_xlabel("")
        st.pyplot(fig, use_container_width=True)
        st.markdown("<div class='small'>Comparer rapidement l’effet d’un patch ou d’une mise à jour majeure.</div>", unsafe_allow_html=True)
    else: st.info("Pas de dates exploitables.")
    st.markdown("</div>", unsafe_allow_html=True)
