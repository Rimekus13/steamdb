# tabs/anomalies.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import DEFAULT_FIGSIZE_WIDE, title, compact_time_axis

def render(st, ctx):
    df_f = ctx["df_f"]
    st.markdown("<div class='card'><h3>Anomalies hebdomadaires</h3>", unsafe_allow_html=True)
    if df_f["review_date"].notna().any() and len(df_f):
        z_thr=st.slider("Seuil z-score (anomalie)", 1.0, 3.0, 2.0, 0.1, key="anom_z_slider")
        ts=df_f.dropna(subset=["review_date"]).copy()
        ts["week"]=ts["review_date"].dt.to_period("W").dt.start_time
        agg=ts.groupby("week").agg(sent=("sentiment","mean"), n=("review_text","count")).reset_index()
        if len(agg)>=3:
            agg["z_sent"]=(agg["sent"]-agg["sent"].mean())/agg["sent"].std(ddof=0)
            agg["z_n"]=(agg["n"]-agg["n"].mean())/agg["n"].std(ddof=0)
            anomalies=agg[(agg["z_sent"].abs()>z_thr)|(agg["z_n"].abs()>z_thr)]
        else: anomalies=pd.DataFrame(columns=agg.columns)
        c1,c2=st.columns([1,1])
        with c1:
            fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_WIDE)
            ax.plot(agg["week"], agg["sent"])
            if not anomalies.empty: ax.scatter(anomalies["week"], anomalies["sent"], s=20)
            compact_time_axis(ax,3,6); title(ax,"Satisfaction (hebdo)")
            st.pyplot(fig, use_container_width=True)
        with c2:
            fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_WIDE)
            ax.plot(agg["week"], agg["n"])
            if not anomalies.empty: ax.scatter(anomalies["week"], anomalies["n"], s=20)
            compact_time_axis(ax,3,6); title(ax,"Volume d’avis (hebdo)")
            st.pyplot(fig, use_container_width=True)
        st.markdown("<div class='small'>Points = semaines atypiques (patch, promo, bad buzz…).</div>", unsafe_allow_html=True)
        if not anomalies.empty: st.dataframe(anomalies, use_container_width=True, height=220)
    else: st.info("Pas de dates exploitables.")
    st.markdown("</div>", unsafe_allow_html=True)
