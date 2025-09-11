# tabs/sentiment.py — couleurs fixes + visuels lisibles (vert/gris/rouge)
import pandas as pd
import numpy as np
import plotly.express as px

def _ensure_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "sentiment" not in df.columns:
        df["sentiment"] = 0.0
    if "sentiment_label" not in df.columns:
        def lab(x):
            try:
                if x > 0.05: return "positif"
                if x < -0.05: return "négatif"
                return "neutre"
            except Exception:
                return "neutre"
        df["sentiment_label"] = df["sentiment"].apply(lab)
    return df

def _colors_from_ctx(ctx):
    return ctx.get("sentiment_colors", {
        "positif": "#22c55e",
        "neutre":  "#9ca3af",
        "négatif": "#ef4444",
    })

def render(st, ctx):
    df = ctx["df_f"].copy()
    if df.empty:
        st.info("Aucun avis avec les filtres actuels.")
        return

    df = _ensure_labels(df)
    colmap = _colors_from_ctx(ctx)

    # Répartition
    st.subheader("Répartition du sentiment")
    c1, c2 = st.columns([1,1])
    counts = df["sentiment_label"].value_counts(dropna=False).rename_axis("sentiment_label").reset_index(name="n")

    with c1:
        if counts.empty:
            st.write("—")
        else:
            fig = px.pie(
                counts, names="sentiment_label", values="n",
                hole=0.55, color="sentiment_label",
                color_discrete_map=colmap
            )
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if counts.empty:
            st.write("—")
        else:
            fig = px.bar(
                counts.sort_values("sentiment_label"),
                x="sentiment_label", y="n",
                color="sentiment_label",
                color_discrete_map=colmap,
            )
            fig.update_layout(xaxis_title="", yaxis_title="Nombre d'avis", margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

    # Evolution
    st.subheader("Évolution du sentiment dans le temps")
    if "review_date" in df.columns and df["review_date"].notna().any():
        tmp = df.dropna(subset=["review_date"]).copy()
        tmp["date"] = tmp["review_date"].dt.date
        daily = tmp.groupby(["date","sentiment_label"]).size().reset_index(name="n")
        total = tmp.groupby("date").size().reset_index(name="tot")
        share = daily.merge(total, on="date")
        share["part"] = share["n"] / share["tot"]
        cat_order = ["négatif","neutre","positif"]
        share["sentiment_label"] = pd.Categorical(share["sentiment_label"], categories=cat_order, ordered=True)
        fig = px.area(
            share.sort_values("date"),
            x="date", y="part",
            color="sentiment_label",
            color_discrete_map=colmap,
        )
        fig.update_yaxes(tickformat=".0%", range=[0,1])
        fig.update_layout(legend_title="", xaxis_title="", yaxis_title="Part d'avis", margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Pas de date disponible pour tracer l'évolution.")

    # Distribution
    st.subheader("Distribution du score (compound)")
    dd = df[["sentiment","sentiment_label"]].dropna().copy()
    if dd.empty:
        st.write("—")
        return
    dd["bin"] = pd.cut(dd["sentiment"], bins=[-1.0,-0.05,0.05,1.0], labels=["négatif","neutre","positif"], include_lowest=True)
    hist = dd.groupby("bin").size().reindex(["négatif","neutre","positif"]).fillna(0).astype(int).reset_index(name="n")
    fig = px.bar(
        hist, x="bin", y="n",
        color="bin", color_discrete_map={"positif":colmap["positif"], "neutre":colmap["neutre"], "négatif":colmap["négatif"]}
    )
    fig.update_layout(xaxis_title="", yaxis_title="Nombre d'avis", margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)
