# tabs/cooccurrences.py — Gold Firestore + fallback local
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from analysis import contains_any
from utils import title


def _gold_to_matrix(df_gold: pd.DataFrame, metric: str = "count") -> pd.DataFrame:
    """
    Transforme les paires (token_a, token_b, metric) en matrice carrée pour heatmap.
    - metric ∈ {"count", "percent"}
    """
    if df_gold is None or df_gold.empty:
        return pd.DataFrame()
    if metric not in df_gold.columns:
        return pd.DataFrame()

    # On garde uniquement les couples non vides
    g = df_gold.copy()
    g["token_a"] = g["token_a"].astype(str).str.strip()
    g["token_b"] = g["token_b"].astype(str).str.strip()
    g = g[(g["token_a"] != "") & (g["token_b"] != "")]
    if g.empty:
        return pd.DataFrame()

    # Construire la matrice symétrique
    tokens = sorted(set(g["token_a"]).union(set(g["token_b"])))
    M = pd.DataFrame(0.0, index=tokens, columns=tokens, dtype=float)

    for _, row in g.iterrows():
        a, b = row["token_a"], row["token_b"]
        v = float(row[metric])
        M.loc[a, b] += v
        M.loc[b, a] += v
        if a == b:
            M.loc[a, a] += v  # si jamais des self-pairs existent (rare)

    # Nettoyage: valeurs très faibles -> 0 pour lisibilité
    tiny = 1e-12 if metric == "percent" else 0.0
    M = M.where(M.abs() > tiny, 0.0)

    return M


def _render_heatmap(st, M: pd.DataFrame, mode_label: str, value_label: str, fmt: str = ".0f"):
    if M is None or M.empty:
        st.info("Aucune cooccurrence à afficher.")
        return

    # Limiter la taille si énorme (pour garder une UI fluide)
    max_tokens = st.sidebar.number_input("Limiter à N termes (triés par somme décroissante)", 10, 500, 50, step=10, key="cooc_limit_tokens")
    if len(M) > max_tokens:
        order = M.sum(axis=1).sort_values(ascending=False).index[:max_tokens]
        M = M.loc[order, order]

    show_values = st.checkbox("Afficher les valeurs sur la carte", value=True, key=f"cooc_show_values_{mode_label}")

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    sns.heatmap(
        M, cbar=True, cbar_kws={"shrink": 0.85, "label": value_label},
        ax=ax, annot=show_values, fmt=fmt, annot_kws={"size": 8},
        linewidths=0.4, linecolor="#ffffff"
    )
    title(ax, f"Carte des cooccurrences — {mode_label}")
    ax.set_xlabel("Terme B")
    ax.set_ylabel("Terme A")
    st.pyplot(fig, use_container_width=True)

    st.download_button(
        "⬇️ Télécharger (CSV)",
        data=M.to_csv().encode("utf-8"),
        file_name=f"cooccurrences_{mode_label.replace(' ', '_').lower()}.csv",
        mime="text/csv",
        key=f"cooc_dl_{mode_label}"
    )


def _render_local_theme_cooc(st, df_f: pd.DataFrame, theme_dict: dict):
    """Fallback: ta logique historique basée sur thèmes/mots-clés et df_f."""
    st.markdown("<div class='card'><h3>Co-mentions de thèmes (fallback local)</h3>", unsafe_allow_html=True)
    st.markdown(
        "<div class='small'>📌 Repérer quels <b>thèmes</b> sont souvent cités ensemble (calculés localement sur les avis filtrés).</div>",
        unsafe_allow_html=True
    )

    base_theme_dict = dict(theme_dict)
    if "active_theme_dict" not in st.session_state:
        st.session_state.active_theme_dict = base_theme_dict

    def dict_to_editable_text(d):
        lines = []
        for k in sorted(d.keys()):
            vals = [str(x) for x in d[k]]
            lines.append(f"{k}: {', '.join(vals)}")
        return "\n".join(lines)

    def parse_editable_text(raw):
        out = {}
        for line in str(raw).splitlines():
            if not line.strip() or ":" not in line:
                continue
            theme, rest = line.split(":", 1)
            theme = theme.strip()
            if not theme:
                continue
            kws = [w.strip().lower() for w in re.split(r"[;,]", rest) if w.strip()]
            seen = set(); kw_clean = []
            for w in kws:
                if w not in seen:
                    kw_clean.append(w); seen.add(w)
            if kw_clean:
                out[theme] = kw_clean
        return out

    with st.expander("⚙️ Personnaliser thèmes & mots-clés"):
        editable_text = st.text_area(
            "Thèmes et mots-clés (format : thème: mot1, mot2, mot3)",
            value=dict_to_editable_text(st.session_state.active_theme_dict),
            key="cooc_edit_text",
            height=160
        )
        cpa, cpb = st.columns([1,1])
        with cpa:
            if st.button("✅ Appliquer", key="cooc_apply_custom"):
                parsed = parse_editable_text(editable_text)
                if parsed:
                    st.session_state.active_theme_dict = parsed
                else:
                    st.warning("Aucun thème valide détecté.")
        with cpb:
            if st.button("↩️ Réinitialiser (défaut)", key="cooc_reset_custom"):
                st.session_state.active_theme_dict = base_theme_dict
                st.session_state.cooc_edit_text = dict_to_editable_text(base_theme_dict)

    th_dict = st.session_state.active_theme_dict
    themes_all = list(th_dict.keys())
    chosen_themes = st.multiselect("Thèmes à cartographier", options=themes_all, default=themes_all, key="cooc_themes_custom")

    if len(df_f) and chosen_themes:
        M = pd.DataFrame(0, index=chosen_themes, columns=chosen_themes, dtype=int)
        # la colonne texte à scanner : cleaned_review si dispo, sinon review_text
        text_col = "cleaned_review" if "cleaned_review" in df_f.columns else ("review_text" if "review_text" in df_f.columns else None)

        for _, row in df_f.iterrows():
            text = row.get(text_col, "") if text_col else ""
            present = [th for th in chosen_themes if contains_any(text, th_dict[th])]
            for i in range(len(present)):
                for j in range(i, len(present)):
                    M.loc[present[i], present[j]] += 1
                    if i != j:
                        M.loc[present[j], present[i]] += 1

        norm_mode = st.selectbox(
            "Mode d'affichage",
            ["Comptes bruts", "% du total", "% par ligne", "% par colonne"],
            index=0, key="cooc_norm_mode_custom"
        )
        M_view = M.copy()
        fmt = ".0f"; cbar_label = "Intensité co-mentions"
        if norm_mode == "% du total":
            total = float(M.values.sum()) or 1.0
            M_view = (M * 100.0 / total).round(2); fmt = ".1f"; cbar_label = "Pourcentage du total"
        elif norm_mode == "% par ligne":
            M_view = M.div(M.sum(axis=1).replace(0, np.nan), axis=0).mul(100.0).round(2).fillna(0); fmt = ".1f"; cbar_label = "% par ligne"
        elif norm_mode == "% par colonne":
            M_view = M.div(M.sum(axis=0).replace(0, np.nan), axis=1).mul(100.0).round(2).fillna(0); fmt = ".1f"; cbar_label = "% par colonne"

        fig, ax = plt.subplots(figsize=(6.8, 3.4))
        sns.heatmap(
            M_view, cbar=True, cbar_kws={"shrink": 0.85, "label": cbar_label},
            ax=ax, annot=True, fmt=fmt, annot_kws={"size": 8},
            linewidths=0.5, linecolor="#ffffff"
        )
        title(ax, "Carte des cooccurrences (thèmes cités ensemble)")
        ax.set_xlabel("Thème B"); ax.set_ylabel("Thème A")
        st.pyplot(fig, use_container_width=True)

        st.download_button(
            "⬇️ Télécharger ces données (CSV)",
            data=M_view.to_csv().encode("utf-8"),
            file_name="cooccurrences_local.csv",
            mime="text/csv",
            key="cooc_dl_custom"
        )
    else:
        st.info("Aucune donnée/thème sélectionné.")

    st.markdown("</div>", unsafe_allow_html=True)


def render(st, ctx):
    df_f = ctx["df_f"]
    theme_dict = ctx["theme_dict"]

    gold_counts_df: pd.DataFrame = ctx.get("gold_counts_df")
    gold_percent_df: pd.DataFrame = ctx.get("gold_percent_df")

    st.markdown("<div class='card'><h3>Cooccurrences (Gold Firestore)</h3>", unsafe_allow_html=True)
    st.markdown(
        "<div class='small'>📌 Analyse basée sur le <b>Gold</b> (paires de tokens par fenêtre et période) "
        "avec fallback local si Gold indisponible.</div>",
        unsafe_allow_html=True
    )

    use_gold = st.toggle("Utiliser Gold Firestore (sinon calcul local par thèmes)", value=True, key="cooc_use_gold")

    if use_gold and (gold_counts_df is not None and not gold_counts_df.empty):
        # Choix de la métrique
        metric_choice = st.radio("Métrique", ["Comptes (count)", "Pourcentage (percent)"], index=0, horizontal=True, key="cooc_metric")
        # Filtrage rapide sur période (si le DAG a stocké period)
        periods = sorted([p for p in (gold_counts_df.get("period").dropna().unique().tolist() if "period" in gold_counts_df.columns else []) if p])
        if metric_choice == "Pourcentage (percent)" and gold_percent_df is not None and not gold_percent_df.empty:
            periods = sorted(set(periods).union(
                [p for p in (gold_percent_df.get("period").dropna().unique().tolist() if "period" in gold_percent_df.columns else []) if p]
            ))

        sel_period = st.selectbox("Période (YYYY-MM)", options=["Toutes"] + periods, index=0, key="cooc_period")

        # Sélection du DF en fonction de la métrique
        if metric_choice == "Comptes (count)":
            df_gold = gold_counts_df.copy()
            value_col = "count"
            fmt = ".0f"
            label = "Comptes"
        else:
            df_gold = gold_percent_df.copy() if (gold_percent_df is not None and not gold_percent_df.empty) else pd.DataFrame()
            value_col = "percent"
            fmt = ".2f"
            label = "Pourcentage"

        if df_gold is None or df_gold.empty:
            st.warning("Gold 'percent' indisponible, bascule sur 'count'.")
            df_gold = gold_counts_df.copy()
            value_col = "count"; fmt = ".0f"; label = "Comptes"

        # Filtre période
        if sel_period != "Toutes" and "period" in df_gold.columns:
            df_gold = df_gold[df_gold["period"] == sel_period]

        # Option de sous-ensemble top-K (si tri déjà fait côté app.py, c’est ok)
        topk = st.number_input("Top K lignes (0=toutes)", min_value=0, value=0, step=10, key="cooc_topk_viz")
        if topk and topk > 0 and len(df_gold) > topk:
            # On trie par valeur descendante pour ne garder que les meilleures lignes
            df_gold = df_gold.sort_values(value_col, ascending=False).head(topk)

        # Conversion en matrice & rendu
        M = _gold_to_matrix(df_gold, metric=value_col)
        _render_heatmap(st, M, mode_label=f"Gold ({label})", value_label=label, fmt=fmt)

        # Affichage table brute
        with st.expander("📄 Voir les données brutes (Gold)"):
            st.dataframe(df_gold.reset_index(drop=True))

    else:
        # Fallback: calcul local par thèmes sur df_f
        _render_local_theme_cooc(st, df_f, theme_dict)

    st.markdown("</div>", unsafe_allow_html=True)
