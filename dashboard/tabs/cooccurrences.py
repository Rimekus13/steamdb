# tabs/cooccurrences.py
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from analysis import contains_any
from utils import title

def render(st, ctx):
    df_f = ctx["df_f"]
    theme_dict = ctx["theme_dict"]

    st.markdown("<div class='card'><h3>Co-mentions de thèmes</h3>", unsafe_allow_html=True)
    st.markdown(
        "<div class='small'>📌 Repérer quels <b>thèmes</b> sont souvent cités ensemble.</div>",
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
        return "\\n".join(lines)

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
        for _, row in df_f.iterrows():
            text = row.get("cleaned_review", "")
            present = [th for th in chosen_themes if contains_any(text, th_dict[th])]
            for i in range(len(present)):
                for j in range(i, len(present)):
                    M.loc[present[i], present[j]] += 1
                    if i != j:
                        M.loc[present[j], present[i]] += 1

        norm_mode = st.selectbox("Mode d'affichage", ["Comptes bruts", "% du total", "% par ligne", "% par colonne"], index=0, key="cooc_norm_mode_custom")
        show_values = st.checkbox("Afficher les valeurs sur la carte", value=True, key="cooc_show_values_custom")

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
        sns.heatmap(M_view, cbar=True, cbar_kws={"shrink": 0.85, "label": cbar_label}, ax=ax, annot=show_values, fmt=fmt, annot_kws={"size": 8}, linewidths=0.5, linecolor="#ffffff")
        title(ax, "Carte des cooccurrences (thèmes cités ensemble)")
        ax.set_xlabel("Thème B"); ax.set_ylabel("Thème A")
        st.pyplot(fig, use_container_width=True)

        st.download_button("⬇️ Télécharger ces données (CSV)", data=M_view.to_csv().encode("utf-8"), file_name="cooccurrences.csv", mime="text/csv", key="cooc_dl_custom")
    else:
        st.info("Aucune donnée/thème sélectionné.")
    st.markdown("</div>", unsafe_allow_html=True)
