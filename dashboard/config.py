# config.py
import seaborn as sns

APP_TITLE = "Avis Steam – Insights"
LAYOUT = "wide"

# Colors & theme
PRIMARY = "#2563eb"
GOOD    = "#16a34a"
WARN    = "#f59e0b"
BAD     = "#dc2626"
GREY    = "#6b7280"
BORDER  = "#e5e7eb"

# Matplotlib / seaborn style
sns.set_style("whitegrid")

# CSS injected into Streamlit
BASE_CSS = f"""
<style>
.card {{ background:#fff; border:1px solid {BORDER}; border-radius:12px; padding:12px 14px 10px; margin-bottom:12px; }}
.small {{ font-size:12px; opacity:.9; }}
h3 {{ margin:0 0 6px 0; }}

/* KPI XXL */
.kpi-xl {{
  border: 1px solid {BORDER};
  border-radius: 14px;
  padding: 14px 16px;
  background: #ffffff;
  box-shadow: 0 1px 0 rgba(0,0,0,.02);
  display: flex; flex-direction: column; gap: 6px;
  height: 100%;
}}
.kpi-xl .hl {{ font-size: 14px; font-weight: 600; color:#111827; display:flex; align-items:center; gap:6px; }}
.kpi-xl .val {{ font-size: 30px; line-height: 1; font-weight: 800; color:#111827; }}
.kpi-xl .tag {{ font-size: 12px; color:#4b5563; }}
.kpi-pos {{ background: linear-gradient(180deg,#ecfdf5 0,#ffffff 60%); border-color:#d1fae5; }}
.kpi-neu {{ background: linear-gradient(180deg,#f3f4f6 0,#ffffff 60%); border-color:#e5e7eb; }}
.kpi-len {{ background: linear-gradient(180deg,#fef3c7 0,#ffffff 60%); border-color:#fde68a; }}
.kpi-lang {{ background: linear-gradient(180deg,#dbeafe 0,#ffffff 60%); border-color:#bfdbfe; }}
</style>
"""
