"""
Olist | Data Lakehouse : Dashboard Streamlit (style "bento" multi-thèmes)
Lit UNIQUEMENT les tables Gold (HDFS / Delta Lake) :
  - gold_monthly_evolution      - gold_top_month_by_status
  - gold_pareto_order_status    - olist_customer_segments
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pyspark.sql import SparkSession

st.set_page_config(page_title="Olist · Data Lakehouse", page_icon=":material/analytics:", layout="wide",
                   initial_sidebar_state="collapsed")

GOLD_ROOT = "hdfs://namenode:9000/lakehouse/gold"
MONTHS_FR = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
FEATURES = ["total_spent", "number_orders", "avg_order_value", "avg_delivery_delay"]
SEG_ORDER = ["Petits acheteurs", "Acheteurs intermédiaires", "Gros acheteurs occasionnels", "Clients récurrents"]
SEG_ICONS = {
    "Petits acheteurs": "savings",
    "Acheteurs intermédiaires": "balance",
    "Gros acheteurs occasionnels": "diamond",
    "Clients récurrents": "autorenew",
}
NAV = ["Dashboard", "Segments", "Statuts", "Données"]
NAV_ICONS = {"Dashboard": "dashboard", "Segments": "groups", "Statuts": "donut_large", "Données": "table_chart"}
THEME_ICONS = {"Nuit": "dark_mode", "Ardoise": "contrast", "Clair": "light_mode"}

# ============================================================
# THÈMES (nuit / ardoise / clair)
# ============================================================

DARK_PAL = ["#19E3F2", "#2F7BFF", "#1AA7C9", "#5B7FB8", "#2A5C9A", "#8FA6C6"]
LIGHT_PAL = ["#0E1E4A", "#19D3E6", "#3D6EB4", "#5E8BD0", "#7FE7F0", "#A5AEC4"]

THEMES = {
    "Nuit": dict(
        bg="linear-gradient(160deg,#04142B 0%,#071F3D 55%,#0A2A4F 100%)", topbar="#0A1F3A", card="#0C2441",
        alt="#0A1F3A", hero="linear-gradient(150deg,#14407D 0%,#0B2650 100%)", ring_inner="#0E2D5E",
        text="#E8F4FF", muted="#7FA0C4", alt_text="#E8F4FF", alt_muted="#7FA0C4",
        accent="#19E3F2", accent_soft="rgba(25,227,242,0.16)", border="rgba(25,227,242,0.16)",
        shadow="10px 10px 26px rgba(0,0,0,.45), -4px -4px 14px rgba(70,130,200,.06)",
        grid="rgba(127,160,196,0.14)", alt_grid="rgba(127,160,196,0.14)", bar_off="#1B3B63", pal=DARK_PAL),
    "Ardoise": dict(
        bg="linear-gradient(160deg,#7D90B3 0%,#98A9C7 100%)", topbar="#14264F", card="#F1F4FB",
        alt="#16295C", hero="linear-gradient(150deg,#1D3F8A 0%,#102358 100%)", ring_inner="#12296A",
        text="#14264F", muted="#6B7A99", alt_text="#EAF2FF", alt_muted="#9FB3D9",
        accent="#19D3E6", accent_soft="rgba(25,211,230,0.18)", border="rgba(20,38,79,0.10)",
        shadow="10px 10px 24px rgba(40,55,95,.35), -6px -6px 16px rgba(255,255,255,.35)",
        grid="rgba(20,38,79,0.10)", alt_grid="rgba(159,179,217,0.18)", bar_off="#3A4868", pal=LIGHT_PAL),
    "Clair": dict(
        bg="linear-gradient(160deg,#E4E8F8 0%,#F2F0FB 100%)", topbar="#F7F9FE", card="#F3F6FD",
        alt="#16295C", hero="linear-gradient(150deg,#1D3F8A 0%,#102358 100%)", ring_inner="#12296A",
        text="#14264F", muted="#6B7A99", alt_text="#EAF2FF", alt_muted="#9FB3D9",
        accent="#19D3E6", accent_soft="rgba(25,211,230,0.20)", border="rgba(20,38,79,0.08)",
        shadow="10px 10px 24px rgba(163,177,198,.45), -8px -8px 18px rgba(255,255,255,.95)",
        grid="rgba(20,38,79,0.08)", alt_grid="rgba(159,179,217,0.18)", bar_off="#3A4868", pal=LIGHT_PAL),
}

# --- état initial (lu AVANT d'injecter le CSS) ---
st.session_state.setdefault("theme", "Nuit")
st.session_state.setdefault("nav", "Dashboard")
st.session_state.setdefault("metric", "Mensuel")
theme_name = st.session_state.get("theme") if st.session_state.get("theme") in THEMES else "Nuit"
T = THEMES[theme_name]
topbar_dark = theme_name != "Clair"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,500,0,0&display=swap');
:root{--text:__text__;--muted:__muted__;--accent:__accent__;--card:__card__;--border:__border__;}
html, body, .stApp, [class*="css"]{font-family:'Plus Jakarta Sans',sans-serif;}
.stApp{background:__bg__;color:var(--text);}
.block-container{max-width:1500px;padding:1.1rem 1.6rem 3rem;}
header[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="collapsedControl"]{display:none;}
.stApp label, .stApp label p, .stApp [data-testid="stWidgetLabel"] *{color:var(--text) !important;}

/* cartes neumorphiques */
div[class*="st-key-card"]{background:var(--card);border:1px solid var(--border);border-radius:24px;
  padding:18px 20px;box-shadow:__shadow__;color:var(--text);margin-bottom:6px;}
div[class*="st-key-card_alt"]{background:__alt__;--text:__alt_text__;--muted:__alt_muted__;color:__alt_text__;}
div[class*="st-key-card_hero"]{background:__hero__;--text:__alt_text__;--muted:__alt_muted__;color:__alt_text__;}
div[class*="st-key-topbar"]{background:__topbar__;--text:__bar_text__;--muted:__bar_muted__;color:__bar_text__;
  border:1px solid var(--border);border-radius:22px;padding:10px 20px;box-shadow:__shadow__;margin-bottom:14px;}
div[class*="st-key-card"] label, div[class*="st-key-card"] label p{color:var(--text) !important;}

/* textes */
.logo{display:flex;align-items:center;gap:10px;font-weight:800;font-size:1.15rem;color:var(--text);}
.logo i{font-style:normal;background:linear-gradient(135deg,#19E3F2,#2F7BFF);border-radius:10px;width:34px;height:34px;display:grid;place-items:center;color:#fff;}
.mi{font-family:'Material Symbols Rounded';font-weight:normal;font-style:normal;font-size:20px;line-height:1;letter-spacing:normal;text-transform:none;display:inline-block;white-space:nowrap;direction:ltr;vertical-align:middle;font-feature-settings:'liga';-webkit-font-feature-settings:'liga';}
.ct .mi{margin-right:6px;color:var(--accent);}
.ct-row{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;}
.ct{font-weight:700;font-size:1.02rem;color:var(--text);}
.ct-right{font-weight:700;font-size:1.05rem;color:var(--text);}
.sub{color:var(--muted);font-size:.78rem;}
.big{font-weight:800;font-size:2rem;line-height:1.1;color:var(--text);}
.mid{font-weight:800;font-size:1.45rem;color:var(--text);}
.pos{color:#34D399;font-weight:700;} .neg{color:#FF6B8A;font-weight:700;}
.hero-top{display:flex;justify-content:space-between;align-items:center;}
.ring{--p:50;width:58px;height:58px;border-radius:50%;display:grid;place-items:center;
  background:conic-gradient(var(--accent) calc(var(--p)*1%), rgba(255,255,255,.2) 0);}
.ring span{width:46px;height:46px;border-radius:50%;background:__ring_inner__;display:grid;place-items:center;
  font-weight:700;font-size:.78rem;color:#fff;}
.prog{height:7px;border-radius:99px;background:rgba(127,160,196,.25);overflow:hidden;margin:4px 0 8px;}
.prog b{display:block;height:100%;border-radius:99px;}
.seg-row{display:grid;grid-template-columns:30px 1fr 52px;gap:10px;align-items:center;margin:9px 0;}
.seg-ico{width:30px;height:30px;border-radius:9px;display:grid;place-items:center;background:rgba(127,160,196,.22);color:var(--accent);}
.seg-name{font-size:.8rem;font-weight:600;color:var(--text);}
.seg-pct{text-align:right;font-size:.8rem;font-weight:700;color:var(--text);}
.lg-r{display:grid;grid-template-columns:12px 1fr 52px 78px;gap:8px;align-items:center;font-size:.78rem;margin:7px 0;color:var(--text);}
.lg-r .dot{width:9px;height:9px;border-radius:50%;} .lg-r .p,.lg-r .v{text-align:right;} .lg-r .p{color:var(--muted);}
.lg-r .v{font-weight:700;}
.bubble{background:rgba(127,160,196,.16);border:1px solid var(--border);border-radius:16px;padding:10px 14px;text-align:center;}
.bubble b{font-size:1.3rem;font-weight:800;color:var(--text);}
.insight{background:__accent_soft__;border:1px solid var(--border);border-radius:14px;padding:12px 16px;color:var(--text);}

/* boutons arrondis (nav, année, thème) */
[data-testid="stButtonGroup"]{background:transparent !important;gap:6px;}
button[data-testid^="stBaseButton-pills"], button[data-testid^="stBaseButton-segmented_control"]{
  background:transparent !important;border:1px solid var(--border) !important;border-radius:999px !important;
  color:var(--text) !important;font-weight:600;}
button[data-testid="stBaseButton-pillsActive"], button[data-testid="stBaseButton-segmented_controlActive"],
[data-testid="stButtonGroup"] button[data-testid$="Active"], [data-testid="stButtonGroup"] button[aria-checked="true"],
[data-testid="stButtonGroup"] button[aria-pressed="true"]{
  background:__accent_soft__ !important;border-color:var(--accent) !important;color:var(--accent) !important;}
[data-testid="stButtonGroup"] button, [data-baseweb="button-group"] button{
  background:transparent !important;border:1px solid var(--border) !important;border-radius:999px !important;
  color:var(--text) !important;font-weight:600;}
[data-testid="stButtonGroup"] button[data-testid$="Active"], [data-testid="stButtonGroup"] button[aria-checked="true"],
[data-testid="stButtonGroup"] button[aria-pressed="true"]{background:__accent_soft__ !important;border-color:var(--accent) !important;color:var(--accent) !important;}
[data-testid="stButtonGroup"] button *{color:inherit !important;}
button[data-testid^="stBaseButton-pills"] p, button[data-testid^="stBaseButton-segmented_control"] p{color:inherit !important;}
div[data-testid="stExpander"]{background:var(--card);border:1px solid var(--border);border-radius:14px;}
div[data-testid="stExpander"] summary, div[data-testid="stExpander"] summary *{color:var(--text) !important;}
.stButton button, .stDownloadButton button{background:#2F7BFF;color:#fff;border:0;border-radius:999px;}
div[data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:12px;}
</style>
"""
tokens = {**T, "bar_text": T["alt_text"] if topbar_dark else T["text"],
          "bar_muted": T["alt_muted"] if topbar_dark else T["muted"]}
for k, v in tokens.items():
    CSS = CSS.replace(f"__{k}__", str(v))
st.markdown(CSS, unsafe_allow_html=True)

# ============================================================
# SPARK & CHARGEMENT DES TABLES GOLD
# ============================================================


@st.cache_resource
def get_spark():
    return (SparkSession.builder.appName("OlistGoldDashboard")
            .config("spark.driver.memory", "512m").config("spark.ui.enabled", "false")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true").getOrCreate())


@st.cache_data(ttl=300, show_spinner=False)
def load_gold_table(name: str) -> pd.DataFrame:
    return get_spark().read.format("delta").load(f"{GOLD_ROOT}/{name}").toPandas()


with st.spinner("Chargement des tables Gold depuis HDFS…"):
    try:
        monthly_raw = load_gold_table("gold_monthly_evolution")
        status_raw = load_gold_table("gold_top_month_by_status")
        customers_raw = load_gold_table("olist_customer_segments")
    except Exception as e:  # noqa: BLE001
        st.error("Impossible de charger les tables Gold. Vérifie que HDFS et le NameNode sont démarrés.")
        st.exception(e)
        st.stop()
    try:
        pareto_raw = load_gold_table("gold_pareto_order_status")
    except Exception:  # noqa: BLE001
        pareto_raw = None

# ============================================================
# PRÉPARATION
# ============================================================

monthly = monthly_raw.copy()
monthly["periode_dt"] = pd.to_datetime(monthly["periode"], format="%Y-%m", errors="coerce")
monthly["montant_total"] = pd.to_numeric(monthly["montant_total"], errors="coerce")
monthly = monthly.dropna(subset=["periode_dt", "montant_total"]).sort_values("periode_dt").reset_index(drop=True)
monthly["croissance_pct"] = monthly["montant_total"].pct_change() * 100
monthly["moyenne_mobile_3m"] = monthly["montant_total"].rolling(3, min_periods=1).mean()
monthly["annee"] = monthly["periode_dt"].dt.year
monthly["mois"] = monthly["periode_dt"].dt.month
years = sorted(monthly["annee"].unique().tolist())
annual = monthly.groupby("annee")["montant_total"].sum()

status = status_raw.copy()
status["montant_total"] = pd.to_numeric(status["montant_total"], errors="coerce")
status = status.dropna(subset=["montant_total"])

pareto = None
if pareto_raw is not None and not pareto_raw.empty:
    pareto = pareto_raw.copy()
    for c in ["nombre_articles", "nombre_commandes", "montant_total", "montant_median", "montant_iqr", "pourcentage_cumule"]:
        if c in pareto.columns:
            pareto[c] = pd.to_numeric(pareto[c], errors="coerce")
    pareto = pareto.sort_values("montant_total", ascending=False).reset_index(drop=True)
    if pareto["pourcentage_cumule"].max() <= 1.0001:
        pareto["pourcentage_cumule"] *= 100

customers = customers_raw.copy()
for c in FEATURES:
    customers[c] = pd.to_numeric(customers[c], errors="coerce")
customers = customers.dropna(subset=FEATURES + ["segment"])

# ============================================================
# OUTILS
# ============================================================


def compact(v):
    a = abs(v)
    if a >= 1e6:
        return f"R$ {v / 1e6:,.2f} M"
    if a >= 1e3:
        return f"R$ {v / 1e3:,.1f} k"
    return f"R$ {v:,.0f}"


def short(v):
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.1f}M"
    if a >= 1e3:
        return f"{v / 1e3:.0f}k"
    return f"{v:.0f}"


def seg_color(name):
    i = SEG_ORDER.index(name) if name in SEG_ORDER else len(SEG_ORDER)
    return T["pal"][i % len(T["pal"])]


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def icon(name, size=20):
    return f'<span class="mi" style="font-size:{size}px">{name}</span>'


def card_title(title, right=""):
    html(f'<div class="ct-row"><span class="ct">{title}</span><span class="ct-right">{right}</span></div>')


def base(fig, h, alt=False, legend=False):
    txt = T["alt_text"] if alt else T["text"]
    grid = T["alt_grid"] if alt else T["grid"]
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=h,
                      font=dict(family="Plus Jakarta Sans", color=txt, size=11),
                      margin=dict(l=6, r=6, t=8, b=6), showlegend=legend,
                      legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.2),
                      hoverlabel=dict(bgcolor=T["alt"], font_color=T["alt_text"]))
    fig.update_xaxes(gridcolor=grid, zeroline=False, linecolor="rgba(0,0,0,0)")
    fig.update_yaxes(gridcolor=grid, zeroline=False, linecolor="rgba(0,0,0,0)")
    return fig


def plot(fig, h=260, alt=False, legend=False):
    st.plotly_chart(base(fig, h, alt, legend), width="stretch", config={"displayModeBar": False})


def donut(labels, values, colors, big, small, h=230):
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.72, sort=False, textinfo="none",
                           marker=dict(colors=colors, line=dict(color="rgba(0,0,0,0)", width=0))))
    fig.add_annotation(text=f"<b>{big}</b><br><span style='font-size:11px;opacity:.7'>{small}</span>",
                       showarrow=False, font=dict(size=19, color=T["text"]))
    plot(fig, h)


def legend_rows(rows):
    out = "".join(
        f'<div class="lg-r"><span class="dot" style="background:{c}"></span><span>{n}</span>'
        f'<span class="p">{p}</span><span class="v">{v}</span></div>' for c, n, p, v in rows)
    html(out)


def prog(pct, color):
    return f'<div class="prog"><b style="width:{max(0, min(100, pct)):.1f}%;background:{color}"></b></div>'


def csv_button(df, filename, key):
    st.download_button("Exporter en CSV", df.to_csv(index=False).encode("utf-8-sig"), filename, "text/csv", key=key,
                       icon=":material/download:")


def explain(label, sql, why, language="sql"):
    with st.expander(f":material/manage_search: Requête SQL & Justification statistique · {label}"):
        st.code(sql.strip(), language=language)
        st.markdown(why)


SQL_MONTHLY = """
WITH monthly AS (
    SELECT DATE_FORMAT(order_purchase_timestamp, 'yyyy-MM') AS periode,
           SUM(item_total_amount) AS montant_total
    FROM v_silver GROUP BY DATE_FORMAT(order_purchase_timestamp, 'yyyy-MM')
), comparison AS (
    SELECT periode, montant_total,
           LAG(montant_total, 1) OVER (ORDER BY periode) AS montant_precedent
    FROM monthly
)
SELECT periode, montant_total, montant_precedent,
       montant_total - montant_precedent AS evolution,
       ROUND((montant_total - montant_precedent) / montant_precedent * 100, 2) AS taux_evolution_pct
FROM comparison ORDER BY periode
"""
SQL_PARETO = """
WITH aggregation AS (
    SELECT order_status, COUNT(*) AS nombre_articles, COUNT(DISTINCT order_id) AS nombre_commandes,
           SUM(item_total_amount) AS montant_total,
           PERCENTILE_APPROX(item_total_amount, 0.5) AS montant_median,
           PERCENTILE_APPROX(item_total_amount, 0.75) - PERCENTILE_APPROX(item_total_amount, 0.25) AS montant_iqr
    FROM v_silver WHERE order_status IS NOT NULL GROUP BY order_status
)
SELECT *, DENSE_RANK() OVER (ORDER BY montant_total DESC) AS rang_pareto,
       SUM(montant_total) OVER (ORDER BY montant_total DESC
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) / SUM(montant_total) OVER () * 100 AS pourcentage_cumule
FROM aggregation ORDER BY rang_pareto
"""
SQL_TOP_MONTH = """
WITH agg AS (
    SELECT order_status, DATE_FORMAT(order_purchase_timestamp, 'yyyy-MM') AS periode,
           SUM(item_total_amount) AS montant_total
    FROM v_silver GROUP BY order_status, DATE_FORMAT(order_purchase_timestamp, 'yyyy-MM')
), classement AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY order_status ORDER BY montant_total DESC) AS rang FROM agg
)
SELECT order_status, periode, montant_total FROM classement WHERE rang = 1 ORDER BY montant_total DESC
"""
CODE_SEGMENTS = """
customers_df = (orders_df.join(customers_raw.select("customer_id", "customer_unique_id"), "customer_id")
    .groupBy("customer_unique_id")
    .agg(F.sum("order_amount").alias("total_spent"), F.countDistinct("order_id").alias("number_orders"),
         F.avg("order_amount").alias("avg_order_value"), F.avg("delivery_delay").alias("avg_delivery_delay")))
X = StandardScaler().fit_transform(np.log1p(entities_pd[features]))
entities_pd["cluster_id"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X)
"""

# ============================================================
# BARRE DU HAUT : logo · navigation · année · thème
# ============================================================

if st.session_state.get("year") not in [str(y) for y in years]:
    st.session_state["year"] = str(years[-1])

with st.container(key="topbar"):
    c1, c2, c3, c4 = st.columns([1.5, 3.0, 2.2, 2.6], vertical_alignment="center")
    with c1:
        html(f'<div class="logo"><i>{icon("analytics", 20)}</i>Olist Analytics</div>')
    with c2:
        st.segmented_control("Navigation", NAV, key="nav", label_visibility="collapsed",
                             format_func=lambda x: f":material/{NAV_ICONS[x]}: {x}")
    with c3:
        st.pills("Année", [str(y) for y in years], key="year", label_visibility="collapsed")
    with c4:
        st.segmented_control("Thème", list(THEMES), key="theme", label_visibility="collapsed",
                             format_func=lambda x: f":material/{THEME_ICONS[x]}: {x}")

page = st.session_state.get("nav") or "Dashboard"
Y = int(st.session_state.get("year") or years[-1])

# filtres clients
with st.expander(":material/tune: Filtres clients", expanded=False):
    f1, f2, f3 = st.columns(3)
    seg_all = sorted(customers["segment"].unique().tolist())
    sel_segments = f1.multiselect("Segments", seg_all, default=seg_all)
    s_min, s_max = float(customers["total_spent"].min()), float(customers["total_spent"].max())
    spend_range = f2.slider("Dépense totale (R$)", s_min, s_max, (s_min, s_max), step=1.0)
    d_hi = float(np.ceil(customers["avg_delivery_delay"].max()))
    delay_cap = f3.slider("Retard moyen max (jours)", float(np.floor(customers["avg_delivery_delay"].min())),
                          d_hi, d_hi, step=1.0, help="Valeur négative = livraison en avance.")

fc = customers[customers["segment"].isin(sel_segments)
               & customers["total_spent"].between(*spend_range)
               & (customers["avg_delivery_delay"] <= delay_cap)].copy()


def seg_table(df):
    s = (df.groupby("segment").agg(clients=("segment", "size"), depense_mediane=("total_spent", "median"),
                                   commandes_mediane=("number_orders", "median"),
                                   panier_median=("avg_order_value", "median"),
                                   retard_median=("avg_delivery_delay", "median")).reset_index())
    s["part"] = s["clients"] / s["clients"].sum() * 100
    s["ordre"] = s["segment"].map(lambda x: SEG_ORDER.index(x) if x in SEG_ORDER else 99)
    return s.sort_values("ordre").drop(columns="ordre").reset_index(drop=True)


# ============================================================
# PAGE : DASHBOARD (grille de cartes)
# ============================================================

if page == "Dashboard":
    ry = monthly[monthly["annee"] == Y]
    ys = ry.set_index("mois")["montant_total"].reindex(range(1, 13))
    yg = ry.set_index("mois")["croissance_pct"].reindex(range(1, 13))
    prev_s = None
    if (Y - 1) in years:
        prev_s = monthly[monthly["annee"] == Y - 1].set_index("mois")["montant_total"].reindex(range(1, 13))
    last_row, li = ry.iloc[-1], ry.index[-1]
    delta = last_row["montant_total"] - monthly["montant_total"].iloc[li - 1] if li > 0 else np.nan
    share = annual[Y] / annual.sum() * 100
    seg = seg_table(fc) if not fc.empty else None

    left, right = st.columns([3.3, 1.25], gap="medium")

    with left:
        a, b = st.columns([1.15, 2.1], gap="medium")

        # ---- Carte hero : chiffre d'affaires ----
        with a:
            with st.container(key="card_hero"):
                chg = (f'<span class="{"pos" if delta >= 0 else "neg"}">{"+" if delta >= 0 else "−"}{short(abs(delta))}</span>'
                       if not np.isnan(delta) else "n/d")
                html(f'<div class="hero-top"><div><div class="ct">Chiffre d\'affaires {Y}</div>'
                     f'<div class="big">{compact(annual[Y])}</div></div>'
                     f'<div class="ring" style="--p:{share:.0f}"><span>{share:.0f}%</span></div></div>'
                     f'<div style="margin-top:10px"><div class="mid">{compact(annual.sum())}</div>'
                     f'<div class="sub">Cumul toutes années · l\'anneau = part de {Y}</div></div>')
                spark = go.Figure(go.Scatter(x=MONTHS_FR, y=ys.fillna(0), mode="lines", line=dict(color="#19E3F2", width=3, shape="spline"),
                                             fill="tozeroy", fillcolor="rgba(25,227,242,0.15)"))
                top3 = ys.nlargest(3)
                spark.add_trace(go.Scatter(x=[MONTHS_FR[m - 1] for m in top3.index], y=top3.values, mode="markers",
                                           marker=dict(size=9, color="#fff", line=dict(color="#19E3F2", width=3))))
                spark.update_xaxes(visible=False)
                spark.update_yaxes(visible=False)
                plot(spark, 110, alt=True)
                html(f'<div class="sub">Variation du dernier mois ({last_row["periode"]}) : {chg} · {len(ry)} mois de données</div>')

            # ---- Carte segments (type "objectifs") ----
            with st.container(key="card_segments"):
                kept = len(fc) / len(customers) * 100
                card_title("Segments", f"{len(fc):,} clients")
                html(f'<div class="sub">Clients retenus par les filtres · {kept:.0f} %</div>' + prog(kept, T["accent"]))
                if seg is None:
                    st.info("Aucun client pour ces filtres.")
                else:
                    html("".join(
                        f'<div class="seg-row"><div class="seg-ico">{icon(SEG_ICONS.get(r.segment, "person"), 18)}</div>'
                        f'<div><div class="seg-name">{r.segment}</div>{prog(r.part, seg_color(r.segment))}</div>'
                        f'<div class="seg-pct">{r.part:.1f}%</div></div>' for r in seg.itertuples()))

        with b:
            # ---- Carte CA (type "inflow") ----
            with st.container(key="card_inflow"):
                h1, h2, h3 = st.columns([1.3, 2.2, 1.2], vertical_alignment="center")
                h1.markdown('<span class="ct">Chiffre d\'affaires</span>', unsafe_allow_html=True)
                with h2:
                    st.segmented_control("Vue", ["Mensuel", "Croissance", "Cumul"], key="metric", label_visibility="collapsed")
                h3.markdown(f'<div class="ct-right" style="text-align:right">{compact(annual[Y])}</div>', unsafe_allow_html=True)
                view = st.session_state.get("metric") or "Mensuel"
                fig = go.Figure()
                if view == "Mensuel":
                    v = ys.fillna(0)
                    top = set(ys.nlargest(3).index)
                    fig.add_trace(go.Bar(x=MONTHS_FR, y=v, marker=dict(color=[T["accent"] if m in top else T["bar_off"] for m in v.index],
                                                                      cornerradius=7),
                                         text=[short(x) if m in top else "" for m, x in v.items()], textposition="outside",
                                         cliponaxis=False, hovertemplate="%{x}: %{y:,.0f} R$<extra></extra>"))
                elif view == "Croissance":
                    g = yg.clip(-150, 150)
                    fig.add_trace(go.Bar(x=MONTHS_FR, y=g, marker=dict(color=[T["accent"] if (x or 0) >= 0 else "#FF6B8A" for x in yg],
                                                                      cornerradius=7),
                                         customdata=yg, hovertemplate="%{x}: %{customdata:.1f} %<extra></extra>"))
                    fig.update_yaxes(ticksuffix=" %")
                else:
                    fig.add_trace(go.Scatter(x=MONTHS_FR, y=ys.fillna(0).cumsum(), mode="lines", line=dict(color=T["accent"], width=3, shape="spline"),
                                             fill="tozeroy", fillcolor="rgba(25,227,242,0.15)"))
                fig.update_yaxes(tickformat="~s")
                plot(fig, 300)

            # ---- Carte répartition clients (type "budget") ----
            with st.container(key="card_budget"):
                card_title("Clients par segment", f"{len(fc):,}")
                if seg is None:
                    st.info("Aucun client pour ces filtres.")
                else:
                    d1, d2 = st.columns([1, 1.3], vertical_alignment="center")
                    with d1:
                        donut(seg["segment"], seg["clients"], [seg_color(s) for s in seg["segment"]],
                              short(len(fc)), "clients", 220)
                    with d2:
                        rec = (fc["number_orders"] >= 2).mean() * 100
                        html(f'<div class="sub">Clients récurrents · {rec:.1f} %</div>' + prog(rec, T["accent"])
                             + f'<div class="sub">Achat unique · {100 - rec:.1f} %</div>' + prog(100 - rec, T["pal"][1]))
                        legend_rows([(seg_color(r.segment), r.segment, f"{r.part:.0f}%", f"R$ {r.depense_mediane:,.0f}")
                                     for r in seg.itertuples()])
                        html(f'<div class="bubble"><b>R$ {fc["total_spent"].median():,.0f}</b><div class="sub">Dépense médiane / client</div></div>')

        # ---- Bas : comparaison & plan annuel ----
        c, d = st.columns([1.5, 1], gap="medium")
        with c:
            with st.container(key="card_alt_compare"):
                t1, t2, t3, t4 = st.columns([1.5, 1.2, 1.3, 0.9], vertical_alignment="center")
                t1.markdown('<span class="ct">Comparer les années</span>', unsafe_allow_html=True)
                show_prev = t2.toggle(f"{Y - 1}", value=prev_s is not None, disabled=prev_s is None, key="tg_prev")
                show_ma = t3.toggle("Moy. mobile 3 m", value=False, key="tg_ma")
                t4.markdown(f'<div class="ct-right" style="text-align:right">{short(annual[Y])}</div>', unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=MONTHS_FR, y=ys, name=str(Y), mode="lines+markers", line=dict(color="#19E3F2", width=3, shape="spline"),
                                         fill="tozeroy", fillcolor="rgba(25,227,242,0.14)", connectgaps=False))
                if show_prev and prev_s is not None:
                    fig.add_trace(go.Scatter(x=MONTHS_FR, y=prev_s, name=str(Y - 1), mode="lines", line=dict(color="#5B8CFF", width=2.5, shape="spline")))
                if show_ma:
                    ma = ry.set_index("mois")["moyenne_mobile_3m"].reindex(range(1, 13))
                    fig.add_trace(go.Scatter(x=MONTHS_FR, y=ma, name="Moy. mobile", mode="lines", line=dict(color="#FBBF24", width=2, dash="dash")))
                fig.update_yaxes(tickformat="~s")
                plot(fig, 230, alt=True, legend=True)
        with d:
            with st.container(key="card_alt_annual"):
                card_title("CA par année", short(annual.sum()))
                fig = go.Figure(go.Bar(x=[str(y) for y in annual.index], y=annual.values,
                                       marker=dict(color=["#19E3F2" if y == Y else "#3F63B5" for y in annual.index], cornerradius=7),
                                       text=[short(v) for v in annual.values], textposition="outside", cliponaxis=False))
                fig.update_yaxes(visible=False)
                plot(fig, 230, alt=True)

    with right:
        # ---- Radar ----
        with st.container(key="card_radar"):
            card_title("Profil des segments", "médianes")
            if seg is None:
                st.info("Aucun client pour ces filtres.")
            else:
                cols = ["depense_mediane", "commandes_mediane", "panier_median", "retard_median"]
                labels = ["Dépense", "Commandes", "Panier", "Retard"]
                n = seg.set_index("segment")[cols]
                n = (n - n.min()) / (n.max() - n.min()).replace(0, 1)
                fig = go.Figure()
                for name, row in n.iterrows():
                    fig.add_trace(go.Scatterpolar(r=row.tolist() + [row.iloc[0]], theta=labels + [labels[0]], name=name, fill="toself",
                                                  opacity=0.45, line=dict(color=seg_color(name), width=2)))
                fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                             radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor=T["grid"]),
                                             angularaxis=dict(gridcolor=T["grid"])))
                plot(fig, 260)
                st.caption("Médianes normalisées entre segments (0 à 1).")

        # ---- Statuts ----
        with st.container(key="card_costs"):
            if pareto is None:
                card_title("Statuts de commande")
                st.info("Table gold_pareto_order_status introuvable.")
            else:
                tot = pareto["montant_total"].sum()
                top5 = pareto.head(5)
                others = pareto.iloc[5:]["montant_total"].sum()
                labels = top5["order_status"].tolist() + (["autres"] if others > 0 else [])
                values = top5["montant_total"].tolist() + ([others] if others > 0 else [])
                colors = T["pal"][:len(labels)]
                card_title("Statuts de commande", compact(tot))
                donut(labels, values, colors, compact(tot), "CA total", 240)
                legend_rows([(c_, l_, f"{v_ / tot * 100:.1f}%", compact(v_)) for c_, l_, v_ in zip(colors, labels, values)])

    st.write("")
    html(f'<div class="insight"><b>À retenir ·</b> {Y} représente {share:.0f} % du chiffre d\'affaires cumulé '
         f'({compact(annual[Y])}). Meilleur mois : <b>{ry.loc[ry["montant_total"].idxmax(), "periode"]}</b> '
         f'({compact(ry["montant_total"].max())}).'
         + (f' Les clients récurrents (≥ 2 commandes) ne sont que <b>{(fc["number_orders"] >= 2).mean() * 100:.1f} %</b> des clients filtrés.' if not fc.empty else "")
         + '</div>')
    st.write("")
    explain("Chiffre d'affaires", SQL_MONTHLY,
            "Agrégation **mensuelle** pour lisser le bruit quotidien. Le sélecteur d'année filtre cette table Gold ; la **moyenne mobile 3 mois**, "
            "le cumul et la comparaison N-1 sont calculés côté application sur quelques dizaines de lignes. `LAG` donne la variation "
            "période-sur-période sans auto-jointure. Les années 2016 et 2018 sont partielles : compare-les avec prudence.")
    explain("Statuts de commande", SQL_PARETO,
            "**Pareto** : `DENSE_RANK` + somme cumulée en fenêtre donnent la concentration de la valeur par statut. "
            "On affiche aussi des indicateurs **robustes** (médiane, IQR) car les montants sont très asymétriques (skewness ≈ 7,7).")
    explain("Segments clients", CODE_SEGMENTS,
            "`log1p` réduit l'asymétrie des dépenses, `StandardScaler` met les 4 variables à la même échelle avant le K-Means. "
            "Les segments sont décrits par leurs **médianes** (robustes aux valeurs extrêmes) ; le radar est normalisé en min-max uniquement "
            "pour superposer des unités différentes.", language="python")

# ============================================================
# PAGE : SEGMENTS
# ============================================================

elif page == "Segments":
    if fc.empty:
        st.warning("Aucun client ne correspond aux filtres.")
        st.stop()
    seg = seg_table(fc)
    cols = st.columns(len(seg), gap="medium")
    for i, (col, r) in enumerate(zip(cols, seg.itertuples())):
        with col:
            with st.container(key=f"card_segstat_{i}"):
                html(f'<div class="ct">{icon(SEG_ICONS.get(r.segment, "person"), 20)}{r.segment}</div>'
                     f'<div class="big">{r.clients:,}</div><div class="sub">{r.part:.1f} % · médiane R$ {r.depense_mediane:,.0f} · '
                     f'{r.commandes_mediane:.0f} cmd · retard {r.retard_median:.1f} j</div>' + prog(r.part, seg_color(r.segment)))

    cmap = {s: seg_color(s) for s in seg["segment"]}
    a, b = st.columns(2, gap="medium")
    with a:
        with st.container(key="card_seg_box"):
            card_title("Distribution des dépenses", "échelle log")
            fig = px.box(fc[fc["total_spent"] > 0], x="segment", y="total_spent", color="segment", color_discrete_map=cmap, log_y=True, points=False)
            fig.update_layout(xaxis_title="", yaxis_title="R$")
            plot(fig, 330)
    with b:
        with st.container(key="card_seg_hist"):
            card_title("Retard de livraison", "−40 à +60 jours")
            fig = px.histogram(fc[fc["avg_delivery_delay"].between(-40, 60)], x="avg_delivery_delay", color="segment",
                               color_discrete_map=cmap, nbins=60, barmode="overlay", opacity=0.65)
            fig.update_layout(xaxis_title="jours", yaxis_title="clients")
            plot(fig, 330, legend=True)
    a, b = st.columns([1.6, 1], gap="medium")
    with a:
        with st.container(key="card_seg_scatter"):
            samp = fc[fc["total_spent"] > 0]
            samp = samp.sample(min(len(samp), 6000), random_state=42)
            card_title("Dépense vs retard", f"échantillon de {len(samp):,} clients")
            fig = px.scatter(samp, x="total_spent", y="avg_delivery_delay", color="segment", color_discrete_map=cmap, log_x=True,
                             opacity=0.5, hover_data=["number_orders", "avg_order_value"])
            fig.update_yaxes(range=[-40, 70])
            fig.update_layout(xaxis_title="Dépense totale (R$, log)", yaxis_title="Retard (j)")
            plot(fig, 360, legend=True)
    with b:
        with st.container(key="card_seg_corr"):
            card_title("Corrélations", "Spearman")
            lab = {"total_spent": "Dépense", "number_orders": "Commandes", "avg_order_value": "Panier", "avg_delivery_delay": "Retard"}
            corr = fc[FEATURES].corr(method="spearman").rename(index=lab, columns=lab)
            fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                            color_continuous_scale=[[0, T["pal"][1]], [0.5, T["card"]], [1, T["pal"][0]]])
            fig.update_layout(coloraxis_showscale=False)
            plot(fig, 360)
    explain("Segments clients", CODE_SEGMENTS,
            "Boîtes à moustaches en **échelle log** : médiane et IQR comparables malgré les valeurs extrêmes. Corrélation de **Spearman** : "
            "mesure un lien monotone sur les rangs, adaptée aux variables asymétriques (Pearson serait biaisé). "
            "Le nuage est échantillonné (6 000 points) pour rester fluide.", language="python")

# ============================================================
# PAGE : STATUTS
# ============================================================

elif page == "Statuts":
    if pareto is not None:
        with st.container(key="card_pareto"):
            card_title("Diagramme de Pareto par statut", "montants en échelle log")
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=pareto["order_status"], y=pareto["montant_total"], name="Montant (R$)",
                                 marker=dict(color=T["accent"], cornerradius=7), text=[compact(v) for v in pareto["montant_total"]],
                                 textposition="outside"), secondary_y=False)
            fig.add_trace(go.Scatter(x=pareto["order_status"], y=pareto["pourcentage_cumule"], name="% cumulé", mode="lines+markers",
                                     line=dict(color="#FBBF24", width=3)), secondary_y=True)
            fig.add_trace(go.Scatter(x=pareto["order_status"], y=[80] * len(pareto), name="Seuil 80 %", mode="lines",
                                     line=dict(color="#FF6B8A", dash="dot")), secondary_y=True)
            fig.update_yaxes(type="log", secondary_y=False)
            fig.update_yaxes(range=[0, 105], ticksuffix=" %", secondary_y=True, showgrid=False)
            plot(fig, 400, legend=True)
        explain("Pareto", SQL_PARETO,
                "`DENSE_RANK` classe les statuts, la somme cumulée en fenêtre divisée par le total donne le % cumulé. Échelle **log** car un statut "
                "écrase les autres de plusieurs ordres de grandeur. Médiane et IQR décrivent un article typique sans biais des extrêmes.")
        with st.container(key="card_pareto_table"):
            card_title("Indicateurs robustes par statut")
            v = pareto[["rang_pareto", "order_status", "nombre_commandes", "nombre_articles", "montant_total", "montant_median",
                        "montant_iqr", "pourcentage_cumule"]].round(2)
            st.dataframe(v, width="stretch", hide_index=True)
            csv_button(v, "olist_pareto_statuts.csv", "dl_pareto")
    else:
        st.warning("La table gold_pareto_order_status est introuvable.")
    with st.container(key="card_topmonth"):
        card_title("Meilleur mois par statut", "échelle log")
        ss = status.sort_values("montant_total")
        fig = px.bar(ss, x="montant_total", y="order_status", orientation="h", text="periode", color_discrete_sequence=[T["accent"]])
        fig.update_traces(marker_cornerradius=7, textposition="inside")
        fig.update_xaxes(type="log", title="R$")
        fig.update_yaxes(title="")
        plot(fig, 380)
    explain("Meilleur mois par statut", SQL_TOP_MONTH,
            "`ROW_NUMBER() OVER (PARTITION BY order_status ORDER BY montant_total DESC) = 1` isole le mois record de chaque statut en une passe.")

# ============================================================
# PAGE : DONNÉES
# ============================================================

else:
    tabs = st.tabs(["Clients", "Évolution mensuelle", "Meilleur mois / statut", "Pareto statuts"])
    with tabs[0]:
        st.caption(f"{len(fc):,} clients sur {len(customers):,} (filtres actifs).")
        cols = st.multiselect("Colonnes", fc.columns.tolist(), default=[c for c in fc.columns if c != "cluster_id"])
        q = st.text_input("Rechercher un customer_unique_id")
        v = fc[cols] if cols else fc
        if q and "customer_unique_id" in v.columns:
            v = v[v["customer_unique_id"].astype(str).str.contains(q, case=False, na=False)]
        st.dataframe(v.head(5000), width="stretch", hide_index=True)
        st.caption("Affichage limité à 5 000 lignes ; l'export contient toutes les lignes filtrées.")
        csv_button(v, "olist_customer_segments.csv", "dl_c")
    with tabs[1]:
        m = monthly.drop(columns=["periode_dt"]).round(2)
        st.dataframe(m, width="stretch", hide_index=True)
        csv_button(m, "olist_monthly.csv", "dl_m")
    with tabs[2]:
        st.dataframe(status, width="stretch", hide_index=True)
        csv_button(status, "olist_top_month.csv", "dl_s")
    with tabs[3]:
        if pareto is None:
            st.info("Table gold_pareto_order_status introuvable.")
        else:
            st.dataframe(pareto.round(2), width="stretch", hide_index=True)
            csv_button(pareto, "olist_pareto.csv", "dl_p")

st.write("")
html(f'<div class="sub" style="text-align:center">Olist Data Lakehouse · Bronze → Silver → Gold · Spark · HDFS · Delta Lake · Streamlit · Plotly</div>')
