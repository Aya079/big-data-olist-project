
import streamlit as st
import plotly.express as px
import pandas as pd
from pyspark.sql import SparkSession

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Olist | Data Lakehouse",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
}
[data-testid="stMetric"] {
    background: #172554;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #263b70;
}
[data-testid="stMetricLabel"] {
    color: #bfdbfe;
}
[data-testid="stMetricValue"] {
    color: white;
}
h1, h2, h3 {
    color: #e0e7ff;
}
div[data-testid="stExpander"] {
    background-color: #111c33;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Olist — Data Lakehouse Analytics")
st.caption("Analyse des ventes, des commandes et de la segmentation clients | Architecture Bronze → Silver → Gold")

# ============================================================
# CONNEXION SPARK ET LECTURE GOLD UNIQUEMENT
# ============================================================

@st.cache_resource
def get_spark():
    return (
        SparkSession.builder
        .appName("OlistGoldDashboard")
        .config("spark.driver.memory", "512m")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


@st.cache_data(ttl=300)
def load_gold_table(table_name):
    spark = get_spark()
    path = f"hdfs://namenode:9000/lakehouse/gold/{table_name}"
    return spark.read.format("delta").load(path).toPandas()


try:
    monthly = load_gold_table("gold_monthly_evolution")
    status = load_gold_table("gold_top_month_by_status")
    customers = load_gold_table("olist_customer_segments")
except Exception as e:
    st.error("Impossible de charger les tables Gold depuis HDFS.")
    st.exception(e)
    st.stop()


# ============================================================
# PREPARATION
# ============================================================

monthly["periode"] = pd.to_datetime(monthly["periode"], format="%Y-%m")
monthly = monthly.sort_values("periode")

status["montant_total"] = pd.to_numeric(
    status["montant_total"], errors="coerce"
)

numeric_columns = [
    "total_spent",
    "number_orders",
    "avg_order_value",
    "avg_delivery_delay"
]

for col in numeric_columns:
    customers[col] = pd.to_numeric(customers[col], errors="coerce")

customers = customers.dropna(subset=numeric_columns + ["segment"])


# ============================================================
# SIDEBAR : FILTRES
# ============================================================

st.sidebar.title("🎛️ Filtres")

segments_available = sorted(customers["segment"].unique())

selected_segments = st.sidebar.multiselect(
    "Segments clients",
    options=segments_available,
    default=segments_available
)

min_spent = float(customers["total_spent"].min())
max_spent = float(customers["total_spent"].max())

if min_spent < max_spent:
    spend_range = st.sidebar.slider(
        "Dépenses totales par client (€)",
        min_value=min_spent,
        max_value=max_spent,
        value=(min_spent, max_spent)
    )
else:
    spend_range = (min_spent, max_spent)

filtered_customers = customers[
    customers["segment"].isin(selected_segments)
    & customers["total_spent"].between(
        spend_range[0], spend_range[1]
    )
]


# ============================================================
# INDICATEURS CLES
# ============================================================

st.header("Vue générale")

total_revenue = monthly["montant_total"].sum()
total_customers = len(customers)
total_orders = customers["number_orders"].sum()
average_spent = customers["total_spent"].median()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Chiffre d'affaires cumulé", f"{total_revenue:,.0f} €")
c2.metric("Clients uniques", f"{total_customers:,.0f}")
c3.metric("Commandes livrées", f"{total_orders:,.0f}")
c4.metric("Dépense médiane / client", f"{average_spent:,.2f} €")

st.divider()


# ============================================================
# 1. EVOLUTION MENSUELLE
# ============================================================

st.header("1. Évolution mensuelle des ventes")

fig_monthly = px.line(
    monthly,
    x="periode",
    y="montant_total",
    markers=True,
    title="Évolution du chiffre d'affaires mensuel",
    labels={
        "periode": "Mois",
        "montant_total": "Montant total (€)"
    },
    template="plotly_dark"
)

fig_monthly.update_layout(
    paper_bgcolor="#0b1120",
    plot_bgcolor="#111c33",
    hovermode="x unified"
)

st.plotly_chart(fig_monthly, use_container_width=True)

with st.expander("Requête SQL & Justification statistique"):
    st.code("""
WITH monthly AS (
    SELECT
        DATE_FORMAT(order_purchase_timestamp, 'yyyy-MM') AS periode,
        SUM(item_total_amount) AS montant_total
    FROM v_silver
    GROUP BY DATE_FORMAT(order_purchase_timestamp, 'yyyy-MM')
),
comparison AS (
    SELECT
        periode,
        montant_total,
        LAG(montant_total, 1) OVER (
            ORDER BY periode
        ) AS montant_precedent
    FROM monthly
)
SELECT
    periode,
    montant_total,
    montant_precedent,
    montant_total - montant_precedent AS evolution
FROM comparison
ORDER BY periode
""", language="sql")

    st.markdown("""
    **Justification statistique :**

    La somme des montants permet de suivre le volume des ventes
    au fil du temps. La fonction LAG compare chaque mois au
    précédent pour mesurer l'évolution mensuelle.

    L'analyse temporelle permet de repérer les périodes de hausse,
    de baisse et les variations saisonnières éventuelles.
    """)


# ============================================================
# 2. MEILLEUR MOIS PAR STATUT
# ============================================================

st.header("2. Analyse des commandes par statut")

fig_status = px.bar(
    status.sort_values("montant_total", ascending=False),
    x="order_status",
    y="montant_total",
    color="order_status",
    text_auto=".2s",
    title="Meilleur mois en montant pour chaque statut",
    labels={
        "order_status": "Statut de commande",
        "montant_total": "Montant total (€)"
    },
    template="plotly_dark"
)

fig_status.update_layout(
    paper_bgcolor="#0b1120",
    plot_bgcolor="#111c33",
    showlegend=False
)

st.plotly_chart(fig_status, use_container_width=True)

with st.expander("Requête SQL & Justification statistique"):
    st.code("""
WITH aggregation_mensuelle AS (
    SELECT
        order_status,
        DATE_FORMAT(order_purchase_timestamp, 'yyyy-MM') AS periode,
        SUM(item_total_amount) AS montant_total
    FROM v_silver
    GROUP BY
        order_status,
        DATE_FORMAT(order_purchase_timestamp, 'yyyy-MM')
),
classement AS (
    SELECT
        order_status,
        periode,
        montant_total,
        ROW_NUMBER() OVER (
            PARTITION BY order_status
            ORDER BY montant_total DESC
        ) AS rang
    FROM aggregation_mensuelle
)
SELECT order_status, periode, montant_total
FROM classement
WHERE rang = 1
ORDER BY montant_total DESC
""", language="sql")

    st.markdown("""
    **Justification statistique :**

    La fonction ROW_NUMBER classe les mois selon le montant
    total pour chaque statut. Le filtre rang = 1 conserve
    le mois ayant le montant le plus élevé.

    Cette méthode permet de comparer les meilleures périodes
    des différentes catégories de statut.
    """)


# ============================================================
# 3. SEGMENTATION CLIENTS
# ============================================================

st.header("3. Segmentation et typologie des clients")

if filtered_customers.empty:
    st.warning("Aucun client ne correspond aux filtres sélectionnés.")
    st.stop()

segment_counts = (
    filtered_customers["segment"]
    .value_counts()
    .rename_axis("segment")
    .reset_index(name="nombre_clients")
)

fig_segments = px.pie(
    segment_counts,
    names="segment",
    values="nombre_clients",
    hole=0.48,
    title="Répartition des clients par segment",
    template="plotly_dark"
)

fig_segments.update_layout(
    paper_bgcolor="#0b1120",
    plot_bgcolor="#111c33"
)

st.plotly_chart(fig_segments, use_container_width=True)

with st.expander("Requête SQL & Justification statistique"):
    st.code("""
-- Agrégation métier équivalente à l'agrégation PySpark
SELECT
    customer_unique_id,
    SUM(order_amount) AS total_spent,
    COUNT(DISTINCT order_id) AS number_orders,
    AVG(order_amount) AS avg_order_value,
    AVG(delivery_delay) AS avg_delivery_delay
FROM orders_with_unique_id
GROUP BY customer_unique_id;

-- Les features sont ensuite transformées avec log1p,
-- normalisées avec StandardScaler et segmentées par K-Means.
""", language="sql")

    st.markdown("""
    **Justification statistique :**

    Chaque client est décrit par quatre indicateurs :
    dépenses totales, nombre de commandes, panier moyen
    et délai moyen de livraison.

    log1p réduit l'asymétrie des variables positives.
    StandardScaler met les variables sur une échelle comparable.
    K-Means regroupe les clients selon leurs comportements.

    Les médianes par cluster permettent ensuite d'interpréter
    les profils sans être trop influencées par les valeurs extrêmes.
    """)


# ============================================================
# 4. COMPARAISON DES PROFILS CLIENTS
# ============================================================

st.header("4. Comparaison des comportements clients")

segment_medians = (
    filtered_customers
    .groupby("segment", as_index=False)
    .agg(
        depense_mediane=("total_spent", "median"),
        commandes_mediane=("number_orders", "median"),
        panier_median=("avg_order_value", "median"),
        retard_median=("avg_delivery_delay", "median")
    )
)

fig_spending = px.bar(
    segment_medians,
    x="segment",
    y="depense_mediane",
    color="segment",
    text_auto=".2f",
    title="Dépense totale médiane par segment",
    labels={
        "segment": "Segment client",
        "depense_mediane": "Dépense médiane (€)"
    },
    template="plotly_dark"
)

fig_spending.update_layout(
    paper_bgcolor="#0b1120",
    plot_bgcolor="#111c33",
    showlegend=False
)

st.plotly_chart(fig_spending, use_container_width=True)

with st.expander("Requête SQL & Justification statistique"):
    st.code("""
SELECT
    segment,
    PERCENTILE_APPROX(total_spent, 0.5) AS depense_mediane,
    PERCENTILE_APPROX(number_orders, 0.5) AS commandes_mediane,
    PERCENTILE_APPROX(avg_order_value, 0.5) AS panier_median,
    PERCENTILE_APPROX(avg_delivery_delay, 0.5) AS retard_median
FROM delta.`/lakehouse/gold/olist_customer_segments`
GROUP BY segment
ORDER BY depense_mediane DESC
""", language="sql")

    st.markdown("""
    **Justification statistique :**

    La médiane décrit le comportement central de chaque segment.
    Elle est moins sensible aux clients ayant des dépenses
    exceptionnellement élevées que la moyenne.

    La comparaison des médianes permet de caractériser les
    profils d'achat et leurs délais moyens de livraison.
    """)


# ============================================================
# TABLEAU DE SYNTHESE
# ============================================================

st.header("Synthèse des segments clients")

st.dataframe(
    segment_medians.sort_values(
        "depense_mediane", ascending=False
    ).round(2),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Source : tables Delta de la couche Gold du Data Lakehouse Olist. "
    "Aucun fichier CSV brut n'est lu par cette application."
)
