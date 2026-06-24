import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from tensorflow import keras
from streamlit_option_menu import option_menu
import time

st.set_page_config(page_title="Customer Behaviour & Demand Prediction System",
                    page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# ==================================================================
# THEME: "Command Center" — deep charcoal/navy + electric cyan + amber
# ==================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg-deep: #0A0E17;
    --bg-panel: #11172A;
    --bg-panel-light: #161D33;
    --accent-cyan: #00E5C7;
    --accent-amber: #FFB02E;
    --accent-rose: #FF5C7A;
    --text-primary: #F1F4F9;
    --text-muted: #8B95AB;
    --border-soft: rgba(255,255,255,0.07);
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(circle at 10% 0%, #131A2E 0%, #0A0E17 55%, #07090F 100%);
    color: var(--text-primary);
}

h1, h2, h3 { font-family: 'Sora', sans-serif !important; letter-spacing: -0.02em; }

h1 { font-weight: 800 !important; }
h2, h3 { font-weight: 700 !important; }

/* Fade-in animation on every page render */
.main .block-container {
    animation: fadeSlide 0.55s ease-out;
}
@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(14px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1220 0%, #0A0E17 100%);
    border-right: 1px solid var(--border-soft);
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(120deg, #0E2230 0%, #11172A 45%, #1A1330 100%);
    border: 1px solid var(--border-soft);
    border-radius: 18px;
    padding: 38px 42px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "";
    position: absolute; top: -60%; right: -10%;
    width: 420px; height: 420px;
    background: radial-gradient(circle, rgba(0,229,199,0.18) 0%, transparent 70%);
    animation: pulseGlow 6s ease-in-out infinite;
}
@keyframes pulseGlow {
    0%, 100% { opacity: 0.6; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.15); }
}
.hero-title {
    font-family: 'Sora', sans-serif; font-size: 2.3rem; font-weight: 800;
    background: linear-gradient(90deg, #FFFFFF 10%, #00E5C7 90%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; position: relative; z-index: 1;
}
.hero-sub {
    color: var(--text-muted); font-size: 1.02rem; margin-top: 10px; position: relative; z-index: 1;
    max-width: 720px;
}
.hero-pill {
    display: inline-block; background: rgba(0,229,199,0.12); color: var(--accent-cyan);
    border: 1px solid rgba(0,229,199,0.35); border-radius: 999px; padding: 4px 14px;
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
    margin-bottom: 14px; position: relative; z-index: 1;
}

/* KPI Cards */
.kpi-card {
    background: linear-gradient(155deg, var(--bg-panel) 0%, var(--bg-panel-light) 100%);
    border: 1px solid var(--border-soft);
    border-radius: 16px;
    padding: 20px 22px;
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    height: 100%;
}
.kpi-card:hover {
    transform: translateY(-4px);
    border-color: rgba(0,229,199,0.4);
    box-shadow: 0 12px 28px rgba(0,229,199,0.08);
}
.kpi-label {
    color: var(--text-muted); font-size: 0.78rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;
}
.kpi-value {
    font-family: 'Sora', sans-serif; font-size: 1.85rem; font-weight: 800; color: var(--text-primary);
}
.kpi-accent { color: var(--accent-cyan); }
.kpi-amber { color: var(--accent-amber); }
.kpi-rose { color: var(--accent-rose); }

/* Section card wrapper */
.panel {
    background: var(--bg-panel);
    border: 1px solid var(--border-soft);
    border-radius: 16px;
    padding: 24px 26px;
    margin-bottom: 20px;
}

/* Pipeline step chips */
.flow-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin: 18px 0; }
.flow-chip {
    background: var(--bg-panel-light); border: 1px solid var(--border-soft);
    border-radius: 10px; padding: 10px 16px; font-size: 0.85rem; font-weight: 600;
    transition: all 0.2s ease;
}
.flow-chip:hover { border-color: var(--accent-cyan); color: var(--accent-cyan); }
.flow-arrow { color: var(--accent-cyan); font-size: 1.1rem; }

/* Recommendation result boxes */
.result-box {
    border-radius: 14px; padding: 22px 24px; margin-top: 10px;
    animation: popIn 0.4s ease-out;
    border-left: 4px solid;
}
@keyframes popIn {
    from { opacity: 0; transform: scale(0.97); }
    to { opacity: 1; transform: scale(1); }
}
.result-critical { background: rgba(255,92,122,0.08); border-color: var(--accent-rose); }
.result-warning { background: rgba(255,176,46,0.08); border-color: var(--accent-amber); }
.result-info { background: rgba(0,229,199,0.08); border-color: var(--accent-cyan); }
.result-success { background: rgba(0,229,199,0.08); border-color: #2ED573; }

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #00E5C7 0%, #00B8D9 100%) !important;
    color: #051018 !important; font-weight: 700 !important; border: none !important;
    border-radius: 10px !important; padding: 0.65em 1.4em !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease !important;
    box-shadow: 0 4px 14px rgba(0,229,199,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 8px 22px rgba(0,229,199,0.4) !important;
}

/* Metrics native widget restyle */
div[data-testid="stMetric"] {
    background: var(--bg-panel); border: 1px solid var(--border-soft);
    border-radius: 14px; padding: 14px 18px;
    transition: transform 0.2s ease;
}
div[data-testid="stMetric"]:hover { transform: translateY(-3px); border-color: rgba(0,229,199,0.35); }
div[data-testid="stMetricLabel"] { color: var(--text-muted) !important; }

/* Tabs */
button[data-baseweb="tab"] { font-weight: 600 !important; }

/* Dataframe */
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Expander */
details {
    background: var(--bg-panel); border: 1px solid var(--border-soft) !important;
    border-radius: 12px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: #243047; border-radius: 6px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }
</style>
""", unsafe_allow_html=True)


def kpi_card(label, value, accent_class=""):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {accent_class}">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------
# LOAD SAVED MODELS / DATA
# ------------------------------------------------------------------
@st.cache_resource
def load_all():
    objs = {}
    objs['churn_model'] = joblib.load('best_churn_model.pkl')
    objs['scaler_churn'] = joblib.load('scaler_churn.pkl')
    objs['clv_model'] = joblib.load('best_clv_model.pkl')
    objs['scaler_clv'] = joblib.load('scaler_clv.pkl')
    objs['demand_model'] = joblib.load('best_demand_model.pkl')
    objs['scaler_demand'] = joblib.load('scaler_demand.pkl')
    objs['le_stock'] = joblib.load('le_stock.pkl')
    objs['kmeans'] = joblib.load('kmeans_model.pkl')
    objs['scaler_seg'] = joblib.load('scaler_seg.pkl')
    objs['segment_names'] = joblib.load('segment_names.pkl')
    objs['cnn_model'] = keras.models.load_model('cnn_model.keras')
    objs['churn_results'] = pd.read_csv('churn_results.csv')
    objs['clv_results'] = pd.read_csv('clv_results.csv')
    objs['demand_results'] = pd.read_csv('demand_results.csv')
    return objs

try:
    data = load_all()
    LOAD_OK = True
except Exception as e:
    LOAD_OK = False
    LOAD_ERR = str(e)

PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#F1F4F9"),
    colorway=["#00E5C7", "#FFB02E", "#FF5C7A", "#5B8DEF", "#9B6BFF"]
)

def style_fig(fig):
    fig.update_layout(**PLOTLY_DARK, margin=dict(t=50, l=10, r=10, b=10))
    return fig

# ------------------------------------------------------------------
# SIDEBAR NAVIGATION & DYNAMIC BROWSE DATA LOADER
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 18px 0;">
        <div style="font-size:1.6rem;">📊</div>
        <div style="font-family:'Sora',sans-serif; font-weight:800; font-size:1.05rem; color:#F1F4F9;">
            BEHAVIOUR ENGINE
        </div>
        <div style="font-size:0.72rem; color:#8B95AB; letter-spacing:0.05em;">PREDICTIVE RETAIL SUITE</div>
    </div>
    """, unsafe_allow_html=True)

    page = option_menu(
        menu_title=None,
        options=["Home", "Data Exploration", "Customer Segments", "Demand Forecasting",
                 "Model Comparison", "Smart Predictor", "About"],
        icons=["house", "bar-chart-line", "people", "boxes",
               "trophy", "lightning-charge", "info-circle"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"color": "#00E5C7", "font-size": "16px"},
            "nav-link": {
                "font-size": "14px", "font-weight": "600", "text-align": "left",
                "margin": "4px 0", "border-radius": "10px", "color": "#C7CDDB",
                "padding": "10px 14px",
            },
            "nav-link-selected": {
                "background-color": "rgba(0,229,199,0.12)",
                "color": "#00E5C7",
                "border": "1px solid rgba(0,229,199,0.35)",
            },
        }
    )
    
    st.markdown("---")
    st.markdown("### 📂 Browse Data")
    
    # Dynamic file uploader that can check calculations across all tabs
    uploaded_data_file = st.file_uploader("Upload tracking data (CSV)", type=["csv"], key="dynamic_data_loader")

    if uploaded_data_file is not None:
        raw_uploaded_df = pd.read_csv(uploaded_data_file)
        # Inspect columns to dynamically route data to the proper calculations
        if 'Recency' in raw_uploaded_df.columns or 'Frequency' in raw_uploaded_df.columns or 'Monetary' in raw_uploaded_df.columns:
            data['rfm'] = raw_uploaded_df
            data['daily_demand'] = pd.read_csv('daily_demand_final.csv')
        else:
            data['daily_demand'] = raw_uploaded_df
            data['rfm'] = pd.read_csv('rfm_final.csv')
    else:
        # Default local fallback baseline data configurations
        data['rfm'] = pd.read_csv('rfm_final.csv')
        data['daily_demand'] = pd.read_csv('daily_demand_final.csv')

    st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(0,229,199,0.06); border:1px solid rgba(0,229,199,0.2);
                border-radius:12px; padding:12px 14px; font-size:0.78rem; color:#8B95AB;">
        Built on the <b style="color:#F1F4F9;">Online Retail II</b> dataset (UCI) —
        real UK e-commerce transactions, 2009–2011.
    </div>
    """, unsafe_allow_html=True)

if not LOAD_OK:
    st.error(f"⚠️ Could not load model files. Run the notebook fully first so all .pkl/.keras "
             f"files are created in this same folder.\n\nDetails: {LOAD_ERR}")
    st.stop()

# ==================================================================
# PAGE: HOME
# ==================================================================
if page == "Home":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-pill">⚡ Live Decision-Support System</div>
        <div class="hero-title">Customer Behaviour & Demand Prediction</div>
        <div class="hero-sub">
            One pipeline that turns raw transactions into action: who your valuable
            customers are, who's about to leave, what they're worth, and how much
            stock you'll need — all from a single input.
        </div>
    </div>
    """, unsafe_allow_html=True)

    rfm = data['rfm']
    dd = data['daily_demand']

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Total Customers", f"{rfm.shape[0]:,}", "kpi-accent")
    with c2: kpi_card("Total Revenue", f"£{rfm['Monetary'].sum():,.0f}", "kpi-amber")
    with c3: kpi_card("Churn Rate", f"{rfm['Churn'].mean()*100:.1f}%", "kpi-rose")
    with c4: kpi_card("Products Tracked", f"{dd['StockCode'].nunique()}", "kpi-accent")

    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("⚙️ The Pipeline")
    st.markdown("""
    <div class="flow-row">
        <div class="flow-chip">📥 Raw Transactions</div>
        <div class="flow-arrow">→</div>
        <div class="flow-chip">🧹 Cleaning</div>
        <div class="flow-arrow">→</div>
        <div class="flow-chip">🧮 RFM Features</div>
        <div class="flow-arrow">→</div>
        <div class="flow-chip">👥 Segmentation</div>
        <div class="flow-arrow">→</div>
        <div class="flow-chip">⚠️ Churn Risk</div>
        <div class="flow-arrow">→</div>
        <div class="flow-chip">💰 CLV</div>
        <div class="flow-arrow">→</div>
        <div class="flow-chip">📦 Demand Forecast</div>
        <div class="flow-arrow">→</div>
        <div class="flow-chip" style="border-color:#00E5C7; color:#00E5C7;">🎯 Recommendation</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("🧩 The Problem")
        st.write("""
        Retailers usually answer *who's valuable*, *who's leaving*, and
        *what to stock* with three disconnected tools. This system answers
        all three from one input, then tells you what to actually do about it —
        not just a number on a dashboard.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("🧠 Algorithms Applied")
        st.write("""
        **Segmentation:** KMeans · **Classification:** Logistic Regression,
        Decision Tree, Random Forest, KNN, SVM, ANN · **Regression:** Linear
        Regression, Decision Tree, Random Forest, ANN · **Deep Learning:** CNN
        on demand-pattern images.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# ==================================================================
# PAGE: DATA EXPLORATION
# ==================================================================
elif page == "Data Exploration":
    st.markdown('<div class="hero-pill">📈 Exploration</div>', unsafe_allow_html=True)
    st.title("Data Exploration")
    rfm = data['rfm']
    dd = data['daily_demand'].copy()
    dd['Date'] = pd.to_datetime(dd['Date'])

    tab1, tab2, tab3 = st.tabs(["💳 Customer Distributions", "📦 Demand Trends", "📋 Raw RFM Table"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(rfm, x="Monetary", nbins=50, title="Customer Spend Distribution",
                               range_x=[0, rfm['Monetary'].quantile(0.95)])
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with col2:
            fig2 = px.histogram(rfm, x="Frequency", nbins=40, title="Purchase Frequency Distribution",
                                 range_x=[0, rfm['Frequency'].quantile(0.95)])
            st.plotly_chart(style_fig(fig2), use_container_width=True)

        fig3 = px.scatter(rfm, x="Recency", y="Monetary", color="Churn",
                           title="Recency vs Monetary Value (colored by Churn)",
                           range_y=[0, rfm['Monetary'].quantile(0.95)],
                           color_continuous_scale=["#00E5C7", "#FF5C7A"])
        st.plotly_chart(style_fig(fig3), use_container_width=True)

    with tab2:
        top10_products = dd.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False).head(10)
        fig4 = px.bar(x=top10_products.index, y=top10_products.values,
                       title="Top 10 Products by Total Quantity Sold",
                       labels={'x': 'Stock Code', 'y': 'Total Quantity'})
        st.plotly_chart(style_fig(fig4), use_container_width=True)

        daily_total = dd.groupby('Date')['Quantity'].sum().reset_index()
        fig5 = px.line(daily_total, x='Date', y='Quantity', title="Total Daily Demand Trend (Top 50 Products)")
        st.plotly_chart(style_fig(fig5), use_container_width=True)

    with tab3:
        st.dataframe(rfm.head(50), use_container_width=True)

# ==================================================================
# PAGE: CUSTOMER SEGMENTS
# ==================================================================
elif page == "Customer Segments":
    st.markdown('<div class="hero-pill">👥 Segmentation</div>', unsafe_allow_html=True)
    st.title("Customer Segments")
    rfm = data['rfm']

    seg_profile = rfm.groupby('SegmentName')[['Recency', 'Frequency', 'Monetary']].mean().round(1)
    seg_profile['Customer Count'] = rfm['SegmentName'].value_counts()

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Segment Profiles")
    st.dataframe(seg_profile, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(rfm, names='SegmentName', title="Customer Distribution by Segment", hole=0.45)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with col2:
        fig2 = px.scatter(rfm, x="Frequency", y="Monetary", color="SegmentName",
                           title="Frequency vs Monetary by Segment",
                           range_y=[0, rfm['Monetary'].quantile(0.97)],
                           range_x=[0, rfm['Frequency'].quantile(0.97)])
        st.plotly_chart(style_fig(fig2), use_container_width=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("🤔 Why 4 Segments (not the mathematically 'best' 2)?")
    st.write("""
    KMeans with k=2 gives the highest silhouette score, but only splits customers
    into "good vs bad" — not useful for targeted action. We chose **k=4** for
    business-actionable groups: Champions, Loyal Customers, Regular Customers,
    and At Risk/Lost — trading a small amount of mathematical cluster purity for
    much greater business usefulness.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    cols = st.columns(len(rfm['SegmentName'].unique()))
    for i, seg in enumerate(sorted(rfm['SegmentName'].unique())):
        sub = rfm[rfm['SegmentName'] == seg]
        with cols[i % len(cols)]:
            kpi_card(seg, f"{len(sub)} customers", "kpi-accent")
            st.caption(f"Avg spend £{sub['Monetary'].mean():,.0f} · {sub['Frequency'].mean():.1f} orders avg")

# ==================================================================
# PAGE: DEMAND FORECASTING
# ==================================================================
elif page == "Demand Forecasting":
    st.markdown('<div class="hero-pill">📦 Forecasting</div>', unsafe_allow_html=True)
    st.title("Demand Forecasting")
    dd = data['daily_demand'].copy()
    dd['Date'] = pd.to_datetime(dd['Date'])

    product_list = sorted(dd['StockCode'].unique())
    selected_product = st.selectbox("Select a product (Stock Code) to view its demand trend:", product_list)

    prod_data = dd[dd['StockCode'] == selected_product].sort_values('Date')
    fig = px.line(prod_data, x='Date', y='Quantity', title=f"Demand Trend — Product {selected_product}",
                   markers=True)
    st.plotly_chart(style_fig(fig), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Avg Daily Demand", f"{prod_data['Quantity'].mean():.1f}", "kpi-accent")
    with c2: kpi_card("Max Daily Demand", f"{prod_data['Quantity'].max():.0f}", "kpi-amber")
    with c3: kpi_card("Avg Price", f"£{prod_data['Price'].mean():.2f}", "kpi-accent")

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Why Demand Forecasting Is Harder Than Churn/CLV")
    st.write("""
    Daily product-level demand is highly volatile — a single large order can spike
    a day's numbers. Our best model uses lag features (yesterday's demand, 7-day
    rolling average) since recent sales velocity is the strongest predictor of
    near-future demand — a pattern seen in real retail forecasting systems.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================================
# PAGE: MODEL COMPARISON
# ==================================================================
elif page == "Model Comparison":
    st.markdown('<div class="hero-pill">🏆 Benchmarks</div>', unsafe_allow_html=True)
    st.title("Model Comparison — Which Algorithm Wins, and Why")

    tab1, tab2, tab3, tab4 = st.tabs(["⚠️ Churn", "💰 CLV", "📦 Demand", "🧠 CNN (Bonus)"])

    with tab1:
        cr = data['churn_results'].sort_values('F1-Score', ascending=False)
        st.dataframe(cr, use_container_width=True)
        fig = px.bar(cr, x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                     barmode='group', title="Churn Model Comparison")
        st.plotly_chart(style_fig(fig), use_container_width=True)
        best = cr.iloc[0]['Model']
        st.markdown(f"""
        <div class="result-box result-success">
        <b>🏆 Best Model: {best}</b><br><br>
        Churn behavior isn't linearly separable — a customer can have decent frequency
        but still be at risk if their average order value is dropping. {best} captures
        these non-linear interactions better than Logistic Regression, while staying
        more stable than KNN on skewed monetary data.<br><br>
        <i>Note: Recency was intentionally excluded from features since it directly
        defines the churn label — including it would cause label leakage and
        artificially inflate accuracy to ~100%.</i>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        clvr = data['clv_results'].sort_values('R2 Score', ascending=False)
        st.dataframe(clvr, use_container_width=True)
        fig = px.bar(clvr, x='Model', y='R2 Score', title="CLV Model Comparison (R² Score)")
        st.plotly_chart(style_fig(fig), use_container_width=True)
        best_clv = clvr.iloc[0]['Model']
        st.markdown(f"""
        <div class="result-box result-success">
        <b>🏆 Best Model: {best_clv}</b><br><br>
        It best captures the non-linear relationship between purchase frequency,
        order value patterns, and total customer spend.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        dr = data['demand_results'].sort_values('R2 Score', ascending=False)
        st.dataframe(dr, use_container_width=True)
        fig = px.bar(dr, x='Model', y='R2 Score', title="Demand Model Comparison (R² Score)")
        st.plotly_chart(style_fig(fig), use_container_width=True)
        best_d = dr.iloc[0]['Model']
        st.markdown(f"""
        <div class="result-box result-warning">
        <b>🏆 Best Model: {best_d}</b><br><br>
        Daily demand depends on non-linear interactions between recent sales history
        (lag features) and seasonality. {best_d} models these directly through
        decision splits, while Linear Regression assumes a straight-line relationship
        that underfits real demand patterns.<br><br>
        <i>Note: R² is moderate (not near 1.0) — this is expected and honest, since
        daily product-level demand is inherently noisy in real retail data.</i>
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.write("""
        Each product's 49-day demand history was reshaped into a 7×7 "image" and
        fed into a Convolutional Neural Network to classify whether that period
        represents a HIGH or LOW relative demand pattern. This demonstrates CNNs
        applied beyond traditional images — to structured time-series-as-image data,
        a real technique used in production forecasting research.
        """)
        kpi_card("CNN Test Accuracy", "~96–99%", "kpi-accent")

# ==================================================================
# PAGE: SMART PREDICTOR
# ==================================================================
elif page == "Smart Predictor":
    st.markdown('<div class="hero-pill">⚡ Live Engine</div>', unsafe_allow_html=True)
    st.title("Smart Recommendation Engine")
    st.write("Enter any customer/product scenario — get segment, churn risk, "
             "expected spend, expected demand, and a recommended action, instantly.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("👤 Customer Behaviour")
        recency = st.slider("Days since last purchase (Recency)", 0, 800, 30)
        frequency = st.slider("Number of past orders (Frequency)", 1, 500, 10)
        avg_order_value = st.number_input("Average order value (£)", min_value=1.0, max_value=5000.0, value=50.0)
        total_quantity = st.number_input("Total quantity purchased (lifetime)", min_value=1, max_value=10000, value=100)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("📦 Product / Timing")
        product_options = list(data['le_stock'].classes_)
        selected_stock = st.selectbox("Product (Stock Code)", product_options)
        price = st.number_input("Product price (£)", min_value=0.1, max_value=500.0, value=5.0)
        month = st.select_slider("Target month", options=list(range(1, 13)), value=6)
        day_of_week = st.select_slider("Day of week", options=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], value="Wed")
        day_of_week_num = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].index(day_of_week)
        st.markdown('</div>', unsafe_allow_html=True)

    predict_clicked = st.button("🔍  Predict & Recommend", type="primary", use_container_width=True)

    if predict_clicked:
        with st.spinner("Running models..."):
            time.sleep(0.4)
            monetary = avg_order_value * frequency

            seg_input = data['scaler_seg'].transform([[recency, frequency, monetary]])
            seg_pred = data['kmeans'].predict(seg_input)[0]
            seg_name = data['segment_names'].get(seg_pred, f"Segment {seg_pred}")

            churn_input = data['scaler_churn'].transform([[frequency, monetary, avg_order_value, total_quantity]])
            churn_proba = data['churn_model'].predict_proba(churn_input)[0][1]

            clv_input = data['scaler_clv'].transform([[recency, frequency, avg_order_value, total_quantity]])
            clv_pred = data['clv_model'].predict(clv_input)[0]

            stock_enc = data['le_stock'].transform([selected_stock])[0]
            dd = data['daily_demand']
            prod_hist = dd[dd['StockCode'] == selected_stock].sort_values('Date')
            lag1 = prod_hist['Quantity'].iloc[-1] if len(prod_hist) > 0 else 10
            rolling7 = prod_hist['Quantity'].tail(7).mean() if len(prod_hist) > 0 else 10
            week_of_year = min(int(month * 4.33), 52)

            demand_input = data['scaler_demand'].transform([[stock_enc, price, month, day_of_week_num,
                                                                week_of_year, lag1, rolling7]])
            demand_pred = max(0, data['demand_model'].predict(demand_input)[0])

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)
        with r1: kpi_card("Predicted Segment", seg_name, "kpi-accent")
        with r2: kpi_card("Churn Risk", f"{churn_proba*100:.1f}%", "kpi-rose" if churn_proba > 0.5 else "kpi-accent")
        with r3: kpi_card("Predicted CLV", f"£{clv_pred:,.0f}", "kpi-amber")
        with r4: kpi_card("Predicted Demand", f"{demand_pred:.0f} units", "kpi-accent")

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.subheader("💡 Recommended Action")

        prod_avg = prod_hist['Quantity'].mean() if len(prod_hist) > 0 else 0

        if churn_proba > 0.6 and monetary > data['rfm']['Monetary'].median():
            st.markdown("""
            <div class="result-box result-critical">
            <b>🎯 HIGH PRIORITY: Send retention discount.</b><br>
            This is a high-value customer at high risk of churning. A targeted discount
            or personal outreach could save significant revenue.
            </div>
            """, unsafe_allow_html=True)
        elif churn_proba > 0.6:
            st.markdown("""
            <div class="result-box result-warning">
            <b>✉️ Send re-engagement email.</b><br>
            Customer shows churn risk. A simple reminder or small incentive may bring them back.
            </div>
            """, unsafe_allow_html=True)
        elif prod_avg and demand_pred > prod_avg * 1.3:
            st.markdown("""
            <div class="result-box result-info">
            <b>📦 Increase stock.</b><br>
            Predicted demand for this product is above average for this period —
            consider increasing inventory to avoid stockouts.
            </div>
            """, unsafe_allow_html=True)
        elif frequency <= 2:
            st.markdown("""
            <div class="result-box result-info">
            <b>🌱 New/low-frequency customer.</b><br>
            Consider an onboarding offer to encourage a second purchase and build loyalty early.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-box result-success">
            <b>✅ Customer relationship looks healthy.</b><br>
            Continue standard engagement — no urgent action needed.
            </div>
            """, unsafe_allow_html=True)

# ==================================================================
# PAGE: ABOUT
# ==================================================================
elif page == "About":
    st.markdown('<div class="hero-pill">ℹ️ Project Info</div>', unsafe_allow_html=True)
    st.title("About This Project")

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.write("""
    ### Customer Behaviour & Demand Prediction System

    **Dataset:** Online Retail II (UCI Machine Learning Repository) — real
    transactional data from a UK-based online retailer (2009–2011).

    **Approach:** Combines customer segmentation (KMeans), churn prediction
    (6 classification algorithms compared), customer lifetime value prediction
    (4 regression algorithms compared), demand forecasting (4 regression
    algorithms with lag features), and a bonus CNN model for demand pattern
    classification.

    **Algorithms covered:** Linear Regression, Logistic Regression, Decision
    Tree, Random Forest, KNN, SVM, ANN (MLP), CNN, and KMeans Clustering.

    **Unique contribution:** A live Smart Recommendation Engine that takes any
    user-entered customer/product scenario — not just the existing dataset —
    and returns a segment, churn risk, predicted value, predicted demand, and
    a plain-English recommended business action in one combined output.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.write("**Team Members:** Komal Bisht and Karan Kumar")
    st.write("**Trainer:** Mr.Prateek Gupta and Mr. Divyank Chauhan ,Ducat India")
    st.markdown('</div>', unsafe_allow_html=True)
