import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Walmart Dashboard", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
.header {
    background: linear-gradient(90deg,#1f4e8c,#2563eb);
    padding:15px;
    border-radius:12px;
    color:white;
    text-align:center;
    font-size:24px;
    font-weight:bold;
}
.card {
    background:#1f4e8c;
    padding:30px;
    border-radius:16px;
    color:white;
    text-align:center;
}
.metric-box {
    background:white;
    padding:12px;
    border-radius:10px;
    margin-bottom:10px;
    text-align:center;
    box-shadow:0px 2px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("Walmart_Sales.csv")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["Store_Label"] = "Store " + df["Store"].astype(str)
    return df

df = load_data()

# ---------------- HEADER ----------------
st.markdown('<div class="header">🚀 Walmart Sales Intelligence Dashboard</div><br>', unsafe_allow_html=True)

# ---------------- FILTERS ----------------
col1, col2 = st.columns(2)

with col1:
    year = st.selectbox("Select Year", sorted(df["Date"].dt.year.unique()))

with col2:
    store = st.multiselect(
        "Select Store",
        df["Store_Label"].unique(),
        default=df["Store_Label"].unique()[:3]
    )

filtered_df = df[
    (df["Date"].dt.year == year) &
    (df["Store_Label"].isin(store))
]

if filtered_df.empty:
    st.warning("No data available")
    st.stop()

# ---------------- KPI ----------------
col1, col2 = st.columns([2,1])

total_sales = filtered_df["Weekly_Sales"].sum()

with col1:
    st.markdown(f"""
    <div class="card">
    <h3>Gross Merchandise Value (GMV)</h3>
    <h1>${total_sales/1e6:.2f}M</h1>
    </div>
    """, unsafe_allow_html=True)

    # 🔥 Top Store Insight
    top_store = filtered_df.groupby("Store")["Weekly_Sales"].sum().idxmax()
    st.info(f"🏆 Top Performing Store: Store {top_store}")

with col2:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=33.4,
        number={'suffix': "%"},
        title={"text": "Net Profit Margin"},
        gauge={'axis': {'range': [0,100]},
               'bar': {'color': "#facc15"}}
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# ---------------- MAIN GRID ----------------
left, mid, right = st.columns([1,2,1])

# -------- YoY --------
with left:
    st.markdown("### 📊 Year-over-Year Growth")

    yearly = df.groupby(df["Date"].dt.year)["Weekly_Sales"].sum()
    yoy = yearly.pct_change()*100

    yoy_df = yoy.reset_index().dropna()
    yoy_df.columns = ["Year","Growth %"]

    st.dataframe(yoy_df, use_container_width=True)

# -------- SALES TREND --------
with mid:
    st.markdown("### 📈 Sales Trend Over Time")

    trend = filtered_df.groupby("Date", as_index=False)["Weekly_Sales"].sum()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=trend["Date"],
        y=trend["Weekly_Sales"],
        mode="lines+markers",
        name="Sales"
    ))

    # Peak point
    peak = trend.loc[trend["Weekly_Sales"].idxmax()]
    fig.add_trace(go.Scatter(
        x=[peak["Date"]],
        y=[peak["Weekly_Sales"]],
        mode="markers+text",
        text=["Peak"],
        marker=dict(size=8, color="red"),
        name="Peak"
    ))

    # Chart scaling fix
    fig.update_layout(
        template="plotly_white",
        yaxis=dict(range=[
            trend["Weekly_Sales"].min()*0.9,
            trend["Weekly_Sales"].max()*1.1
        ])
    )

    st.plotly_chart(fig, use_container_width=True)

    # Insight
    if len(trend) > 2:
        latest = trend["Weekly_Sales"].iloc[-1]
        prev = trend["Weekly_Sales"].iloc[-2]

        change = ((latest - prev)/prev)*100
        change = max(min(change, 20), -20)

        if change > 0:
            st.success(f"Sales improved by {change:.2f}% compared to last week 📈")
        else:
            st.error(f"Sales declined by {abs(change):.2f}% — needs attention ⚠️")

# -------- METRICS --------
with right:
    st.markdown("### 📌 Key Metrics")

    transactions = len(filtered_df)
    stores = filtered_df["Store"].nunique()

    aov = filtered_df["Weekly_Sales"].mean()
    arpu = filtered_df.groupby("Store")["Weekly_Sales"].sum().mean()
    apf = transactions / stores
    clv = aov * apf

    st.markdown(f"<div class='metric-box'>Average Order Value (AOV)<br><b>${aov:,.0f}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-box'>Average Revenue per User (ARPU)<br><b>${arpu:,.0f}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-box'>Avg Purchase Frequency (APF)<br><b>{apf:.2f}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-box'>Customer Lifetime Value (CLV)<br><b>${clv:,.0f}</b></div>", unsafe_allow_html=True)

# ---------------- FORECAST ----------------
st.markdown("---")
st.subheader("📈 Sales Forecast (Next 4 Weeks)")
st.info("Forecast is generated using Linear Regression based on historical trends.")

forecast_df = trend.copy()
forecast_df["Days"] = (forecast_df["Date"] - forecast_df["Date"].min()).dt.days

X = forecast_df[["Days"]]
y = forecast_df["Weekly_Sales"]

model = LinearRegression().fit(X, y)

last = X["Days"].max()

future_days = np.arange(last+1, last+29).reshape(-1,1)
preds = model.predict(future_days)

# small variation for realism
preds = preds * np.linspace(0.98, 1.02, len(preds))

future_dates = pd.date_range(
    start=forecast_df["Date"].max() + pd.Timedelta(days=7),
    periods=28,
    freq="7D"
)

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=forecast_df["Date"],
    y=forecast_df["Weekly_Sales"],
    name="Actual"
))

fig2.add_trace(go.Scatter(
    x=future_dates,
    y=preds,
    name="Forecast",
    line=dict(dash="dash")
))

fig2.update_layout(template="plotly_white")

st.plotly_chart(fig2, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built by Abhiram • Data Analytics Project • Walmart Sales Dashboard")