import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Walmart Sales Dashboard", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
.main { background-color: #0E1117; }
h1, h2, h3, h4 { color: #FAFAFA; }
div[data-testid="metric-container"] {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #333;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HELPER ----------------
def format_currency(x):
    return f"${x:,.0f}"

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Walmart_Sales.csv")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔎 Filters")

store = st.sidebar.multiselect(
    "Select Store(s)",
    options=sorted(df["Store"].unique()),
    default=[df["Store"].unique()[0]]
)

date_range = st.sidebar.date_input(
    "Select Date Range",
    [df["Date"].min(), df["Date"].max()]
)

# ---------------- FILTER ----------------
filtered_df = df[
    (df["Store"].isin(store)) &
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1]))
].copy()

filtered_df["Date"] = filtered_df["Date"].dt.date

# ---------------- TITLE ----------------
st.title("📊 Walmart Sales Analytics Dashboard")

# ---------------- EXECUTIVE SUMMARY ----------------
st.markdown("### 📌 Executive Summary")
st.write(
    "This dashboard analyzes Walmart sales performance, highlighting top stores, "
    "seasonal trends, and forecasting future sales for better decision-making."
)

st.markdown("---")

# ---------------- KPI ----------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("💰 Total Sales", format_currency(filtered_df["Weekly_Sales"].sum()))
col2.metric("📊 Avg Sales", format_currency(filtered_df["Weekly_Sales"].mean()))
col3.metric("📈 Max Sales", format_currency(filtered_df["Weekly_Sales"].max()))
col4.metric("📉 Min Sales", format_currency(filtered_df["Weekly_Sales"].min()))

# ---------------- MAIN LAYOUT ----------------
col1, col2 = st.columns(2)

# ---------------- CHART ----------------
with col1:
    st.subheader("📈 Sales Trend")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(filtered_df["Date"], filtered_df["Weekly_Sales"], linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)

    st.pyplot(fig)

# ---------------- INSIGHTS ----------------
with col2:
    st.subheader("📌 Business Insights")

    store_sales = df.groupby("Store")["Weekly_Sales"].sum()
    top_store = store_sales.idxmax()
    top_value = store_sales.max()

    holiday_avg = filtered_df[filtered_df["Holiday_Flag"] == 1]["Weekly_Sales"].mean()
    non_holiday_avg = filtered_df[filtered_df["Holiday_Flag"] == 0]["Weekly_Sales"].mean()

    best_day = filtered_df.loc[filtered_df["Weekly_Sales"].idxmax()]

    st.success(f"🏆 Top Store: Store {top_store} → {format_currency(top_value)}")

    st.info(
        f"📅 Holiday sales ({format_currency(holiday_avg)}) are "
        f"{'higher' if holiday_avg > non_holiday_avg else 'lower'} than "
        f"non-holiday sales ({format_currency(non_holiday_avg)})."
    )

    st.warning(
        f"📈 Peak Day: {best_day['Date']} → "
        f"{format_currency(best_day['Weekly_Sales'])}"
    )

# ---------------- TOP STORES ----------------
st.subheader("🏆 Top 5 Stores")
top5 = df.groupby("Store")["Weekly_Sales"].sum().nlargest(5)
st.bar_chart(top5)

# ---------------- FORECAST ----------------
st.subheader("📈 Sales Forecast (Next 4 Weeks)")

forecast_df = filtered_df.copy()
forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
forecast_df["Days"] = (forecast_df["Date"] - forecast_df["Date"].min()).dt.days

X = forecast_df[["Days"]]
y = forecast_df["Weekly_Sales"]

model = LinearRegression()
model.fit(X, y)

last_day = forecast_df["Days"].max()
future_days = np.arange(last_day, last_day + 28).reshape(-1, 1)

predictions = model.predict(future_days)

forecast_output = pd.DataFrame({
    "Future Day": range(1, 29),
    "Predicted Sales": predictions
})

st.line_chart(forecast_output.set_index("Future Day"))

# ---------------- DOWNLOAD ----------------
st.subheader("📥 Download Data")

csv = filtered_df.to_csv(index=False)
st.download_button("Download Filtered Data", csv, "sales.csv")

# ---------------- RAW DATA ----------------
with st.expander("📂 View Raw Data"):
    st.dataframe(filtered_df, use_container_width=True)