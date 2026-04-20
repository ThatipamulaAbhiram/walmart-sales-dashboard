import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Walmart Dashboard", layout="wide")

# ------------------ CUSTOM UI ------------------
st.markdown("""
<style>
.main {background-color: #f5f7fa;}
h1 {color: #1f4e79;}
.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.markdown("""
<h1 style='text-align: center;'>📊 Walmart Sales Intelligence Dashboard</h1>
<hr>
""", unsafe_allow_html=True)

# ------------------ LOAD DATA ------------------
df = pd.read_csv("Walmart_Sales.csv")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# ------------------ SIDEBAR ------------------
st.sidebar.header("🔎 Filters")

stores = st.sidebar.multiselect(
    "Select Store(s)",
    options=df["Store"].unique(),
    default=[df["Store"].unique()[0]]
)

date_range = st.sidebar.date_input(
    "Select Date Range",
    [df["Date"].min(), df["Date"].max()]
)

# ------------------ FILTER DATA ------------------
filtered_df = df[
    (df["Store"].isin(stores)) &
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1]))
]

# ------------------ KPIs ------------------
total_sales = filtered_df["Weekly_Sales"].sum()
avg_sales = filtered_df["Weekly_Sales"].mean()
max_sales = filtered_df["Weekly_Sales"].max()
min_sales = filtered_df["Weekly_Sales"].min()

col1, col2, col3, col4 = st.columns(4)

col1.metric("💰 Total Sales", f"${total_sales:,.0f}")
col2.metric("📊 Avg Sales", f"${avg_sales:,.0f}")
col3.metric("📈 Max Sales", f"${max_sales:,.0f}")
col4.metric("📉 Min Sales", f"${min_sales:,.0f}")

# ------------------ SALES TREND ------------------
st.subheader("📈 Sales Trend")

trend = filtered_df.groupby("Date")["Weekly_Sales"].sum()

fig, ax = plt.subplots()
ax.plot(trend.index, trend.values)
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
st.pyplot(fig)

# ------------------ 🚨 ALERT SYSTEM ------------------
filtered_df["Change"] = filtered_df["Weekly_Sales"].pct_change()

alerts = filtered_df[filtered_df["Change"] < -0.15]

st.subheader("🚨 Sales Drop Alerts")

if alerts.empty:
    st.success("No major sales drops detected")
else:
    st.dataframe(alerts[["Store", "Date", "Weekly_Sales"]])

# ------------------ BUSINESS INSIGHTS ------------------
st.subheader("📌 Business Insights")

store_sales = filtered_df.groupby("Store")["Weekly_Sales"].sum()
top_store = store_sales.idxmax()
top_value = store_sales.max()

holiday_avg = filtered_df[filtered_df["Holiday_Flag"] == 1]["Weekly_Sales"].mean()
non_holiday_avg = filtered_df[filtered_df["Holiday_Flag"] == 0]["Weekly_Sales"].mean()

best_day = filtered_df.loc[filtered_df["Weekly_Sales"].idxmax()]

st.success(f"🏆 Top Store: Store {top_store} → ${top_value:,.0f}")
st.info(f"📅 Holiday Sales (${holiday_avg:,.0f}) vs Non-Holiday (${non_holiday_avg:,.0f})")
st.warning(f"📌 Peak Day: {best_day['Date'].date()} → ${best_day['Weekly_Sales']:,.0f}")

# ------------------ 📊 RECOMMENDATIONS ------------------
st.subheader("📊 Recommendations")

if holiday_avg > non_holiday_avg:
    st.info("👉 Increase inventory during holidays.")
else:
    st.info("👉 Focus on regular weeks.")

if top_value > 200000000:
    st.info("👉 Invest more in high-performing stores.")

# ------------------ 🔮 FORECAST ------------------
st.subheader("🔮 Sales Forecast (Next 4 Weeks)")

trend_df = trend.reset_index()
trend_df["Days"] = (trend_df["Date"] - trend_df["Date"].min()).dt.days

X = trend_df[["Days"]]
y = trend_df["Weekly_Sales"]

model = LinearRegression()
model.fit(X, y)

last_day = X["Days"].max()

future_days = np.arange(last_day + 1, last_day + 29).reshape(-1, 1)
forecast = model.predict(future_days)

future_dates = pd.date_range(trend_df["Date"].max(), periods=28)

fig2, ax2 = plt.subplots()
ax2.plot(trend_df["Date"], y, label="Actual")
ax2.plot(future_dates, forecast, label="Forecast")
ax2.legend()
st.pyplot(fig2)