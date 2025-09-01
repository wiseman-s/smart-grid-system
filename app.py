import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import altair as alt

# Page config
st.set_page_config(page_title="Smart Grid Dashboard", layout="wide")

# Header
st.markdown('<h1 style="color:#33C1FF;">⚡️ Smart Grid Dashboard</h1>', unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using sample simulated data")
    df = pd.read_csv("sample_simulated_14days.csv")

# Convert datetime
df['datetime'] = pd.to_datetime(df['datetime'])

# --- Dynamic Insights Grid ---
st.header("Insights Summary")
col1, col2, col3, col4 = st.columns(4)

total_supply = df[["geothermal_MW","hydro_MW","solar_MW","wind_MW"]].sum()
perc = (total_supply / total_supply.sum())*100

solar_peak_hour = int(df.groupby(df['datetime'].dt.hour)['solar_MW'].mean().idxmax())
wind_peak_hour = int(df.groupby(df['datetime'].dt.hour)['wind_MW'].mean().idxmax())
demand_peak_hours = df.groupby(df['datetime'].dt.hour)['demand_MW'].mean().sort_values(ascending=False).index[:2].tolist()

col1.markdown(f"<div class='card'><h3>🌍 Geothermal</h3><div class='small'>Stable baseload<br><b>{perc['geothermal_MW']:.1f}% of sampled generation</b></div></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='card'><h3>☀️ Solar</h3><div class='small'>Peaks around <b>{solar_peak_hour}:00</b><br><b>{perc['solar_MW']:.1f}% avg contribution</b></div></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='card'><h3>🌬 Wind</h3><div class='small'>Night-biased, peak around <b>{wind_peak_hour}:00</b><br><b>{perc['wind_MW']:.1f}% avg contribution</b></div></div>", unsafe_allow_html=True)
col4.markdown(f"<div class='card'><h3>🌊 Hydro</h3><div class='small'>Variable with rain<br><b>{perc['hydro_MW']:.1f}% avg contribution</b></div></div>", unsafe_allow_html=True)

# --- Sparks / Mini Charts ---
def sparkline(series, color):
    base = alt.Chart(pd.DataFrame({'index': range(len(series)), 'value': series})).mark_line(color=color).encode(
        x='index', y='value'
    )
    return base

st.header("Generation Sparklines")
col1, col2, col3, col4 = st.columns(4)
col1.altair_chart(sparkline(df['geothermal_MW'], "#FF8C00"), use_container_width=True)
col2.altair_chart(sparkline(df['solar_MW'], "#FFD700"), use_container_width=True)
col3.altair_chart(sparkline(df['wind_MW'], "#1E90FF"), use_container_width=True)
col4.altair_chart(sparkline(df['hydro_MW'], "#32CD32"), use_container_width=True)

# --- Forecast & Modeling ---
st.markdown('<h2 style="color:#33C1FF;">🤖 Forecast & Modeling</h2>', unsafe_allow_html=True)
model_option = st.selectbox("Choose model", ["Gradient Boosting", "Random Forest", "Linear Regression"])

if st.button("Run Forecast"):
    features = ['geothermal_MW','hydro_MW','solar_MW','wind_MW']
    X = df[features]
    y = df['demand_MW']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_option == "Gradient Boosting":
        model = GradientBoostingRegressor()
    elif model_option == "Random Forest":
        model = RandomForestRegressor()
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.success(f"Forecast complete using {model_option}!")
    chart_df = pd.DataFrame({'Actual': y_test.reset_index(drop=True), 'Predicted': y_pred})
    st.line_chart(chart_df)

# Footer
st.markdown('<div style="color:#9aa7b2; font-size:12px;">Prototype by Simon Wanyoike | Contact: allinmer57@gmail.com<br>Harnessing Data to Power Kenya\'s Clean Energy Future</div>', unsafe_allow_html=True)
