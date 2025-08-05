import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt

st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")

# 📊 Streamlit Dashboard
st.title("📊 Stock Market Forecast Dashboard")
stock = st.text_input("Enter Stock Symbol (e.g., TCS.NS)", "TCS.NS")
start = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end = st.date_input("End Date", pd.to_datetime("2025-06-30"))

n_days = st.slider("Days to Forecast", 15, 90, 30)

if st.button("Generate Dashboard"):

    # Load data
    data = yf.download(stock, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Close'][stock]
    else:
        data = data['Close']
    data = data.to_frame().dropna()
    data.columns = ['Close']
    data.index = pd.to_datetime(data.index)

    st.line_chart(data['Close'])

    # Split into train/test
    train = data['Close'][:-n_days]
    test = data['Close'][-n_days:]

    # ARIMA
    model_arima = ARIMA(train, order=(2, 1, 1))
    fit_arima = model_arima.fit()
    forecast_arima = fit_arima.forecast(steps=n_days)
    rmse_arima = sqrt(mean_squared_error(test, forecast_arima))
    st.subheader(" ARIMA Forecast")
    st.line_chart(pd.DataFrame(forecast_arima, index=test.index, columns=['ARIMA']))

    # SARIMA
    model_sarima = SARIMAX(train, order=(1,1,1), seasonal_order=(0,1,1,12))
    fit_sarima = model_sarima.fit()
    forecast_sarima = fit_sarima.forecast(steps=n_days)
    rmse_sarima = sqrt(mean_squared_error(test, forecast_sarima))
    st.subheader("SARIMA Forecast")
    st.line_chart(pd.DataFrame(forecast_sarima, index=test.index, columns=['SARIMA']))

    # Prophet
    df_prophet = data.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    model_prophet = Prophet()
    model_prophet.fit(df_prophet[:-n_days])
    future = model_prophet.make_future_dataframe(periods=n_days)
    forecast_prophet = model_prophet.predict(future)
    prophet_pred = forecast_prophet[['ds', 'yhat']].set_index('ds')[-n_days:]
    rmse_prophet = sqrt(mean_squared_error(test, prophet_pred['yhat']))
    st.subheader("Prophet Forecast")
    st.line_chart(prophet_pred)

    # Dummy LSTM (or simulated RMSE)
    rmse_lstm = np.random.uniform(rmse_arima - 5, rmse_arima + 5)  # Replace with real model

    # Accuracy Summary
    st.subheader(" Model Accuracy (RMSE)")
    st.write(f'ARIMA RMSE: {rmse_arima:.2f}')
    st.write(f'SARIMA RMSE: {rmse_sarima:.2f}')
    st.write(f'Prophet RMSE: {rmse_prophet:.2f}')
    st.write(f'LSTM RMSE: {rmse_lstm:.2f}  *(simulated)*')

    st.success("✅ All forecasts generated successfully!")
