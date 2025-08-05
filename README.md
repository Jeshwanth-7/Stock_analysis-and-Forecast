# 📈 Stock Price Forecasting using Time Series & Deep Learning


# 📈 Tata Stock Forecasting using ARIMA, SARIMA, Prophet & LSTM

This project presents a comprehensive analysis and forecasting system for **TATAELXSI.NS** stock using both classical time series models and deep learning. We explore and compare four popular models — **ARIMA**, **SARIMA**, **Facebook Prophet**, and **LSTM** — to identify trends and forecast future stock prices.

---

## 🔍 Project Overview

Using historical data from 2015 to 2025, this project:
- Downloads and cleans stock price data from Yahoo Finance
- Applies four forecasting models on the **Close** price
- Compares model performance visually
- Builds reproducible and interpretable forecasts

---

## 📊 Models Implemented

### ✅ ARIMA
- Captures autoregressive and moving average components
- Tuned with `(order=(7,2,3))`
- Forecasts the next 30 days

### ✅ SARIMA
- Extends ARIMA with seasonality
- Configured as `order=(1,1,1)` and `seasonal_order=(1,1,1,12)`
- Produces 30-day ahead predictions with confidence intervals

### ✅ Facebook Prophet
- Handles trend shifts, holidays, and seasonality well
- Automatically decomposes trend and seasonality
- Forecasts 365 future days

### ✅ LSTM (Deep Learning)
- Uses scaled data and sequence modeling
- Captures long-term dependencies in stock price movements
- Achieved **very low training loss (~1.3e-4)**

