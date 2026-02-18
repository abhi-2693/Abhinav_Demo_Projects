# Project 13: Forecasting — Stock Price Time Series (AAPL)

## Situation / Objective
Financial time series are noisy and non-stationary, but forecasting workflows still help estimate future ranges and compare modeling approaches. The objective is to forecast Apple (AAPL) stock prices using historical market data.

## Task
- Acquire time series data for AAPL.
- Perform time-series diagnostics (trend/seasonality/stationarity).
- Fit forecasting models and evaluate them.
- Produce a forecast for a defined horizon.

## Actions
- Pulled market data via `yfinance` for the selected time window.
- Performed exploratory diagnostics and stationarity checks (e.g., decomposition / ADF test).
- Trained multiple forecasting approaches (ARIMA-style methods, Prophet, and regression-style baselines) and compared error metrics.
- Produced a 365-day forecast and summarized model behavior.

## Results / Summary
- Delivered an end-to-end forecasting notebook from data acquisition to model comparison and forecasting.
- Demonstrated how different model classes behave on financial time series.

## Repository contents
- `Stock_forecasting_aapl.ipynb`
