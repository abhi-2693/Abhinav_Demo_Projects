import os
import numpy as np
import pandas as pd
from src.logger import logger
import streamlit as st
from src.config import RESULTS_DIR, OOT_SUMMARY_CSV, SHORT_TERM_FORECAST_HORIZON, LONG_TERM_FORECAST_HORIZON
from src.model_utils import (
    auto_sarimax_fit_template,
    train_arima,
    train_prophet,
    train_xgboost,
    train_lgbm,
    train_voting_regressor,
    train_random_forest,
    train_catboost
)

def final_forecast(df_clean, short_horizon=int(SHORT_TERM_FORECAST_HORIZON), long_horizon=int(LONG_TERM_FORECAST_HORIZON)):
    metrics_df = pd.read_csv(OOT_SUMMARY_CSV)

    # Select best model using lowest MAPE_mean
    best_model = metrics_df.loc[metrics_df['MAPE_mean'].idxmin(), 'Model']
    logger.info(f"\nBest Model Selected: {best_model}")

    # Prepare full dataset
    y_full = df_clean["value"]
    X_full = df_clean.drop(columns=["value"])

    # Refit best model on full dataset (optional, in case retraining is needed)
    model_dispatcher = {
        'SARIMAX': auto_sarimax_fit_template,
        'ARIMA': train_arima,
        'PROPHET': train_prophet,
        'XGBOOST': train_xgboost,
        'LightGBM': train_lgbm,
        'ENSEMBLE_VOTE': train_voting_regressor,
        'RandomForest': train_random_forest,
        'CatBoost': train_catboost
    }

    final_model = model_dispatcher[best_model](y_full, X_full)

    # Generate forecast dates
    last_date = df_clean.index.max()
    fc_index = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=long_horizon, freq="MS")

    # Create future feature set
    X_forecast = pd.DataFrame(index=fc_index)
    X_forecast['temp_indicator_sim'] = X_forecast.index.month.map(lambda m: 1 if m in [5,6,7,8,9,10] else 0)
    X_forecast['Month_sin'] = np.sin(2 * np.pi * X_forecast.index.month / 12)
    X_forecast['Month_cos'] = np.cos(2 * np.pi * X_forecast.index.month / 12)
    X_forecast['Month_lag_6'] = X_forecast['Month_sin'].shift(6).fillna(0)
    X_forecast['Month_lag_12'] = X_forecast['Month_sin'].shift(12).fillna(0)

    # Prediction using only the best model
    logger.info(f"Generating forecast using best model: {best_model}")

    if best_model in ['SARIMAX', 'ARIMA']:
        fc_long = final_model.predict(n_periods=long_horizon, X=X_forecast)
    elif best_model == 'PROPHET':
        future_df = X_forecast.copy()
        future_df["ds"] = fc_index
        fc_long = final_model.predict(future_df)['yhat']
    else:
        fc_long = final_model.predict(X_forecast)

    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        "date": fc_index,
        "forecast_long": fc_long,
    })
    
    forecast_df["forecast_short"] = forecast_df["forecast_long"]
    forecast_df.iloc[short_horizon:, forecast_df.columns.get_loc("forecast_short")] = np.nan

    # Save output
    forecast_csv = os.path.join(RESULTS_DIR, "fl_forecast.csv")
    forecast_df.to_csv(forecast_csv, index=False)
    logger.info(f"\nSaved best model forecast to {forecast_csv}")

    return metrics_df, forecast_df, best_model
