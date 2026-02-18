# src/train.py

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

from src.config import OOT_RESULTS_CSV, MODEL_DIR, OOT_SUMMARY_CSV, LONG_TERM_FORECAST_HORIZON
from src.data_download import fetch_eia_series_v2
from src.features import data_preparation
from src.eda_script import run_eda
import streamlit as st
from prophet.serialize import model_to_json

progress_bar = st.progress(0)

# Import model utilities (shared helpers)
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

def split_data(df_clean: pd.DataFrame, n_holdout=int(LONG_TERM_FORECAST_HORIZON)):
    if len(df_clean) <= (n_holdout + 12):
        n_holdout = min(LONG_TERM_FORECAST_HORIZON, max(12, int(len(df_clean) * 0.2)))
        st.info(f"Adjusted holdout to {n_holdout} months.")

    train_df = df_clean.iloc[:-n_holdout]
    test_df = df_clean.iloc[-n_holdout:]

    return (
        train_df["value"], train_df.drop(columns=["value"]),
        test_df["value"], test_df.drop(columns=["value"])
    )

def train_all_models(y_train, X_train, y_test, X_test, st_progress=None):

    models = {}
    total_steps = 6
    step = 0

    st.info("Training SARIMAX...")
    models['SARIMAX'] = auto_sarimax_fit_template(y_train, X_train)
    step += 1
    if st_progress: st_progress.progress(step / total_steps)

    st.info("Training ARIMA...")
    models['ARIMA'] = train_arima(y_train, X_train)
    step += 1
    if st_progress: st_progress.progress(step / total_steps)

    st.info("Training Prophet...")
    models['PROPHET'] = train_prophet(y_train, X_train)
    step += 1
    if st_progress: st_progress.progress(step / total_steps)

    st.info("Training XGBoost...")
    models['XGBOOST'] = train_xgboost(y_train, X_train)
    step += 1
    if st_progress: st_progress.progress(step / total_steps)

    st.info("Training LightGBM...")
    models['LightGBM'] = train_lgbm(y_train, X_train)
    step += 1
    if st_progress: st_progress.progress(step / total_steps)

    st.info("Training Voting Regressor...")
    models['ENSEMBLE_VOTE'] = train_voting_regressor(models, y_train, X_train)
    step += 1
    if st_progress: st_progress.progress(step / total_steps)

    st.info("Training RandomForest...")
    models["RandomForest"] = train_random_forest(y_train, X_train)
    step += 1
    if st_progress: st_progress.progress(step / total_steps)

    st.info("Training CatBoost...")
    models["CatBoost"] = train_catboost(y_train, X_train)
    step += 1
    if st_progress: st_progress.progress(1.0)

    for name, model in models.items():
        if name == 'PROPHET':
            with open(os.path.join(MODEL_DIR, f"{name}_final.json"), 'w') as f:
                json.dump(model_to_json(model), f)
        else:
            joblib.dump(model, os.path.join(MODEL_DIR, f"{name}_final.pkl"))
    return models

#BACKTEST
def validation_forecast(models, df_clean, n_splits=25):
    st.info(f"\nBacktesting models with {n_splits} folds...")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_cv_metrics = []

    y = df_clean["value"]
    X = df_clean.drop(columns=['value'])

    fold = 1

    for train_idx, test_idx in tscv.split(df_clean):        
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

        fold_start_date, fold_end_date = df_clean.index[train_idx[0]], df_clean.index[test_idx[-1]]
        st.info(f"\tFold {fold} : Fold Start Date: {fold_start_date} to Fold End Date: {fold_end_date}")
        
        for model_name, model_obj in models.items():
            # Predict
            if model_name in ["SARIMAX", "ARIMA"]:
                fc = model_obj.predict(n_periods=len(y_test), X=X_test)
            elif model_name == "PROPHET":
                future_df = pd.DataFrame({"ds": y_test.index})
                for col in X_test.columns:
                    future_df[col] = X_test[col].values
                fc = model_obj.predict(future_df)["yhat"].values
            else:  # XGB / LGBM / Ensemble
                fc = model_obj.predict(X_test)

            y_true = y_test.values

            # Metrics
            rmse = np.sqrt(mean_squared_error(y_true, fc))
            mae = mean_absolute_error(y_true, fc)
            mape = mean_absolute_percentage_error(y_true, fc) * 100
            bias = np.mean(fc - y_true)

            all_cv_metrics.append({
                "Fold": fold,
                "Fold Start Date": fold_start_date,
                "Fold End Date": fold_end_date,
                "Model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "Bias": bias
            })

        fold += 1

    # SAVE RAW CSV
    cv_df = pd.DataFrame(all_cv_metrics)
    cv_df.to_csv(OOT_RESULTS_CSV, index=False)
    
    # SUMMARY TABLE (MEAN + STD)
    summary = cv_df.groupby("Model").agg(
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        MAPE_mean=("MAPE", "mean"),
        MAPE_std=("MAPE", "std"),
        Bias_mean=("Bias", "mean"),
        Bias_std=("Bias", "std")
    ).reset_index()

    summary = summary.sort_values("MAPE_mean")
    summary.to_csv(OOT_SUMMARY_CSV, index=False)

    return cv_df, summary

def main_train_pipeline():
    df_raw = fetch_eia_series_v2()
    df_clean = data_preparation(df_raw)
    run_eda(df_clean)

    y_train, X_train, y_test, X_test = split_data(df_clean)
    models = train_all_models(y_train, X_train, y_test, X_test)
    # Time-Series Cross Validation 
    cv_metrics = validation_forecast(models, df_clean, n_splits=25)

    # Import locally to avoid circular dependency
    from src.predict import final_forecast  

    metrics_df, forecast_df, best_model = final_forecast(df_clean)
    st.success("\nPipeline Execution Complete.")

    return metrics_df, forecast_df, best_model

if __name__ == "__main__":
    with st.spinner("Training in progress..."):
        main_train_pipeline()
    
