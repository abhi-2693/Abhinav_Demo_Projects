import os
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from src.logger import logger
import streamlit as st

try:
    from src.config import RESULTS_DIR
except ImportError:
    from config import RESULTS_DIR

def run_eda(df_clean):
    avg_consumption = df_clean.groupby(df_clean.index.month)["value"].mean().round(2).reset_index()
    avg_consumption.rename(columns={'index': 'month', 'value': 'avg_value'}, inplace=True)

    plt.figure(figsize=(14, 5))
    plt.plot(avg_consumption.date, avg_consumption.avg_value, label='Average Consumption', linewidth=1, color='#14e8c8')
    plt.title("Monthly Average Electricity Consumption (Thousand MWh)")
    plt.xlabel("Month")
    plt.ylabel("Consumption")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    avg_ts_plot_path = os.path.join(RESULTS_DIR, "fl_monthly_avg_ts_plot.png")
    plt.savefig(avg_ts_plot_path)
    plt.close()
    logger.info(f"Saved monthly average time-series plot to {avg_ts_plot_path}")


    plt.figure(figsize=(14,5))
    plt.plot(df_clean.index, df_clean["value"], label="Consumption", linewidth=1, color='#14e8c8')
    plt.title("Florida Electricity Consumption (Monthly) Time Series")
    plt.xlabel("Date")
    plt.ylabel("Consumption (Thousand MWh)")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    ts_plot_path = os.path.join(RESULTS_DIR, "fl_ts_plot.png")
    plt.savefig(ts_plot_path)
    plt.close()
    logger.info(f"Saved time-series plot to {ts_plot_path}")

    logger.info("Performing STL decomposition (period=12) to check trend/seasonality...")
    stl = STL(df_clean["value"], period=12, robust=True)
    stl_res = stl.fit()
    fig = stl_res.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle("STL Decomposition (Trend / Seasonal / Resid)", fontsize=14)
    stl_path = os.path.join(RESULTS_DIR, "fl_stl_decomposition.png")
    plt.tight_layout()
    plt.savefig(stl_path)
    plt.close()
    logger.info(f"Saved STL decomposition to {stl_path}")

    adf_res = adfuller(df_clean["value"].dropna())
    logger.info(f"ADF Statistic : {adf_res[0]:.4f}")
    logger.info(f"p-value       : {adf_res[1]:.4f}")
    if adf_res[1] < 0.05:
        logger.info("=> Series appears STATIONARY (reject H0). SARIMA model may not need differencing (d=0).")
    else:
        logger.info("=> Series appears NON-STATIONARY (fail to reject H0). auto_arima will handle differencing.")

    # ------------- ACF & PACF diagnostics -------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    plot_acf(df_clean["value"], lags=40, ax=axes[0])
    axes[0].set_title("Autocorrelation (ACF)")

    plot_pacf(df_clean["value"], lags=40, ax=axes[1], method="ywm")
    axes[1].set_title("Partial Autocorrelation (PACF)")

    plt.tight_layout()
    acf_pacf_path = os.path.join(RESULTS_DIR, "fl_acf_pacf.png")
    plt.savefig(acf_pacf_path)
    plt.close()

    logger.info(f"Saved ACF/PACF plot to {acf_pacf_path}")

    st.info("\nEDA done")
    return df_clean
