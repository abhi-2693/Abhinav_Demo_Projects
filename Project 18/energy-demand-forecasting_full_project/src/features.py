import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from src.logger import logger
import streamlit as st

try:
    from src.config import PROCESSED
except ImportError:
    from config import PROCESSED

def data_preparation(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.set_index("date").copy()
    logger.info(f"Total data rows: {len(df_clean)}")

    df_clean = df_clean.dropna()
    logger.info(f"Dropped rows with missing values: {len(df_clean)}")

    df_clean = df_clean[df_clean["value"] > 0]
    logger.info(f"Dropped rows with non-positive values: {len(df_clean)}")

    df_clean = df_clean.asfreq("MS")
    logger.info(f"Dropped rows with non-positive values: {len(df_clean)}")

    if df_clean["value"].isna().sum() > 0:
        logger.info("Interpolating missing months:", df_clean["value"].isna().sum())
        df_clean["value"] = df_clean["value"].interpolate(method="time")

    np.random.seed(42)

    # Create a new feature column to identify the high consumption months
    # This new feature is binary (1 if month is in [5, 6, 7, 8, 9, 10], else 0)
    # This can help the model learn different patterns for peak vs off-peak seasons
    df_clean['temp_indicator_sim'] = df_clean.index.month.map(
        lambda m: 1 if m in [5, 6, 7, 8, 9, 10] else 0
    )

    # Create two new feature columns for sine and cosine of the month component of the date index
    # These features can be used to capture seasonal patterns in the consumption data
    df_clean['Month_sin'] = np.sin(2 * np.pi * df_clean.index.month / 12)
    df_clean['Month_cos'] = np.cos(2 * np.pi * df_clean.index.month / 12)
    
    # Add shifted features
    df_clean['Month_lag_6'] = df_clean['value'].shift(6).fillna(0)
    df_clean['Month_lag_12'] = df_clean['value'].shift(12).fillna(0)

    df_clean.to_csv(PROCESSED, index=True)
    logger.info(f"\nSaved processed series to {PROCESSED}")
    st.info("\nData Preparation done")
    return df_clean
