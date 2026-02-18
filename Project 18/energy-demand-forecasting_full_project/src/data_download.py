import pandas as pd
import requests
from src.logger import logger
import streamlit as st

try:
    from src.config import API_KEY, SERIES_ID, BASE_URL, RAW_FILE, REGION
except ImportError:
    from config import API_KEY, SERIES_ID, BASE_URL, RAW_FILE, REGION

def fetch_eia_series_v2(api_key: str = API_KEY, series_id: str = SERIES_ID) -> pd.DataFrame:
    endpoint = f"{BASE_URL}{series_id}/"
    params = {"api_key": api_key, "out": "json"}
    st.info(f"\nAPI Call: Fetching data for {REGION}")
    r = requests.get(endpoint, params=params, timeout=30)

    if r.status_code != 200:
        raise Exception(f"API Error {r.status_code}: {r.text}")

    js = r.json()
    rows = js.get("response", {}).get("data", [])

    if not rows:
        raise RuntimeError("No data returned in the 'response/data' payload.")

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df["period"], format="%Y-%m")
    df['value'] = pd.to_numeric(df["sales"], errors="coerce")
    df = df[["date", "value"]].sort_values("date").reset_index(drop=True)

    df.to_csv(RAW_FILE, index=False)
    logger.info(f"Saved raw data artifact to {RAW_FILE}")
    return df

if __name__ == "__main__":
    fetch_eia_series_v2()
