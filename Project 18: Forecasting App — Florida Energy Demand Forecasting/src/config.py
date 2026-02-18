import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

API_KEY = os.getenv("EIA_API_KEY")
SERIES_ID = os.getenv("EIA_SERIES_ID")
BASE_URL = os.getenv("EIA_BASE_URL")
REGION = os.getenv("EIA_REGION")

SHORT_TERM_FORECAST_HORIZON = os.getenv("SHORT_TERM_FORECAST_HORIZON")
LONG_TERM_FORECAST_HORIZON = os.getenv("LONG_TERM_FORECAST_HORIZON")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR   = os.path.join(BASE_DIR, "data")
RAW_FILE   = os.path.join(DATA_DIR, "raw", "florida_consumption.csv")
PROCESSED  = os.path.join(DATA_DIR, "processed", "florida_consumption_processed.csv")

RESULTS_DIR = os.path.join(DATA_DIR, "results")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
OOT_RESULTS_CSV = os.path.join(RESULTS_DIR, "cv_results.csv")
OOT_SUMMARY_CSV = os.path.join(RESULTS_DIR, "cv_results_SUMMARY.csv")

# Ensure directories exist
os.makedirs(os.path.join(DATA_DIR, "raw"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
