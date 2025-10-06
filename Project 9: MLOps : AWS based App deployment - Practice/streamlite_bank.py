# ================================================================
# BANK CUSTOMER SUBSCRIPTION PREDICTION APP
# ================================================================
# • Uses artefacts generated in the training script:
#       – bank_pre_process.joblib   (preprocessing pipeline)
#       – best_fit_model.pkl        (trained classifier)
# • Upload customer data (csv / xlsx) and receive predictions
# • Optional: persist results to MySQL for reporting
# ----------------------------------------------------------------

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
from PIL import Image
from sqlalchemy import create_engine
from urllib.parse import quote
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------
# 1 -- Load artefacts
# ---------------------------------------------------------------
MODEL_PATH = "best_fit_model.pkl"
PIPE_PATH  = "bank_pre_process.joblib"

# Store the model and pipeline as global variables
global model, pipeline

# Load the pipeline first with error handling
if not os.path.exists(PIPE_PATH):
    raise FileNotFoundError(f"Pipeline file not found at {os.path.abspath(PIPE_PATH)}")
pipeline = joblib.load(PIPE_PATH)
print("Successfully loaded preprocessing pipeline")

# Load the model with error handling
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {os.path.abspath(MODEL_PATH)}")

try:
    # Load the model and pipeline
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPE_PATH)
    
    print(f"Successfully loaded model from {os.path.abspath(MODEL_PATH)}")
    print(f"Model type: {type(model)}")
    
    # Verify the model is fitted and has predict method
    if not hasattr(model, 'predict'):
        raise ValueError("Loaded model does not have a predict method")
    print("Model has predict method")
    
    if hasattr(model, 'get_params'):
        print(f"Model parameters: {model.get_params()}")
    print(f"Successfully loaded pipeline from {os.path.abspath(PIPE_PATH)}")

except Exception as e:
    st.error(f"Error loading model or pipeline: {str(e)}")
    st.error(f"Error type: {type(e).__name__}")
    st.error(f"Error details: {str(e)}")
    st.stop()

# ---------------------------------------------------------------
# 2 -- Helper: timestamp feature engineering
#      (mirrors the transformation done in training script)
# ---------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for modeling"""

    for col in df.columns:
        df.rename(columns={col: col.replace('.', '_').replace('-', '_')}, inplace=True)

    # Replace 'unknown' values with NaN and drop them
    df.replace('unknown', pd.NA, inplace=True)
    df.dropna(inplace=True)
    
    # Convert month to numerical (Jan=1, Feb=2, etc.)
    month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    df['month_num'] = df['month'].map(month_map)
    # Create quarter feature
    df['quarter'] = pd.cut(df['month_num'], bins=[0, 3, 6, 9, 12], 
                          labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df['age_bin'] = pd.cut(df['age'], bins=[0, 20, 40, 60, 80], 
                          labels=['0-20', '20-40', '40-60', '60-80'])
    df['interaction_duration_type'] = np.where(df['duration']>np.mean(df['duration']), 'Long', 'Short')
    df['campaign_type'] = np.where(df['campaign']>np.mean(df['campaign']), 'High', 'Low')
    df['last_contact_type'] = np.where(df['pdays']<0, 'Never','Prev_contacted')
    
    # Label encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object', 'category']).columns:
        print(col)
        df[col] = le.fit_transform(df[col])
        print(df[col].describe())
    
    return df

# ---------------------------------------------------------------
# 3 -- Core inference routine
# ---------------------------------------------------------------
def make_predictions(raw_df: pd.DataFrame,
                     db_user: str,
                     db_pw:   str,
                     db_name: str
                     ) -> pd.DataFrame:

    # 3-A  feature engineering (outside pipeline)
    df = engineer_features(raw_df.copy())

    # 3-B  preprocessing + feature-selection
    X_clean = pipeline.transform(df)              # ndarray

    # 3-C  inference
    preds   = model.predict(X_clean)

    # 3-D  assemble results
    result  = pd.concat([pd.Series(preds, name="Predicted_Status"), df.reset_index(drop=True)], axis=1)

    # 3-E  optional DB persistence
    if all([db_user, db_pw, db_name]):
        endpoint = 'database-1.cvqcm6owuev9.ap-southeast-2.rds.amazonaws.com'            # skip if credentials blank
        try:
            eng = create_engine(
                f"mysql+pymysql://{db_user}:{quote(db_pw)}@{os.getenv(endpoint)}/{db_name}"
            )
            result.to_sql("bank_predictions", eng, if_exists="replace",
                          index=False, chunksize=1000)
            st.success("Predictions saved to MySQL - table `bank_predictions`")
        except Exception as e:
            st.error(f"Could not write to MySQL: {e}")

    return result

# ---------------------------------------------------------------
# 4 -- Streamlit UI
# ---------------------------------------------------------------
def main():
    st.title("Bank Default Prediction")
    st.sidebar.header("Upload CSV / Excel File")

    # -- file upload
    file = st.sidebar.file_uploader("Choose CSV or Excel", type=["csv", "xlsx"])
    if not file:
        st.info("Awaiting file upload …")
        st.stop()

    # -- read file
    try:
        data = pd.read_csv(file)
    except Exception:
        data = pd.read_excel(file)


    # -- MySQL creds (optional)
    st.sidebar.header("Optional - MySQL Persistence")
    user = st.sidebar.text_input("User")
    pw   = st.sidebar.text_input("Password", type="password")
    db   = st.sidebar.text_input("Database")

    if st.button("Predict"):
        with st.spinner("Running inference …"):
            results = make_predictions(data,user,pw,db)

        # -- pretty display
        cmap = sns.light_palette("green", as_cmap=True)
        st.subheader("Results")
        st.table(results.style.background_gradient(cmap=cmap))

        # quick counts
        counts = results["Predicted_Status"].value_counts().to_dict()
        st.info(f"Will Subscribe: {counts.get(1,0)} | Will Not Subscribe: {counts.get(0,0)}")
        

# ---------------------------------------------------------------
if __name__ == "__main__":
    main()

