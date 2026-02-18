import os
import pandas as pd
import streamlit as st
from src.config import PROCESSED, RESULTS_DIR, OOT_RESULTS_CSV, OOT_SUMMARY_CSV, REGION, SHORT_TERM_FORECAST_HORIZON, LONG_TERM_FORECAST_HORIZON
from src.train import main_train_pipeline

# Streamlit settings
st.set_page_config(page_title="Energy Forecasting Dashboard", layout="wide")

# Sidebar
st.sidebar.title("Team :\n **Group 13 (Term 4)**")
st.sidebar.markdown("\n## Contributers:\n")
st.sidebar.markdown("**Abhinav Paul**")


st.title(f"SMBA Pvt Ltd.\nEnergy Demand Forecasting Dashboard \n\nRegion: {REGION}")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚀 Run Pipeline", "📈 Historical Series", "📊 Model Metrics", "🔮 Forecast Results", "📜 Logs"])

# ---------- TAB 1: Run Pipeline ----------
with tab1:
    st.subheader("🚀 Run Training & Forecasting Pipeline")
    
    if st.button("Start Training Pipeline"):
        log_placeholder = st.empty()

        with st.spinner("Training models... This will take sometime to complete.\n\nKeep calm and take deep breaths :)"):
            metrics_df, forecast_df, best_model_name = main_train_pipeline()
        st.success(f"🎉 Training Complete! Best Model: **{best_model_name}**")

# ---------- TAB 2: Historical Processed Series ----------
with tab2:
    st.subheader("📈 Historical Processed Series")
    if os.path.exists(PROCESSED):
        df_processed = pd.read_csv(PROCESSED, parse_dates=["date"], index_col="date")
        st.line_chart(df_processed["value"])
    
    st.subheader("Average Consumption Plot")
    avg_ts_plot_path = os.path.join(RESULTS_DIR, "fl_monthly_avg_ts_plot.png")
    if os.path.exists(avg_ts_plot_path):
        st.image(avg_ts_plot_path)
        
    st.subheader("BASE Time Series Plot")
    ts_plot_path = os.path.join(RESULTS_DIR, "fl_ts_plot.png")
    if os.path.exists(ts_plot_path):
        st.image(ts_plot_path)
    
    st.subheader("STL Decomposition")
    stl_path = os.path.join(RESULTS_DIR, "fl_stl_decomposition.png")
    if os.path.exists(stl_path):
        st.image(stl_path)

    st.subheader("ACF & PACF")
    acf_pacf_path = os.path.join(RESULTS_DIR, "fl_acf_pacf.png")
    if os.path.exists(acf_pacf_path):
        st.image(acf_pacf_path)

    else:
        st.warning("No processed data available. Please run the pipeline first.")

# ---------- TAB 3: Model Metrics ----------
with tab3:
    st.subheader("📊 Model Performance Comparison")
    
    if os.path.exists(OOT_RESULTS_CSV):
        metrics_df = pd.read_csv(OOT_RESULTS_CSV)
        st.dataframe(metrics_df.sort_values(['Model', 'Fold'], ascending=[True, True]), width='stretch')

    if os.path.exists(OOT_SUMMARY_CSV):
        metrics_df = pd.read_csv(OOT_SUMMARY_CSV)
        st.dataframe(metrics_df, width='stretch')    
        # Display Best Model
        best_row = metrics_df.loc[metrics_df['MAPE_mean'].idxmin()]
        st.metric("🏆 Best Model", best_row["Model"], f"{best_row['MAPE_mean']:.2f}% MAPE")
        
    else:
        st.warning("No metrics available. Please run the pipeline first.")

# ---------- TAB 4: Forecast Results ----------
with tab4:
    st.subheader("🔮 Forecast Visualization")

    forecast_file = os.path.join(RESULTS_DIR, "fl_forecast.csv")
    
    if os.path.exists(forecast_file):
        forecast_df = pd.read_csv(forecast_file, parse_dates=["date"]).set_index("date")
        
        st.subheader(f"Short-Term Forecast ({SHORT_TERM_FORECAST_HORIZON}m)")
        st.line_chart(forecast_df["forecast_short"], use_container_width=True)
        
        st.subheader(f"Long-Term Forecast ({LONG_TERM_FORECAST_HORIZON}m)")
        st.line_chart(forecast_df["forecast_long"], use_container_width=True)
        
        st.write("📥 Download Forecast CSV")
        st.download_button("Download CSV", forecast_df.to_csv().encode('utf-8'), "forecast_results.csv", "text/csv")
     
    else:
        st.warning("No forecast available. Run pipeline first.")

# ---------- TAB 5: Logs ----------
with tab5:
    st.subheader("📜 Training Logs")
    if os.path.exists("logs/train.log"):
        with open("logs/train.log", "r") as log_file:
            st.text_area("Logs", log_file.read(), height=400)
    else:
        st.info("No logs yet. Run pipeline to generate logs.")
