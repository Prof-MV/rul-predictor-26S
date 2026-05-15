"""
Predictive Maintenance — RUL Predictor
Streamlit app template for Week 5 capstone deployment.

Students: replace placeholders marked with TODO.
Run locally: streamlit run streamlit_app.py
Deploy: push to public GitHub repo, then connect at share.streamlit.io
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Engine RUL Predictor",
    page_icon="🔧",
    layout="wide",
)

st.title("🔧 Turbofan Engine RUL Predictor")
st.caption("Predicts Remaining Useful Life (RUL) in operating cycles from sensor data.")
# TODO: add your name and course

# -------------------------------------------------------------------
# Load model and feature list (cached so it loads once)
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("rul_model.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return model, feature_cols

try:
    model, feature_cols = load_model()
except FileNotFoundError:
    st.error("Model files not found. Make sure rul_model.pkl and feature_cols.pkl are in the repo.")
    st.stop()

# -------------------------------------------------------------------
# Helper: build rolling features for an uploaded engine history
# -------------------------------------------------------------------
USEFUL_SENSORS = [f"sensor_{i}" for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]

def add_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add rolling mean and std for each useful sensor. Assumes single engine."""
    out = df.copy()
    for sensor in USEFUL_SENSORS:
        if sensor in out.columns:
            out[f"{sensor}_rolling_mean"] = out[sensor].rolling(window, min_periods=1).mean()
            out[f"{sensor}_rolling_std"] = out[sensor].rolling(window, min_periods=1).std().fillna(0)
    return out

# -------------------------------------------------------------------
# Sidebar — input mode
# -------------------------------------------------------------------
st.sidebar.header("Input")
mode = st.sidebar.radio("How do you want to provide data?",
                        ["Upload CSV (full engine history)", "Use sample data"])

# -------------------------------------------------------------------
# Main panel
# -------------------------------------------------------------------
df_input = None

if mode == "Upload CSV (full engine history)":
    uploaded = st.sidebar.file_uploader(
        "CSV with columns: time_in_cycles, sensor_1 ... sensor_21",
        type=["csv"],
    )
    if uploaded is not None:
        df_input = pd.read_csv(uploaded)
elif mode == "Use sample data":
    if st.sidebar.button("Load sample engine"):
        # TODO: replace with a small CSV in your repo
        st.info("Add a sample.csv to your repo and load it here.")

# -------------------------------------------------------------------
# Predict and display
# -------------------------------------------------------------------
if df_input is not None and len(df_input) > 0:
    st.subheader("Input data preview")
    st.dataframe(df_input.tail(10), use_container_width=True)

    # Feature engineering
    df_features = add_rolling_features(df_input)

    # Use only the most recent cycle for the prediction
    latest = df_features.iloc[[-1]]

    # Align to model's expected columns
    missing = [c for c in feature_cols if c not in latest.columns]
    if missing:
        st.error(f"Input is missing features: {missing[:5]}...")
        st.stop()

    X = latest[feature_cols]
    rul_pred = float(model.predict(X)[0])

    # ----- prediction display with color -----
    if rul_pred < 30:
        color, status = "red", "⚠️ CRITICAL — schedule maintenance now"
    elif rul_pred < 60:
        color, status = "orange", "🟡 CAUTION — plan maintenance soon"
    else:
        color, status = "green", "✅ HEALTHY — no immediate action"

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Predicted RUL", f"{rul_pred:.0f} cycles")
        st.markdown(f"<h3 style='color:{color}'>{status}</h3>", unsafe_allow_html=True)
    with col2:
        # ----- sensor trend plot -----
        st.subheader("Recent sensor trends")
        sensors_to_plot = ["sensor_11", "sensor_4", "sensor_12", "sensor_7"]
        sensors_to_plot = [s for s in sensors_to_plot if s in df_input.columns]
        fig, ax = plt.subplots(figsize=(8, 4))
        for s in sensors_to_plot:
            ax.plot(df_input["time_in_cycles"], df_input[s], label=s)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Sensor value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
else:
    st.info("👈 Upload a CSV or load sample data from the sidebar to get a prediction.")

# -------------------------------------------------------------------
# About expander — required for the documentation rubric
# -------------------------------------------------------------------
with st.expander("ℹ️ About this model — limitations and honest disclosure"):
    st.markdown("""
**Training data:** NASA C-MAPSS turbofan FD001 (1 operating condition, 1 fault mode, 100 engines).

**Model:** Random Forest regressor with rolling-window features, RUL clipped at 125 cycles.

**Validation performance:** RMSE ≈ 20 cycles on held-out engines. This means predictions
of "60 cycles remaining" could realistically be anywhere from 40 to 80.

**Do NOT use this model for:**
- Real engines outside the C-MAPSS dataset
- Different operating conditions than FD001
- Safety-critical maintenance decisions without engineering review

This is a class project. A production predictive-maintenance system needs continuous
monitoring, drift detection, retraining, and validation against your specific equipment.
    """)
    # TODO: add your name and contact
