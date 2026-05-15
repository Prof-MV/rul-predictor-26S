"""
Predictive Maintenance — RUL Predictor
Streamlit app template for Week 5 capstone deployment.

Students: replace placeholders marked with TODO.
This template is designed to run on Streamlit Community Cloud
(no local installation required).

Repo must contain alongside this file:
  - rul_model.pkl          (your trained Random Forest from Week 4)
  - feature_cols.pkl       (your feature column list from Week 4)
  - sample.csv             (a one-engine sensor history for demos)
  - requirements.txt       (Python dependencies)
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
# Derive which sensors need rolling features by inspecting the model's feature list.
# This way, the template works no matter which sensors your specific model used.
USEFUL_SENSORS = sorted(set(
    c.replace("_rolling_mean", "").replace("_rolling_std", "")
    for c in feature_cols
    if "_rolling_" in c
))

def add_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add rolling mean and std for each useful sensor. Assumes single engine."""
    out = df.copy()
    for sensor in USEFUL_SENSORS:
        if sensor in out.columns:
            out[f"{sensor}_rolling_mean"] = out[sensor].rolling(window, min_periods=1).mean()
            out[f"{sensor}_rolling_std"] = out[sensor].rolling(window, min_periods=1).std().fillna(0)
    return out

# -------------------------------------------------------------------
# Sidebar — input mode and sample download
# -------------------------------------------------------------------
st.sidebar.header("Input")
mode = st.sidebar.radio(
    "How do you want to provide data?",
    ["Upload CSV (full engine history)", "Use sample data"],
)

# Sample download button — lets visitors grab the input format
try:
    with open("sample.csv", "rb") as _f:
        st.sidebar.download_button(
            "📥 Download sample CSV (input format)",
            _f.read(),
            "sample.csv",
            "text/csv",
            help="Use this as a template for your own input",
        )
except FileNotFoundError:
    pass  # sample.csv not in repo; skip the download button silently

# -------------------------------------------------------------------
# Get input data
# -------------------------------------------------------------------
df_input = None

if mode == "Upload CSV (full engine history)":
    uploaded = st.sidebar.file_uploader(
        "CSV with columns: time_in_cycles, sensor_1 ... sensor_21",
        type=["csv"],
    )
    if uploaded is not None:
        try:
            df_input = pd.read_csv(uploaded)
            st.sidebar.success(f"Loaded {len(df_input)} cycles")
        except Exception as e:
            st.sidebar.error(f"Could not read CSV: {e}")

elif mode == "Use sample data":
    if st.sidebar.button("Load sample engine"):
        try:
            df_input = pd.read_csv("sample.csv")
            st.sidebar.success(f"Loaded sample with {len(df_input)} cycles")
        except FileNotFoundError:
            st.sidebar.error(
                "sample.csv not found in repo. "
                "Generate it by running the Task 1 snippet in your Colab notebook, "
                "then upload to the repo root."
            )
        except Exception as e:
            st.sidebar.error(f"Error loading sample: {e}")

# -------------------------------------------------------------------
# Predict and display
# -------------------------------------------------------------------
if df_input is not None and len(df_input) > 0:
    st.subheader("Input data preview (last 10 cycles)")
    st.dataframe(df_input.tail(10), use_container_width=True)

    # Feature engineering
    df_features = add_rolling_features(df_input)

    # Use only the most recent cycle for the prediction
    latest = df_features.iloc[[-1]]

    # Align to model's expected columns
    missing = [c for c in feature_cols if c not in latest.columns]
    if missing:
        st.error(f"Input is missing required features. First few: {missing[:5]}")
        st.info(
            "If you uploaded a CSV from a different source, check that column names "
            "match what the model expects (sensor_1, sensor_2, ..., sensor_21)."
        )
        st.stop()

    X = latest[feature_cols]
    rul_pred = float(model.predict(X)[0])

    # ----- prediction display with color -----
    # TODO: These threshold values are placeholders. In production they would come
    # from the maintenance team based on scheduling lead time, cost of unplanned
    # failure, and operational risk tolerance.
    CRITICAL_THRESHOLD = 30
    CAUTION_THRESHOLD = 60

    if rul_pred < CRITICAL_THRESHOLD:
        color, status = "red", "⚠️ CRITICAL — schedule maintenance now"
    elif rul_pred < CAUTION_THRESHOLD:
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
    st.info("👈 Use the sidebar to upload a CSV or load sample data.")

# -------------------------------------------------------------------
# About expander — required for the documentation rubric
# -------------------------------------------------------------------
with st.expander("ℹ️ About this model — limitations and honest disclosure"):
    st.markdown("""
**Training data:** NASA C-MAPSS turbofan FD001 (1 operating condition, 1 fault mode, 100 engines).

**Model:** Random Forest regressor with rolling-window features over 5 cycles,
RUL labels clipped at 125, trained on 80 engines and validated on 20.

**Validation performance:** RMSE ≈ 20 cycles on held-out engines. This means
predictions of "60 cycles remaining" could realistically be anywhere from 40 to 80.

**Threshold values (30 cycles for red, 60 cycles for yellow) are placeholders.**
In production these would come from the maintenance team based on:
- Scheduling lead time (how long to plan and execute a repair)
- Cost of unplanned failure (downtime, secondary damage)
- Operating risk tolerance

**Do NOT use this model for:**
- Real engines outside the C-MAPSS dataset
- Operating conditions different from FD001
- Safety-critical maintenance decisions without engineering review
- Anything that affects life, limb, or significant capital

This is a class project. A production predictive-maintenance system needs
continuous monitoring, drift detection, retraining against new data, and
validation against your specific equipment.
    """)
    # TODO: add your name and contact
