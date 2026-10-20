from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Landslide-Risk Monitor", page_icon="🌧️", layout="centered")

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "landslide_random_forest.joblib"
SAMPLE_PATH = PROJECT_DIR / "data" / "sample_landslide_data.csv"

FEATURES = ["Soil_Moisture", "Vibration", "Rainfall"]
TARGET = "Landslide_Risk_Level"


def make_demo_data(n=600, random_state=42):
    """Create synthetic demonstration data if the CSV was not included in deployment."""
    rng = np.random.default_rng(random_state)
    soil = rng.uniform(15, 100, n)
    vibration = np.clip(rng.lognormal(mean=-1.8, sigma=1.0, size=n), 0, 20)
    rainfall = np.clip(rng.gamma(shape=1.2, scale=12, size=n), 0, 120)
    score = (
        0.50 * (soil / 100)
        + 0.30 * (np.minimum(vibration, 15) / 15)
        + 0.20 * (rainfall / 120)
    )
    risk = np.where(score < 0.32, "Low", np.where(score < 0.57, "Medium", "High"))
    return pd.DataFrame({
        "Soil_Moisture": soil,
        "Vibration": vibration,
        "Rainfall": rainfall,
        TARGET: risk,
    })


@st.cache_resource
def load_or_train_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH), "saved model from the Jupyter notebook", False

    if SAMPLE_PATH.exists():
        training_data = pd.read_csv(SAMPLE_PATH)
        source = "included sample CSV"
    else:
        training_data = make_demo_data()
        source = "generated synthetic demo data (the CSV file was missing)"

    missing_columns = [c for c in FEATURES + [TARGET] if c not in training_data.columns]
    if missing_columns:
        raise ValueError(
            "Training data is missing required columns: " + ", ".join(missing_columns)
        )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(training_data[FEATURES], training_data[TARGET])
    return model, source, True


st.title("🌧️ Landslide-Risk Monitor")
st.write(
    "Enter soil moisture, vibration, and rainfall/raindrop readings to obtain "
    "a Low, Medium, or High classification."
)
st.warning(
    "Educational prototype only. This is not a validated emergency warning system. "
    "Predictions from synthetic or proxy labels do not demonstrate real-world landslide "
    "prediction accuracy."
)

try:
    model, model_source, demo_model = load_or_train_model()
except Exception as exc:
    st.error(f"Could not load or train the model: {exc}")
    st.info(
        "Check that app.py, requirements.txt, and any model/data files are committed to "
        "the same GitHub repository. The app can also run a demo model without the CSV."
    )
    st.stop()

st.caption(f"Model source: {model_source}")
if demo_model:
    st.info(
        "The app is running with demonstration data because no saved notebook model was found. "
        "For your coursework, run the notebook and deploy the saved model if you have verified labels."
    )

st.subheader("Sensor readings")
col1, col2 = st.columns(2)
with col1:
    soil = st.number_input(
        "Soil moisture (0–100)", min_value=0.0, max_value=100.0, value=78.0, step=1.0
    )
    vibration = st.number_input(
        "Vibration reading", min_value=0.0, max_value=1000.0, value=3.2, step=0.1
    )
with col2:
    rainfall = st.number_input(
        "Rainfall / raindrop reading", min_value=0.0, max_value=10000.0, value=35.0, step=1.0
    )

if st.button("Predict landslide risk", type="primary", use_container_width=True):
    reading = pd.DataFrame([{
        "Soil_Moisture": soil,
        "Vibration": vibration,
        "Rainfall": rainfall,
    }])
    prediction = str(model.predict(reading[FEATURES])[0])
    st.subheader("Predicted risk level")
    if prediction.lower() == "high":
        st.error(f"🔴 {prediction} risk")
    elif prediction.lower() == "medium":
        st.warning(f"🟠 {prediction} risk")
    else:
        st.success(f"🟢 {prediction} risk")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(reading[FEATURES])[0]
        probability_table = pd.DataFrame({
            "Risk level": model.classes_,
            "Model probability": [f"{p:.1%}" for p in probabilities],
        })
        st.write("Model class probabilities (not a guarantee of actual risk):")
        st.dataframe(probability_table, use_container_width=True, hide_index=True)

with st.expander("About this project"):
    st.markdown(
        "- **Inputs:** soil moisture, vibration, and rainfall/raindrop reading.\n"
        "- **Algorithm:** Random Forest Classification.\n"
        "- **Output:** Low, Medium, or High risk level.\n"
        "- A raindrop sensor reading is not automatically rainfall in millimetres; calibrate it.\n"
        "- For real-world evaluation, use verified historical or expert-labelled landslide outcomes."
    )
