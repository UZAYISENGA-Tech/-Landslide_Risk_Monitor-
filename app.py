from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Landslide-Risk Monitor", page_icon="🌧️", layout="centered")

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "landslide_random_forest.joblib"
SAMPLE_PATH = PROJECT_DIR / "data" / "sample_landslide_data.csv"

st.title("🌧️ Landslide-Risk Monitor")
st.write("Enter sensor readings to estimate a landslide risk category using a Random Forest classifier.")

st.warning(
    "Educational prototype only. Predictions are not a validated emergency warning. "
    "If the model was trained on demonstration/proxy labels, it reproduces those labels "
    "and does not prove real-world landslide prediction accuracy."
)

@st.cache_resource
def load_or_train_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH), "saved notebook model"

    # Fallback so the app can run before the notebook has been executed.
    # This trains on synthetic demonstration data included in the ZIP.
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Sample dataset not found: {SAMPLE_PATH}")
    df = pd.read_csv(SAMPLE_PATH)
    features = ["Soil_Moisture", "Vibration", "Rainfall"]
    target = "Landslide_Risk_Level"
    model = __import__("sklearn.ensemble", fromlist=["RandomForestClassifier"]).RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(df[features], df[target])
    return model, "synthetic sample data (demo model)"

try:
    model, model_source = load_or_train_model()
except Exception as exc:
    st.error(f"Could not load or train the model: {exc}")
    st.stop()

st.caption(f"Model source: {model_source}")

st.subheader("Sensor readings")
col1, col2 = st.columns(2)
with col1:
    soil = st.number_input("Soil moisture (0–100)", min_value=0.0, max_value=100.0, value=78.0, step=1.0)
    vibration = st.number_input("Vibration reading", min_value=0.0, max_value=1000.0, value=3.2, step=0.1)
with col2:
    rainfall = st.number_input("Rainfall / raindrop reading", min_value=0.0, max_value=10000.0, value=35.0, step=1.0)

if st.button("Predict landslide risk", type="primary", use_container_width=True):
    sample = pd.DataFrame([{
        "Soil_Moisture": soil,
        "Vibration": vibration,
        "Rainfall": rainfall
    }])
    prediction = model.predict(sample)[0]
    st.subheader("Predicted risk level")
    if prediction == "High":
        st.error(f"🔴 {prediction} risk")
    elif prediction == "Medium":
        st.warning(f"🟠 {prediction} risk")
    else:
        st.success(f"🟢 {prediction} risk")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(sample)[0]
        prob_df = pd.DataFrame({"Risk level": model.classes_, "Model probability": probs})
        st.write("Model class probabilities (not a guarantee of actual risk):")
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

with st.expander("About this project"):
    st.markdown(
        "- **Inputs:** soil moisture, vibration, and rainfall/raindrop reading.\n"
        "- **Algorithm:** Random Forest Classification.\n"
        "- **Output:** Low, Medium, or High risk level.\n"
        "- For meaningful real-world evaluation, train on verified historical or expert-labelled risk outcomes. "
        "A raindrop sensor reading must be calibrated before treating it as rainfall in millimetres."
    )
