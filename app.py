from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Landslide Risk Monitor",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "landslide_random_forest.joblib"
SAMPLE_PATH = PROJECT_DIR / "data" / "sample_landslide_data.csv"

FEATURES = ["Soil_Moisture", "Vibration", "Rainfall"]
TARGET = "Landslide_Risk_Level"

# Clean, consistent dashboard styling
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 1200px;
        }
        .hero {
            padding: 1.5rem 1.7rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 18px;
            margin-bottom: 1.4rem;
        }
        .hero-kicker {
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            opacity: 0.7;
            margin-bottom: 0.5rem;
        }
        .hero h1 {
            margin: 0;
            font-size: clamp(1.8rem, 4vw, 2.5rem);
        }
        .hero p {
            margin: 0.55rem 0 0 0;
            opacity: 0.8;
            font-size: 1rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 14px;
            padding: 1rem;
        }
        div.stButton > button {
            min-height: 3rem;
            font-weight: 650;
            border-radius: 10px;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.2);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def make_demo_data(n=600, random_state=42):
    """Create synthetic demonstration data if no training CSV is deployed."""
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

    return pd.DataFrame(
        {
            "Soil_Moisture": soil,
            "Vibration": vibration,
            "Rainfall": rainfall,
            TARGET: risk,
        }
    )


@st.cache_resource
def load_or_train_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH), "Saved notebook model", False

    if SAMPLE_PATH.exists():
        training_data = pd.read_csv(SAMPLE_PATH)
        source = "Sample CSV"
        is_demo = False
    else:
        training_data = make_demo_data()
        source = "Synthetic demonstration data"
        is_demo = True

    missing_columns = [
        column for column in FEATURES + [TARGET]
        if column not in training_data.columns
    ]
    if missing_columns:
        raise ValueError(
            "Training data is missing required columns: "
            + ", ".join(missing_columns)
        )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(training_data[FEATURES], training_data[TARGET])
    return model, source, is_demo


# Sidebar: details are available without cluttering the main dashboard.
with st.sidebar:
    st.title("🌧️ Landslide Monitor")
    st.caption("Environmental risk classification")
    st.divider()
    st.subheader("Project details")
    st.write("**Algorithm:** Random Forest")
    st.write("**Inputs:** Soil moisture, vibration, rainfall")
    st.write("**Output:** Low, Medium, or High")
    st.divider()
    st.subheader("Model information")
    st.caption("The model source is shown here after loading.")
    st.caption(
        "For reliable real-world assessment, train and evaluate the model "
        "using verified, expert-labelled landslide data."
    )

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Environmental Monitoring</div>
        <h1>Landslide Risk Monitor</h1>
        <p>Assess potential risk from current soil, vibration, and rainfall readings.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, model_source, demo_model = load_or_train_model()
except Exception as exc:
    st.error(f"Unable to load the prediction model: {exc}")
    st.info(
        "Check that the model or sample CSV exists in the expected project folder "
        "and contains the required feature and target columns."
    )
    st.stop()

with st.sidebar:
    if demo_model:
        st.warning(f"Active model: {model_source}")
    else:
        st.success(f"Active model: {model_source}")

st.subheader("Sensor readings")
st.caption("Enter readings from your sensors, then run the risk assessment.")

col1, col2, col3 = st.columns(3)
with col1:
    soil = st.number_input(
        "Soil moisture (%)",
        min_value=0.0,
        max_value=100.0,
        value=78.0,
        step=1.0,
        help="Enter a value from 0 to 100. Confirm the scale used by your sensor.",
    )
with col2:
    vibration = st.number_input(
        "Vibration reading",
        min_value=0.0,
        max_value=1000.0,
        value=3.2,
        step=0.1,
        help="Use the same unit and scale as the data used to train the model.",
    )
with col3:
    rainfall = st.number_input(
        "Rainfall / raindrop reading",
        min_value=0.0,
        max_value=10000.0,
        value=35.0,
        step=1.0,
        help="A raindrop sensor value is not automatically rainfall in millimetres.",
    )

predict = st.button("Assess landslide risk", type="primary", use_container_width=True)

if predict:
    reading = pd.DataFrame(
        [
            {
                "Soil_Moisture": soil,
                "Vibration": vibration,
                "Rainfall": rainfall,
            }
        ]
    )

    try:
        prediction = str(model.predict(reading[FEATURES])[0])
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.divider()
    st.subheader("Risk assessment")

    result_col, detail_col = st.columns([1, 2], gap="large")
    with result_col:
        if prediction.lower() == "high":
            st.error(f"🔴 {prediction} risk")
        elif prediction.lower() == "medium":
            st.warning(f"🟠 {prediction} risk")
        else:
            st.success(f"🟢 {prediction} risk")

    with detail_col:
        st.markdown("**Submitted readings**")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Soil moisture", f"{soil:g}%")
        metric_cols[1].metric("Vibration", f"{vibration:g}")
        metric_cols[2].metric("Rainfall / raindrop", f"{rainfall:g}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(reading[FEATURES])[0]
        probability_table = pd.DataFrame(
            {
                "Risk level": model.classes_,
                "Model probability": probabilities,
            }
        ).sort_values("Model probability", ascending=False)
        probability_table["Model probability"] = probability_table[
            "Model probability"
        ].map(lambda value: f"{value:.1%}")

        with st.expander("View model probabilities"):
            st.dataframe(
                probability_table,
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "These are model scores, not verified probabilities of a real landslide."
            )

st.divider()
st.caption(
    "Educational prototype only. This prediction is not a validated emergency warning."
)

with st.expander("Model notes"):
    st.write(f"**Current model source:** {model_source}")
    if demo_model:
        st.warning(
            "This deployment is using synthetic demonstration data because neither "
            "a saved model nor the sample CSV was found. Its predictions demonstrate "
            "the application workflow and must not be interpreted as real-world risk."
        )
    st.write(
        "Before real-world use, calibrate the sensors, confirm input units, and "
        "evaluate the model against verified historical or expert-labelled outcomes."
    )
