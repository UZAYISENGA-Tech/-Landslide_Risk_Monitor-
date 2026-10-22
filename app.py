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
