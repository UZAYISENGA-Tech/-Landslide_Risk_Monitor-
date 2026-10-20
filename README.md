# Landslide-Risk Monitor — Jupyter Notebook Project

## Project goal
Use a Random Forest classifier to estimate a landslide risk category from:
- soil moisture
- vibration
- rainfall

The model output is **Low**, **Medium**, or **High** risk.

## Important data limitation
The sensor Excel file previously discussed (`Raw_Sensor_Data_Landslide_Monitoring.xlsx`) contains sensor measurements such as `Soil20cm`, `Vibration`, and `Raindrop`, but no verified landslide-risk label column was found. A supervised classifier needs labelled examples. This project therefore:
1. Runs immediately using `data/sample_landslide_data.csv`, which is synthetic demo data.
2. Can load your real Excel file if you copy it into `data/`.
3. If your real Excel has no risk label, creates **proxy labels from a simple rule** only to demonstrate the ML workflow. These are NOT observed landslide outcomes and are NOT a validated warning system.

Do not use demo/proxy predictions for real emergency decisions. For a scientifically valid model, collect confirmed landslide/non-landslide or risk labels from experts / historical events and evaluate on those labels.

## Files
- `Landslide_Risk_Monitor.ipynb` — main notebook
- `data/sample_landslide_data.csv` — synthetic demo data to test the notebook
- `requirements.txt` — Python packages
- `run_notebook_instructions.txt` — how to run in Jupyter
- `models/` and `outputs/` — locations for generated model and evaluation outputs

## Run in Jupyter
1. Extract the ZIP.
2. Open Jupyter Notebook or JupyterLab.
3. Open `Landslide_Risk_Monitor.ipynb`.
4. Run cells from top to bottom.
5. To use your actual sensor Excel file, place it in `data/` with the name `Raw_Sensor_Data_Landslide_Monitoring.xlsx`. The notebook will prefer it over the sample CSV.

## Sensor mapping for your Excel dataset
- `Soil20cm` → `Soil_Moisture` (one depth measurement; optionally improve by engineering a mean of Soil20cm/40cm/60cm)
- `Vibration` → `Vibration`
- `Raindrop` → `Rainfall` proxy. This is a raindrop sensor count/reading, not necessarily rainfall in mm. Calibrate it before interpreting it as rainfall.

If your Excel has a real label column such as `Landslide_Risk_Level`, `Risk_Level`, or `Label`, the notebook uses it. Otherwise, it makes rule-based proxy labels and prints a warning.
