# Diabetes Prediction System

A Streamlit web app for the **Pima Indians Diabetes Dataset** pipeline —
predicts whether a patient is likely diabetic from 8 health readings, using
a Linear Regression model (StandardScaler + LinearRegression), matching the
training steps in `diabetes_ml_pipeline.ipynb`.

## Project structure

```
diabetes_app/
├── app.py               # Streamlit UI
├── train_model.py       # standalone training script (mirrors the notebook)
├── diabetes.csv         # dataset (you provide this — see Setup)
├── diabetes_model.joblib  # trained model (auto-generated on first run)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Add the dataset: place `diabetes.csv` (the Pima Indians Diabetes Dataset) in
this folder. Get it from Kaggle ("Pima Indians Diabetes Database") if you
don't already have it from the notebook.

## Run

```bash
streamlit run app.py
```

On first launch, if `diabetes_model.joblib` isn't found, the app trains it
automatically from `diabetes.csv` (same cleaning → scaling → training steps
as the notebook) and caches it to disk, so every run after that loads
instantly.

Alternatively, train it manually up front:

```bash
python train_model.py
```

## What the app does

- **Sidebar sliders** for all 8 features: Pregnancies, Glucose, Blood
  Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age.
- Two **example presets** (low-risk / high-risk) to demo quickly.
- **Predict button** → risk score (0–1), a Diabetic / Not Diabetic verdict,
  and a gauge chart.
- **Feature contribution chart** showing which readings pushed the
  prediction up (red) or down (green) — direct read-out of the model's
  learned coefficients times the patient's scaled values.

## Deploying to Streamlit Community Cloud

1. Push this folder (including `diabetes.csv`) to a GitHub repo.
2. On [share.streamlit.io](https://share.streamlit.io), deploy pointing at
   `app.py`.
3. If you hit `ImportError: import cv2` or similar native-library errors —
   not applicable here since this app has no OpenCV dependency, but if you
   add one later, remember `opencv-python-headless` + a `packages.txt` with
   `libgl1` / `libglib2.0-0` is the standard fix.
4. Pin `runtime.txt` to `python-3.11` if Streamlit Cloud defaults to a very
   new Python version that some dependency doesn't have wheels for yet.

## Notes

- The notebook (and this app) intentionally uses **Linear Regression**, not
  Logistic Regression, per the assigned task — predictions are clipped to
  [0, 1] and rounded at a 0.5 threshold to act as a classifier.
- This is an educational project, not a medical diagnostic tool.
