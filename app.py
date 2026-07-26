"""
Diabetes Prediction System — Streamlit UI

Wraps the trained Linear Regression pipeline (StandardScaler + LinearRegression,
trained on the Pima Indians Diabetes Dataset) from `diabetes_ml_pipeline.ipynb`
in an interactive web app.

Run with:
    streamlit run app.py

On first run, if `diabetes_model.joblib` isn't found next to this file, the app
will automatically train it from `diabetes.csv` (must be in the same folder)
using the exact same cleaning/scaling/training steps as the notebook, then
cache it to disk so future runs load instantly.
"""

import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MODEL_PATH = "diabetes_model.joblib"
DATA_PATH = "diabetes.csv"

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

# (min, max, default, step, help) — ranges based on the dataset's real min/max
FEATURE_CONFIG = {
    "Pregnancies":              (0, 17, 1, 1, "Number of times pregnant"),
    "Glucose":                  (40, 200, 120, 1, "Plasma glucose concentration (mg/dL)"),
    "BloodPressure":            (20, 130, 70, 1, "Diastolic blood pressure (mm Hg)"),
    "SkinThickness":            (5, 100, 20, 1, "Triceps skinfold thickness (mm)"),
    "Insulin":                  (0, 850, 80, 5, "2-Hour serum insulin (mu U/mL)"),
    "BMI":                      (15.0, 70.0, 25.0, 0.1, "Body Mass Index (weight in kg / (height in m)^2)"),
    "DiabetesPedigreeFunction": (0.05, 2.50, 0.47, 0.01, "Genetic diabetes risk score based on family history"),
    "Age":                      (18, 90, 30, 1, "Age in years"),
}

st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        color: #9ca3af;
        font-size: 1.02rem;
        margin-bottom: 1.4rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 12px 16px 6px 16px;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        height: 3em;
    }
    .result-card {
        border-radius: 14px;
        padding: 22px 26px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">🩺 Diabetes Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Linear Regression model trained on the Pima Indians Diabetes Dataset — '
    'enter a patient\'s health readings to estimate diabetes risk</div>',
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Train (mirrors the notebook exactly) — only runs if no saved model exists
# --------------------------------------------------------------------------
def train_and_save_model():
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(DATA_PATH)
    df_clean = df.copy()

    medical_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in medical_cols:
        df_clean[col] = df_clean[col].replace(0, np.nan)
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    X = df_clean.drop(columns=["Outcome"])
    y = df_clean["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    save_data = {"model": model, "scaler": scaler, "columns": list(X.columns)}
    joblib.dump(save_data, MODEL_PATH)
    return save_data


@st.cache_resource(show_spinner=False)
def load_pipeline():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    if os.path.exists(DATA_PATH):
        return train_and_save_model()
    return None


with st.spinner("Loading model..."):
    pipeline = load_pipeline()

if pipeline is None:
    st.error(
        "No trained model found. Place either **diabetes_model.joblib** "
        "(exported from the notebook) or **diabetes.csv** (the raw Pima "
        "Indians Diabetes dataset) in the same folder as this app, then "
        "restart it."
    )
    st.stop()

model = pipeline["model"]
scaler = pipeline["scaler"]
columns = pipeline["columns"]

# --------------------------------------------------------------------------
# Sidebar — patient inputs
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("🧾 Patient Health Readings")
    st.caption("Adjust the sliders to match the patient's readings, then click Predict.")

    inputs = {}
    for feat in FEATURES:
        lo, hi, default, step, help_text = FEATURE_CONFIG[feat]
        if isinstance(step, float):
            inputs[feat] = st.slider(feat, float(lo), float(hi), float(default), step, help=help_text)
        else:
            inputs[feat] = st.slider(feat, int(lo), int(hi), int(default), int(step), help=help_text)

    st.divider()
    presets = st.selectbox(
        "Or load an example patient",
        ["-- none --", "Low-risk example", "High-risk example"],
    )

# Apply presets by overriding the widget defaults via session state trick:
# simplest approach — recompute inputs directly if a preset is chosen.
if presets == "Low-risk example":
    inputs = dict(zip(FEATURES, [1, 85, 66, 29, 0, 26.6, 0.351, 31]))
elif presets == "High-risk example":
    inputs = dict(zip(FEATURES, [6, 148, 72, 35, 0, 33.6, 0.627, 50]))

# --------------------------------------------------------------------------
# Main panel
# --------------------------------------------------------------------------
col_form, col_result = st.columns([1, 1.1], gap="large")

with col_form:
    st.subheader("📋 Current Readings")
    display_df = pd.DataFrame({"Feature": FEATURES, "Value": [inputs[f] for f in FEATURES]})
    st.dataframe(display_df, hide_index=True, use_container_width=True)

    predict_clicked = st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True)

with col_result:
    st.subheader("📊 Prediction")

    if predict_clicked:
        patient_df = pd.DataFrame([[inputs[f] for f in FEATURES]], columns=columns)
        patient_scaled = scaler.transform(patient_df)

        raw_score = model.predict(patient_scaled)[0]
        score = float(np.clip(raw_score, 0, 1))
        is_diabetic = score >= 0.5

        if is_diabetic:
            st.markdown(
                f"""
                <div class="result-card" style="background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.35);">
                    <h3 style="margin:0; color:#ef4444;">🔴 Likely Diabetic</h3>
                    <p style="color:#9ca3af; margin-top:6px;">Risk score: <b>{score:.2f}</b> / 1.00</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card" style="background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.35);">
                    <h3 style="margin:0; color:#22c55e;">🟢 Likely Not Diabetic</h3>
                    <p style="color:#9ca3af; margin-top:6px;">Risk score: <b>{score:.2f}</b> / 1.00</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={"valueformat": ".2f"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#ef4444" if is_diabetic else "#22c55e"},
                "steps": [
                    {"range": [0, 0.5], "color": "rgba(34,197,94,0.15)"},
                    {"range": [0.5, 1], "color": "rgba(239,68,68,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.8,
                    "value": 0.5,
                },
            },
            title={"text": "Diabetes Risk Score"},
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Feature contribution breakdown
        st.markdown("**What drove this prediction?**")
        contributions = model.coef_ * patient_scaled[0]
        contrib_df = pd.DataFrame({
            "Feature": columns,
            "Contribution": contributions,
        }).sort_values("Contribution")

        bar_fig = go.Figure(go.Bar(
            x=contrib_df["Contribution"],
            y=contrib_df["Feature"],
            orientation="h",
            marker_color=["#ef4444" if v > 0 else "#22c55e" for v in contrib_df["Contribution"]],
        ))
        bar_fig.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Contribution to risk score",
        )
        st.plotly_chart(bar_fig, use_container_width=True)
        st.caption("🔴 Red bars push the prediction toward diabetic · 🟢 Green bars push it toward not diabetic")

    else:
        st.info("Set the patient's readings on the left and click **Predict Diabetes Risk**.")

st.divider()
st.caption(
    "⚠️ Educational project only — not a medical diagnostic tool. Built as part of a "
    "CodeAlpha AI/ML internship task, using Linear Regression on the Pima Indians Diabetes Dataset."
)
