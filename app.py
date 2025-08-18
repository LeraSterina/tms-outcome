import os, base64
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import _diag

st.set_page_config(page_title="TMS Outcome Predictor", page_icon="design-00f48579-0d70-4b7b-9c04-7017c5570fbd.png", layout="centered")

# --- debug / health switches via query params ---
q = st.experimental_get_query_params()
if q.get("health", ["0"])[0] == "1":
    # Minimal page for curl/uptime checks
    st.write("OK")
    st.write(_diag.json_text())
    st.stop()
if q.get("debug", ["0"])[0] == "1":
    with st.sidebar:
        st.markdown("### Diagnostics")
        st.code(_diag.json_text(), language="json")

# --- safe model load (surface errors in UI) ---
pipe = None
numeric = []
categorical = []
model_error = None
try:
    bundle = joblib.load("model.pkl")
    pipe = bundle.get("pipeline")
    numeric = bundle.get("numeric", [])
    categorical = bundle.get("categorical", bundle.get("categororical", []))
except Exception as e:
    model_error = str(e)

# --- Header: clickable logo + title ---
logo_path = Path("design-00f48579-0d70-4b7b-9c04-7017c5570fbd.png")
logo_b64 = base64.b64encode(logo_path.read_bytes()).decode() if logo_path.exists() else ""
col1, col2 = st.columns([1,4])
with col1:
    if logo_b64:
        st.markdown(f'''
        <a href="https://siddiqi.bwh.harvard.edu/" target="_blank" rel="noopener">
            <img src="data:image/png;base64,{logo_b64}" width="120" alt="PsyNet Lab"/>
        </a>
        ''', unsafe_allow_html=True)
with col2:
    st.title("TMS Outcome Predictor (Session 36)")
    st.caption("Enter early responses and demographics to predict the final outcome.")

st.markdown("---")

if model_error:
    st.error(f"Model load failed: {model_error}")
    st.stop()
if not pipe or not (numeric or categorical):
    st.error("Model pipeline not available or feature lists missing.")
    st.stop()

# --- Inputs ---
score10 = st.number_input("Score at Session 10", value=0.0, min_value=0.0)
score20 = st.number_input("Score at Session 20", value=0.0, min_value=0.0)
score30 = st.number_input("Score at Session 30", value=0.0, min_value=0.0)
age = st.number_input("Age", min_value=18, max_value=100, value=40)
sex = st.selectbox("Sex", ["M","F"])
education = st.selectbox("Education", ["HS","College","Graduate"])
site = st.selectbox("Site", ["BWH","MGH","Other"])

errors = []
if age < 18 or age > 100: errors.append("Age must be between 18 and 100.")
if errors: st.warning(" • " + "\n • ".join(errors))

# --- Predict ---
if st.button("Predict final outcome") and not errors:
    row = {"score10":score10,"score20":score20,"score30":score30,"age":age,"sex":sex,"education":education,"site":site}
    X = pd.DataFrame([row])[numeric + categorical]
    yhat = float(pipe.predict(X)[0])
    st.markdown("### Predicted outcome at Session 36")
    st.metric(label="Score", value=f"{yhat:.2f}")
