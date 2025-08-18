import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="TMS Outcome Predictor", page_icon="🧠", layout="centered")

# Header with clickable logo + title
st.markdown("""
<div style="display:flex;gap:14px;align-items:center;margin:10px 0 20px 0">
  <a href="https://siddiqi.bwh.harvard.edu/" target="_blank" rel="noopener">
    <img src="https://raw.githubusercontent.com/LeraSterina/tms-outcome/main/design-00f48579-0d70-4b7b-9c04-7017c5570fbd.png"
         alt="PsyNet Lab" width="64">
  </a>
  <div>
    <h1 style="margin:0">TMS Outcome Predictor (Session 36)</h1>
    <div style="opacity:.7">Enter early responses and demographics to predict the final outcome.</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Load trained pipeline if present
pipe = numeric = categorical = None
if os.path.exists("model.pkl"):
    bundle = joblib.load("model.pkl")
    pipe = bundle["pipeline"]
    numeric = bundle["numeric"]
    categorical = bundle["categorical"]

# Inputs
score10 = st.number_input("Score at Session 10", value=0.0)
score20 = st.number_input("Score at Session 20", value=0.0)
score30 = st.number_input("Score at Session 30", value=0.0)
age = st.number_input("Age", min_value=18, max_value=100, value=40)
sex = st.selectbox("Sex", ["M", "F"])
education = st.selectbox("Education", ["HS", "College", "Graduate"])
site = st.selectbox("Site", ["BWH", "MGH", "Other"])

# Predict
if st.button("Predict final outcome"):
    if pipe is None:
        st.error("model.pkl not found in repo. Commit it, then redeploy.")
    else:
        X = pd.DataFrame([{
            "score10": score10,
            "score20": score20,
            "score30": score30,
            "age": age,
            "sex": sex,
            "education": education,
            "site": site,
        }])[numeric + categorical]
        yhat = float(pipe.predict(X)[0])
        st.success(f"Predicted outcome at Session 36: {yhat:.2f}")

st.caption("© PsyNet Lab — demo")
