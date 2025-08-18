import base64
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# ---- load trained pipeline ----
bundle = joblib.load("model.pkl")
pipe = bundle["pipeline"]
numeric = bundle["numeric"]
categorical = bundle["categorical"]

# ---- embed logo as base64 (avoids path issues) ----
logo_path = Path("design-00f48579-0d70-4b7b-9c04-7017c5570fbd.png")
logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()

# ---- Header: clickable logo + title side-by-side ----
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown(
        f'''
        <a href="https://siddiqi.bwh.harvard.edu/" target="_blank" rel="noopener">
            <img src="data:image/png;base64,{logo_b64}" width="120" alt="PsyNet Lab"/>
        </a>
        ''',
        unsafe_allow_html=True
    )
with col2:
    st.title("TMS Outcome Predictor (Session 36)")
    st.caption("Enter early responses and demographics to predict the final outcome.")

st.markdown("---")

# ---- Inputs ----
score10 = st.number_input("Score at Session 10", value=0.0)
score20 = st.number_input("Score at Session 20", value=0.0)
score30 = st.number_input("Score at Session 30", value=0.0)
age = st.number_input("Age", min_value=18, max_value=100, value=40)

sex = st.selectbox("Sex", ["M", "F"])
education = st.selectbox("Education", ["HS", "College", "Graduate"])
site = st.selectbox("Site", ["BWH", "MGH", "Other"])

# ---- Predict ----
if st.button("Predict final outcome"):
    row = {
        "score10": score10,
        "score20": score20,
        "score30": score30,
        "age": age,
        "sex": sex,
        "education": education,
        "site": site,
    }
    X = pd.DataFrame([row])[numeric + categorical]
    yhat = float(pipe.predict(X)[0])
    st.success(f"Predicted outcome at Session 36: {yhat:.2f}")
