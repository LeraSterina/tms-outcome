import os, base64
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="TMS Outcome Predictor", page_icon="design-00f48579-0d70-4b7b-9c04-7017c5570fbd.png", layout="centered")

bundle = joblib.load("model.pkl")
pipe = bundle["pipeline"]
numeric = bundle["numeric"]
categorical = bundle["categororical"] if "categororical" in bundle else bundle["categorical"]

logo_path = Path("design-00f48579-0d70-4b7b-9c04-7017c5570fbd.png")
logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()

col1, col2 = st.columns([1,4])
with col1:
    st.markdown(f'''
    <a href="https://siddiqi.bwh.harvard.edu/" target="_blank" rel="noopener">
        <img src="data:image/png;base64,{logo_b64}" width="120" alt="PsyNet Lab"/>
    </a>
    ''', unsafe_allow_html=True)
with col2:
    st.title("TMS Outcome Predictor (Session 36)")
    st.caption("Enter early responses and demographics to predict the final outcome.")

st.markdown("---")

APP_PASS = os.environ.get("APP_PASS", "")
if APP_PASS:
    pw = st.text_input("Enter access code", type="password")
    if pw != APP_PASS:
        st.stop()

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

if st.button("Predict final outcome") and not errors:
    row = {"score10":score10,"score20":score20,"score30":score30,"age":age,"sex":sex,"education":education,"site":site}
    X = pd.DataFrame([row])[numeric + categorical]
    yhat = float(pipe.predict(X)[0])
    st.markdown("### Predicted outcome at Session 36")
    st.metric(label="Score", value=f"{yhat:.2f}")

with st.expander("About this tool"):
    st.markdown("""
- Prototype for educational use; not a clinical device.
- Inputs: early session scores (10/20/30) + demographics.
- Output: predicted outcome at session 36 (regression baseline).
""")
st.markdown('Built by [PsyNet Lab](https://siddiqi.bwh.harvard.edu/). Questions? <a href="mailto:lerasterin@gmail.com">Email us</a>.', unsafe_allow_html=True)
st.code("https://tmssiddiqilottery.streamlit.app/", language="text")
