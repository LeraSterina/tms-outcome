import streamlit as st
import pandas as pd
import math

st.set_page_config(page_title="TMS Outcome Predictors", page_icon="🧠", layout="centered")

# --------- header (single, at top) ---------
col1, col2 = st.columns([1,4])
with col1:
    st.image("logo.png", width=110)
with col2:
    st.title("TMS Outcome Predictors (Tx35)")
    st.caption("Enter age, sex, baseline QIDS, and percent change at Tx9/19/29 to estimate response probability and predicted final % improvement at Tx35.")

st.divider()

# --------- helpers (logistic & gaussian) ---------
def logistic(x):  # 1 / (1 + exp(-x))
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def clamp_prop(v):
    # allow a bit beyond [0,1] or negative if worsened; clamp for display
    return max(min(v, 1.0), -1.0)

# Binomial models (probability of response by Tx35)
def prob_tx9(pct, s1, age, sex):
    z = (-1.285726 + 2.284841*pct + -0.011642*s1 + 0.012988*age + -0.406301*sex)
    return logistic(z)

def prob_tx19(pct, s1, age, sex):
    z = (-1.838541 + 3.910043*pct + -0.02134*s1 + 0.014683*age + -0.727389*sex)
    return logistic(z)

def prob_tx29(pct, s1, age, sex):
    z = (-3.105403 + 4.010878*pct + -0.001933*s1 + 0.013706*age + -0.433236*sex)
    return logistic(z)

# Gaussian models (predicted final % improvement at Tx35; return proportion)
def gauss_tx9(pct, s1, age):
    return (0.094591 + 0.461364*pct + 0.004714*s1 + 0.002096*age)

def gauss_tx19(pct, s1, age):
    return (0.058819 + 0.614008*pct + 0.0037*s1 + 0.001321*age)

def gauss_tx29(pct, s1, age):
    return (0.041973 + 0.775197*pct + 0.001496*s1 + 0.000547*age)

def fmt_pct(x):   # display proportion as 0–100%
    return f"{round(100*clamp_prop(x),1)}%"

def fmt_prob(p):  # display probability as 0–100%
    return f"{round(100*clamp_prop(p),0):.0f}%"

# --------- inputs ---------
st.subheader("Inputs")
c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age (years)", min_value=12, max_value=100, value=35, step=1)
    s1  = st.number_input("Baseline QIDS total at Session 1 (≥6)", min_value=0, max_value=30, value=18, step=1)
with c2:
    sex_label = st.selectbox("Sex (model coding)", ["Female (0)", "Male (1)"])
    sex = 0 if sex_label.startswith("Female") else 1
    st.caption("Sex coded as 0=female, 1=male for the models.")

st.markdown("**Percent change at each interim treatment** (enter as a proportion; e.g., 0.20 for 20%, −0.10 if worse).")
p9, p19, p29 = st.columns(3)
with p9:
    pct9  = st.number_input("Tx9 Δ% (prop)", value=0.40, format="%.3f")
with p19:
    pct19 = st.number_input("Tx19 Δ% (prop)", value=0.40, format="%.3f")
with p29:
    pct29 = st.number_input("Tx29 Δ% (prop)", value=0.40, format="%.3f")

st.divider()

# --------- compute ---------
rows = [
    ("Tx9 Binomial – Response probability",   prob_tx9(pct9,  s1, age, sex),  None),
    ("Tx9 Gaussian – Predicted final improvement", None,                     gauss_tx9(pct9,  s1, age)),
    ("Tx19 Binomial – Response probability",  prob_tx19(pct19, s1, age, sex), None),
    ("Tx19 Gaussian – Predicted final improvement", None,                    gauss_tx19(pct19, s1, age)),
    ("Tx29 Binomial – Response probability",  prob_tx29(pct29, s1, age, sex), None),
    ("Tx29 Gaussian – Predicted final improvement", None,                    gauss_tx29(pct29, s1, age)),
]

# --------- split tables (no Nones) ---------
prob_rows = [r for r in rows if "Binomial" in r[0]]
gauss_rows = [r for r in rows if "Gaussian" in r[0]]

prob_df = pd.DataFrame([{
    "Model": name,
    "Probability (0–1)": round(p, 6),
    "Probability (%)": f"{round(100*p):.0f}%"
} for name, p, g in prob_rows])

gauss_df = pd.DataFrame([{
    "Model": name,
    "Proportion (0–1)": round(g, 6),
    "Predicted (%)": f"{round(100*g,1)}%"
} for name, p, g in gauss_rows])

st.subheader("Response probability @ Tx35")
st.dataframe(prob_df, use_container_width=True, hide_index=True)

st.subheader("Predicted final % improvement @ Tx35")
st.dataframe(gauss_df, use_container_width=True, hide_index=True)

# --------- highlights ---------
st.markdown("### Highlights")
a, b, c = st.columns(3)
with a:
    st.metric("Response prob. @Tx9",  fmt_prob(prob_rows[0][1]))
with b:
    st.metric("Response prob. @Tx19", fmt_prob(prob_rows[1][1]))
with c:
    st.metric("Response prob. @Tx29", fmt_prob(prob_rows[2][1]))

d, e, f = st.columns(3)
with d:
    st.metric("Predicted final % @Tx9",  fmt_pct(gauss_rows[0][2]))
with e:
    st.metric("Predicted final % @Tx19", fmt_pct(gauss_rows[1][2]))
with f:
    st.metric("Predicted final % @Tx29", fmt_pct(gauss_rows[2][2]))
