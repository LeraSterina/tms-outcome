import streamlit as st
import pandas as pd
import math

st.set_page_config(page_title="TMS Outcome Predictors", page_icon="🧠", layout="centered")

# ===== Header =====
col1, col2 = st.columns([1,4])
with col1:
    st.image("logo.png", width=110)
with col2:
    st.title("Predict Probability of Treatment Response or Percent Change in Final QIDS Score Using Change in QIDS at Treatment Milestones (10,20,30)")
    st.caption(
        "This app uses **separate models** for each milestone (Treatment 10/20/30). "
        "Choose a milestone first—your results reflect **only** the inputs for that milestone."
    )

st.divider()

# ===== Constants (unchanged) =====
CI_PARAMS = {
    "Predicted Probability of Response From Progress at Treatment 10": {
        "order": ["(Intercept)", "pct_change_s2", "session_1", "basis24_demo_age", "sex"],
        "beta":  [-1.285726, 2.284841, -0.011642, 0.012988, -0.406301],
        "cov": [
            [ 0.297410, -0.013584, -0.008807, -0.002300, -0.015825],
            [-0.013584,  0.370821, -0.001943, -0.000209,  0.002740],
            [-0.008807, -0.001943,  0.000407,  0.000023, -0.000031],
            [-0.002300, -0.000209,  0.000023,  0.000090, -0.000005],
            [-0.015825,  0.002740, -0.000031, -0.000005,  0.013574],
        ],
    },
    "Predicted Probability of Response From Progress at Treatment 20": {
        "order": ["(Intercept)", "pct_change_s3", "session_1", "basis24_demo_age", "sex"],
        "beta":  [-1.838541, 3.910043, -0.021340, 0.014683, -0.727389],
        "cov": [
            [ 0.517603, -0.036762, -0.014784, -0.003507, -0.029540],
            [-0.036762,  0.763600, -0.002279, -0.000359,  0.005733],
            [-0.014784, -0.002279,  0.000642,  0.000046, -0.000076],
            [-0.003507, -0.000359,  0.000046,  0.000112, -0.000010],
            [-0.029540,  0.005733, -0.000076, -0.000010,  0.024228],
        ],
    },
    "Predicted Probability of Response From Progress at Treatment 30": {
        "order": ["(Intercept)", "pct_change_s4", "session_1", "basis24_demo_age", "sex"],
        "beta":  [-3.105403, 4.010878, -0.001933, 0.013706, -0.433236],
        "cov": [
            [ 0.930938, -0.077331, -0.009133, -0.003877, -0.036743],
            [-0.077331,  0.825982,  0.006996,  0.000057, -0.004979],
            [-0.009133,  0.006996,  0.000732,  0.000057, -0.000972],
            [-0.003877,  0.000057,  0.000057,  0.000290,  0.000090],
            [-0.036743, -0.004979, -0.000972,  0.000090,  0.030805],
        ],
    },
    "Predicted Percent Change in QIDS From Progress at Treatment 10": {
        "order": ["(Intercept)", "pct_change_s2", "session_1", "basis24_demo_age"],
        "beta":  [0.094591, 0.461364, 0.004714, 0.002096],
        "cov": [
            [ 0.002057, -0.001720, -0.000098, -0.000011],
            [-0.001720,  0.016904, -0.000028, -0.000013],
            [-0.000098, -0.000028,  0.000006,  0.000000],
            [-0.000011, -0.000013,  0.000000,  0.000001],
        ],
    },
    "Predicted Percent Change in QIDS From Progress at Treatment 20": {
        "order": ["(Intercept)", "pct_change_s3", "session_1", "basis24_demo_age"],
        "beta":  [0.058819, 0.614008, 0.003700, 0.001321],
        "cov": [
            [ 0.002353, -0.002471, -0.000095, -0.000014],
            [-0.002471,  0.018601, -0.000040, -0.000011],
            [-0.000095, -0.000040,  0.000007,  0.000000],
            [-0.000014, -0.000011,  0.000000,  0.000001],
        ],
    },
    "Predicted Percent Change in QIDS From Progress at Treatment 30": {
        "order": ["(Intercept)", "pct_change_s4", "session_1", "basis24_demo_age"],
        "beta":  [0.041973, 0.775197, 0.001496, 0.000547],
        "cov": [
            [ 0.002747, -0.002978, -0.000059, -0.000012],
            [-0.002978,  0.018864,  0.000022, -0.000010],
            [-0.000059,  0.000022,  0.000007,  0.000000],
            [-0.000012, -0.000010,  0.000000,  0.000001],
        ],
    },
}

# ===== Helpers =====
def logistic(x):
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def clamp_prop(v):  # keep within [-1, 1] for robustness
    return max(min(float(v), 1.0), -1.0)

def linpred_and_se(x_vec, beta, cov):
    z = sum(x*b for x, b in zip(x_vec, beta))
    var = 0.0
    k = len(x_vec)
    for i in range(k):
        for j in range(k):
            var += x_vec[i] * cov[i][j] * x_vec[j]
    se = math.sqrt(max(var, 0.0))
    return z, se

def ci_logistic_prob(x_vec, model_key):
    pars = CI_PARAMS[model_key]
    z, se = linpred_and_se(x_vec, pars["beta"], pars["cov"])
    return logistic(z), logistic(z - 1.96*se), logistic(z + 1.96*se)

def ci_gaussian_prop(x_vec, model_key):
    pars = CI_PARAMS[model_key]
    y, se = linpred_and_se(x_vec, pars["beta"], pars["cov"])
    return clamp_prop(y), clamp_prop(y - 1.96*se), clamp_prop(y + 1.96*se)

# ===== Sidebar: How inputs are interpreted =====
with st.sidebar:
    st.markdown("### How this works")
    st.markdown(
        "- **Pick one milestone** (Treatment 10/20/30). The app runs that interim’s model only.\n"
        "- **Common inputs** (age, sex, baseline QIDS at Session 1) are shared across models, "
        "but **percent change** must correspond to the **selected milestone**.\n"
        "- Outputs are **not aggregated** across milestones."
    )

# ===== Step 1: Choose Interim (required) =====
st.subheader("Step 1 — Choose the interim")
interval = st.selectbox(
    "Select the treatment milestone you are entering percent change for:",
    ["— Select —", "Treatment 10", "Treatment 20", "Treatment 30"],
    index=0,
    help="This choice controls which model runs. Other milestones are ignored."
)

if interval == "— Select —":
    st.info("Select Treatment 10, Treatment 20, or Treatment 30 to proceed. The app will only use the model for the milestone you choose.")
    st.stop()

# ===== Step 2: Baseline / Demographics =====
st.subheader("Step 2 — Baseline & demographics")
c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age (years)", min_value=12, max_value=100, value=35, step=1)
    s1  = st.number_input("Baseline QIDS total at Session 1", min_value=0, max_value=30, value=18, step=1,
                          help="Baseline depressive symptom severity at Session 1.")
with c2:
    sex_label = st.selectbox("Sex (model coding)", ["Female (0)", "Male (1)"],
                             help="Model uses 0=female, 1=male.")
    sex = 0 if sex_label.startswith("Female") else 1

# ===== Step 3: Percent change for the selected interim =====
st.subheader("Step 3 — Percent change at the selected milestone")
st.markdown(
    "**Enter the percent change at the selected milestone**.\n\n"
    "- You can enter as a **proportion** (e.g., `0.20` for 20%) or a **percent** (e.g., `20`).\n"
    "- Negative values are allowed for worsening (e.g., `-0.10` or `-10`)."
)

pct_default = 0.40
if interval == "Treatment 10":
    pct = st.number_input("Tx9 change (prop or %)", value=pct_default, format="%.3f")
    pct_var = "pct_change_s2"
    bin_key = "Tx9 Binomial"
    gau_key = "Tx9 Gaussian"
elif interval == "Tx19":
    pct = st.number_input("Tx19 change (prop or %)", value=pct_default, format="%.3f")
    pct_var = "pct_change_s3"
    bin_key = "Tx19 Binomial"
    gau_key = "Tx19 Gaussian"
else:
    pct = st.number_input("Tx29 change (prop or %)", value=pct_default, format="%.3f")
    pct_var = "pct_change_s4"
    bin_key = "Tx29 Binomial"
    gau_key = "Tx29 Gaussian"

# Accept either 0–1 proportion or 0–100 percent
pct = float(pct)
if abs(pct) > 1.0:
    pct = pct / 100.0
pct = clamp_prop(pct)

st.divider()

# ===== Build X vectors & compute =====
def x_vector(model_key, pct, s1, age, sex):
    order = CI_PARAMS[model_key]["order"]
    mapping = {
        "(Intercept)": 1.0,
        "session_1": float(s1),
        "basis24_demo_age": float(age),
        "sex": float(sex),
        "pct_change_s2": float(pct),
        "pct_change_s3": float(pct),
        "pct_change_s4": float(pct),
    }
    return [mapping[name] for name in order]

x_bin = x_vector(bin_key, pct, s1, age, sex)
x_gau = x_vector(gau_key, pct, s1, age, sex)

p, plo, phi = ci_logistic_prob(x_bin, bin_key)
g, glo, ghi = ci_gaussian_prop(x_gau, gau_key)

# ===== Results tables =====
prob_df = pd.DataFrame([{
    "Model": f"{interval} Binomial – Response probability",
    "Probability (0–1)": round(p, 6),
    "95% CI (0–1)": f"{round(plo,6)} – {round(phi,6)}",
    "Probability (%)": f"{round(100*p):.0f}%",
    "95% CI (%)": f"{round(100*plo,1)}% – {round(100*phi,1)}%",
}])

gauss_df = pd.DataFrame([{
    "Model": f"{interval} Gaussian – Predicted final improvement",
    "Proportion (0–1)": round(g, 6),
    "95% CI (0–1)": f"{round(glo,6)} – {round(ghi,6)}",
    "Predicted (%)": f"{round(100*g,1)}%",
    "95% CI (%)": f"{round(100*glo,1)}% – {round(100*ghi,1)}%",
}])

st.subheader(f"Response probability @ Tx35 ({interval} model only)")
st.dataframe(prob_df, use_container_width=True, hide_index=True)

st.subheader(f"Predicted final % improvement @ Treatment 36 ({interval} model only)")
st.dataframe(gauss_df, use_container_width=True, hide_index=True)

# ===== Export =====
inputs_table = pd.DataFrame([
    ["Interim (model)", interval],
    ["Age (years)", age],
    ["Sex (0=female,1=male)", sex],
    ["Baseline QIDS total at Session 1", s1],
    [f"{interval} change (proportion)", pct],
], columns=["Parameter", "Value"])

export_df = pd.concat([prob_df, gauss_df], ignore_index=True)

csv = []
csv.append("Inputs")
csv.append(inputs_table.to_csv(index=False))
csv.append("")
csv.append("Results")
csv.append(export_df.to_csv(index=False))
csv_bytes = ("\n".join(csv)).encode("utf-8")

st.download_button(
    label="📥 Download CSV (Inputs + Results, with 95% CIs)",
    data=csv_bytes,
    file_name=f"tms_outcome_results_{interval}.csv",
    mime="text/csv",
)

# ===== Highlights =====
a, b = st.columns(2)
with a:
    st.metric(
        f"{interval} response prob.",
        f"{round(100*p):.0f}%",
        help=f"95% CI: {round(100*plo,1)}% – {round(100*phi,1)}%"
    )
with b:
    st.metric(
        f"{interval} predicted final %",
        f"{round(100*g,1)}%",
        help=f"95% CI: {round(100*glo,1)}% – {round(100*ghi,1)}%"
    )

st.caption(
    "Each milestone has its **own** fitted model; results are **not aggregated**. "
    "CIs use the model variance–covariance matrix (delta method). Binomial CIs are mapped to probability via the inverse logit."
)
