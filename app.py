import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="ABC Ltd — Customer Retention Predictor", page_icon="📊", layout="centered")

# ----------------------------------------------------------------------
# 1. LOAD the pre-trained model bundle (fast — no training at runtime).
#    Generated once, offline, by running: python train_model.py
# ----------------------------------------------------------------------
@st.cache_resource
def load_models():
    with open("model_bundle.pkl", "rb") as f:
        return pickle.load(f)

M = load_models()

# ----------------------------------------------------------------------
# 2. UI
# ----------------------------------------------------------------------
st.title("📊 ABC Ltd — Customer Retention Predictor")
st.caption("Trained on 7,032 anonymized customer records. No data-science background required.")

tab1, tab2 = st.tabs(["🚨 Churn risk", "💰 Customer value"])

# ---------------- TAB 1: CHURN RISK ----------------
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("Tenure (months with us)", 0, 72, 24)
        mc = st.slider("Monthly bill ($)", 18, 120, 65)
        contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "None"])
    with col2:
        payment = st.selectbox("Payment method", ["Bank transfer (automatic)", "Credit card (automatic)",
                                                    "Electronic check", "Mailed check"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        family = st.selectbox("Family status", ["Neither", "Partner only", "Partner + dependents"])

    if st.button("Predict churn risk", type="primary"):
        row = pd.DataFrame([{
            "tenure": tenure, "MonthlyCharges": mc, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Contract_One year": 1 if contract == "One year" else 0,
            "Contract_Two year": 1 if contract == "Two year" else 0,
            "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
            "InternetService_No": 1 if internet == "None" else 0,
            "PaymentMethod_Credit card (automatic)": 1 if payment == "Credit card (automatic)" else 0,
            "PaymentMethod_Electronic check": 1 if payment == "Electronic check" else 0,
            "PaymentMethod_Mailed check": 1 if payment == "Mailed check" else 0,
            "PaperlessBilling_Yes": 1 if paperless == "Yes" else 0,
            "Partner_Yes": 1 if family in ["Partner only", "Partner + dependents"] else 0,
            "Dependents_Yes": 1 if family == "Partner + dependents" else 0,
        }])[M["logit_cols"]]
        row[M["num_cols"]] = M["scaler"].transform(row[M["num_cols"]])

        prob = M["logit"].predict_proba(row)[0, 1]
        pct = round(prob * 100)

        if pct < 30:
            st.success(f"### {pct}% risk — LOW RISK")
        elif pct < 60:
            st.warning(f"### {pct}% risk — WATCH")
        else:
            st.error(f"### {pct}% risk — HIGH RISK")
        st.progress(prob)

        st.markdown("**Why the model thinks this:**")
        drivers = []
        if contract == "Month-to-month": drivers.append("Month-to-month contract (no lock-in) → raises risk")
        elif contract == "Two year": drivers.append("Two-year contract (strong lock-in) → lowers risk")
        else: drivers.append("One-year contract → lowers risk")
        if internet == "Fiber optic": drivers.append("Fiber optic service (higher-churn segment) → raises risk")
        if tenure < 12: drivers.append(f"Short tenure ({tenure} months) → raises risk")
        elif tenure > 48: drivers.append(f"Long tenure ({tenure} months) → lowers risk")
        if payment == "Electronic check": drivers.append("Pays by electronic check → raises risk")
        if mc > 80: drivers.append(f"High monthly bill (${mc}) → raises risk")
        for d in drivers[:5]:
            st.markdown(f"- {d}")

        m = M["logit_metrics"]
        st.caption(f"Model: logistic regression · accuracy {m['accuracy']:.0%} · "
                   f"ROC-AUC {m['roc_auc']:.2f} · recall {m['recall']:.0%} · "
                   f"baseline churn rate {M['churn_rate']:.1%}")

# ---------------- TAB 2: CUSTOMER VALUE ----------------
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        tenure2 = st.slider("Tenure (months)", 0, 72, 24, key="t2")
        mc2 = st.slider("Monthly bill ($)", 18, 120, 65, key="m2")
        contract2 = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"], key="c2")
    with col2:
        internet2 = st.selectbox("Internet service", ["DSL", "Fiber optic", "None"], key="i2")
        payment2 = st.selectbox("Payment method", ["Bank transfer (automatic)", "Credit card (automatic)",
                                                     "Electronic check", "Mailed check"], key="p2")

    if st.button("Predict total billing", type="primary"):
        row2 = pd.DataFrame([{
            "tenure": tenure2, "MonthlyCharges": mc2,
            "Contract_One year": 1 if contract2 == "One year" else 0,
            "Contract_Two year": 1 if contract2 == "Two year" else 0,
            "InternetService_Fiber optic": 1 if internet2 == "Fiber optic" else 0,
            "InternetService_No": 1 if internet2 == "None" else 0,
            "PaymentMethod_Credit card (automatic)": 1 if payment2 == "Credit card (automatic)" else 0,
            "PaymentMethod_Electronic check": 1 if payment2 == "Electronic check" else 0,
            "PaymentMethod_Mailed check": 1 if payment2 == "Mailed check" else 0,
        }])[M["lin_cols"]]

        pred = max(0, M["linreg"].predict(row2)[0])
        st.metric("Predicted total billing to date", f"${pred:,.0f}")

        lm = M["lin_metrics"]
        st.caption(f"Model: linear regression · R² = {lm['r2']:.2f} "
                   f"(explains {lm['r2']:.0%} of the variation) · average error ≈ ${lm['mae']:,.0f}")

st.divider()
st.caption("Built for ABC Ltd's managerial AI-adoption study. Predictions are decision support, not a substitute for manager judgment.")
